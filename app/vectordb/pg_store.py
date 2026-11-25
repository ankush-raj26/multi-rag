"""Vector store on Postgres + pgvector (the production replacement for the JSON file store).

Same contract as `PersistentVectorStore`: add (upsert by id), search (cosine, best first),
search_bm25 (keyword side of hybrid retrieval, here Postgres full-text search ranked with
ts_rank_cd), delete_document, count, sources, vectors_for.

Why Postgres for vectors too: the app already needs it for users and chat history, so one
container holds everything, and `docker compose down -v` is the only cleanup anyone needs.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.models import Chunk, RetrievedChunk


def _clean(text: str) -> str:
    """Postgres text cannot hold NUL bytes; PDFs sometimes carry them."""
    return text.replace("\x00", "")


def _floats(value: Any) -> list[float]:
    """pgvector hands back a numpy array or a `Vector`, depending on the adapter: normalise."""
    if hasattr(value, "to_list"):
        value = value.to_list()
    return [float(x) for x in value]


def _to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        text=row["text"],
        source=row["source"],
        position=row["position"],
        metadata=row.get("metadata") or {},
    )


class PgVectorStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def _conn(self):  # type: ignore[no-untyped-def]
        cm = self._pool.connection()
        conn = cm.__enter__()
        register_vector(conn)
        return cm, conn

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import numpy as np

        cm, conn = self._conn()
        try:
            with conn.cursor() as cur:
                for chunk, vector in zip(chunks, embeddings, strict=False):
                    cur.execute(
                        """INSERT INTO chunks (id, doc_id, source, position, text, metadata, embedding)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, source = EXCLUDED.source,
                               position = EXCLUDED.position, text = EXCLUDED.text,
                               metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                        (
                            chunk.id,
                            chunk.doc_id,
                            chunk.source,
                            chunk.position,
                            _clean(chunk.text),
                            Jsonb(chunk.metadata),
                            np.asarray(vector, dtype="float32"),
                        ),
                    )
            conn.commit()
        finally:
            cm.__exit__(None, None, None)

    def delete_document(self, doc_id: str) -> int:
        cm, conn = self._conn()
        try:
            cur = conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            cm.__exit__(None, None, None)

    def count(self) -> int:
        cm, conn = self._conn()
        try:
            row = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()
            return int(row["n"]) if row else 0
        finally:
            cm.__exit__(None, None, None)

    def sources(self) -> list[str]:
        cm, conn = self._conn()
        try:
            return [r["source"] for r in conn.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()]
        finally:
            cm.__exit__(None, None, None)

    def vectors_for(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        cm, conn = self._conn()
        try:
            rows = conn.execute("SELECT id, embedding FROM chunks WHERE id = ANY(%s)", (list(chunk_ids),)).fetchall()
            return {r["id"]: _floats(r["embedding"]) for r in rows if r["embedding"] is not None}
        finally:
            cm.__exit__(None, None, None)

    def search(
        self, query_embedding: list[float], k: int, sources: Collection[str] | None = None
    ) -> list[RetrievedChunk]:
        import numpy as np

        cm, conn = self._conn()
        try:
            q = np.asarray(query_embedding, dtype="float32")
            where = "WHERE source = ANY(%s)" if sources else ""
            params: list[Any] = [q, *([list(sources)] if sources else []), q, k]
            rows = conn.execute(
                f"""SELECT id, doc_id, source, position, text, metadata, 1 - (embedding <=> %s) AS score
                    FROM chunks {where} ORDER BY embedding <=> %s LIMIT %s""",
                params,
            ).fetchall()
            return [RetrievedChunk(chunk=_to_chunk(dict(r)), score=float(r["score"])) for r in rows]
        finally:
            cm.__exit__(None, None, None)

    def search_bm25(self, query: str, k: int, sources: Collection[str] | None = None) -> list[RetrievedChunk]:
        """Keyword search: Postgres full-text ranking stands in for BM25 (same role, same shape)."""
        cm, conn = self._conn()
        try:
            where = "AND source = ANY(%s)" if sources else ""
            params: list[Any] = [query, query, *([list(sources)] if sources else []), k]
            rows = conn.execute(
                f"""SELECT id, doc_id, source, position, text, metadata,
                           ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                    FROM chunks WHERE tsv @@ websearch_to_tsquery('english', %s) {where}
                    ORDER BY score DESC LIMIT %s""",
                params,
            ).fetchall()
            return [RetrievedChunk(chunk=_to_chunk(dict(r)), score=float(r["score"])) for r in rows]
        finally:
            cm.__exit__(None, None, None)
