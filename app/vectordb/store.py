"""MILESTONE 5 — Vector store (persistent, hybrid-ready).

Stores each chunk together with its dense vector on disk, and can find the
nearest chunks to a query vector by cosine similarity.

It ALSO keeps a BM25 keyword index over the same chunk texts. The dense side
answers "what means the same?"; the BM25 side answers "what contains these exact
words?". The retriever (M7) fuses both for 'hybrid RAG' — but the fusion lives
there, because only the retriever has the query *text* that BM25 needs.

Everything is written under `persist_dir` (default settings.vector_dir) as a
single JSON file, so a brand-new store opened on the same folder sees all the data.

Contract checked by the evaluator:
  - search(v, k): <= k hits, scores descending, best first
  - searching a stored chunk's own vector returns that chunk first
  - add() upserts by chunk.id (no duplicates)
  - reopening the same persist_dir sees the same data
  - delete_document(doc_id) removes every chunk of that doc
  - searching an empty store returns []
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Protocol

from app.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]: ...
    def delete_document(self, doc_id: str) -> int: ...
    def count(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: 1.0 = identical direction, 0.0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, used by the BM25 keyword index."""
    return _TOKEN.findall(text.lower())


class PersistentVectorStore:
    """A small JSON-backed store: fine for a training-scale corpus (thousands of chunks)."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "store.json"
        self._records: dict[str, dict] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {r["id"]: r for r in data.get("records", [])}

    def _save(self) -> None:
        payload = {"records": list(self._records.values())}
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _to_chunk(self, rec: dict) -> Chunk:
        """Rebuild a full Chunk object from a stored record (vector left out)."""
        return Chunk(
            id=rec["id"], doc_id=rec["doc_id"], text=rec["text"],
            source=rec["source"], position=rec["position"], metadata=rec.get("metadata", {}),
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or REPLACE chunks keyed by chunk.id (upsert -> no duplicates)."""
        for chunk, vector in zip(chunks, embeddings, strict=False):
            rec = chunk.model_dump()
            rec["vector"] = list(vector)
            self._records[chunk.id] = rec
        self._bm25 = None
        self._save()

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to `doc_id`. Returns how many were removed."""
        ids = [cid for cid, r in self._records.items() if r["doc_id"] == doc_id]
        for cid in ids:
            del self._records[cid]
        if ids:
            self._bm25 = None
            self._save()
        return len(ids)

    def count(self) -> int:
        return len(self._records)

    def sources(self) -> list[str]:
        """Every indexed file name (chunk.source), sorted: what the document picker offers."""
        return sorted({r["source"] for r in self._records.values()})

    def vectors_for(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Stored dense vectors for these chunk ids (used to score relevance)."""
        return {cid: self._records[cid]["vector"] for cid in chunk_ids if cid in self._records}

    def search(
        self, query_embedding: list[float], k: int, sources: Collection[str] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to k chunks most similar to the query vector, best first.

        `sources` (optional) limits the search to chunks of those files (the document picker).
        """
        if not self._records:
            return []
        scored = [
            RetrievedChunk(chunk=self._to_chunk(r), score=_cosine(query_embedding, r["vector"]))
            for r in self._records.values()
            if not sources or r["source"] in sources
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def _ensure_bm25(self) -> None:
        """Build the BM25 index over current chunk texts if it is stale."""
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._bm25_ids = list(self._records.keys())
        corpus = [_tokenize(self._records[i]["text"]) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search_bm25(self, query: str, k: int, sources: Collection[str] | None = None) -> list[RetrievedChunk]:
        """Return up to k chunks ranked by BM25 keyword overlap with the query text.

        With `sources`, only chunks of those files are ranked (scores still use the whole corpus).
        """
        if not self._records:
            return []
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        pairs = zip(self._bm25_ids, scores, strict=False)
        if sources:
            pairs = ((cid, sc) for cid, sc in pairs if self._records[cid]["source"] in sources)
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        hits = [
            RetrievedChunk(chunk=self._to_chunk(self._records[cid]), score=float(score))
            for cid, score in ranked[:k]
        ]
        return hits


def get_vector_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Open (or create) the store under `persist_dir` (default: settings.vector_dir)."""
    if persist_dir is None:
        from app.config import get_settings

        persist_dir = get_settings().vector_dir
    return PersistentVectorStore(persist_dir)
