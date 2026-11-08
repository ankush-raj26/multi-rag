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
        self.path = self.dir / "store.json"          # the one file that holds everything
        # id -> record; a record is the Chunk fields plus its "vector".
        self._records: dict[str, dict] = {}
        self._bm25 = None                            # lazily (re)built BM25 index
        self._bm25_ids: list[str] = []               # row order of the BM25 index
        self._load()                                 # read any existing data on open

    # ----- persistence ----------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {r["id"]: r for r in data.get("records", [])}

    def _save(self) -> None:
        # Write to a temp file then atomically replace, so a crash never leaves a
        # half-written store (same trick the session storage uses).
        payload = {"records": list(self._records.values())}
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ----- helpers --------------------------------------------------------
    def _to_chunk(self, rec: dict) -> Chunk:
        """Rebuild a full Chunk object from a stored record (vector left out)."""
        return Chunk(
            id=rec["id"], doc_id=rec["doc_id"], text=rec["text"],
            source=rec["source"], position=rec["position"], metadata=rec.get("metadata", {}),
        )

    # ----- write API ------------------------------------------------------
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or REPLACE chunks keyed by chunk.id (upsert -> no duplicates)."""
        for chunk, vector in zip(chunks, embeddings, strict=False):
            rec = chunk.model_dump()        # all Chunk fields as a dict
            rec["vector"] = list(vector)    # store its dense vector alongside
            self._records[chunk.id] = rec   # same id overwrites -> no duplicate
        self._bm25 = None                   # corpus changed -> BM25 must rebuild
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

    # ----- dense search (the M5 contract) --------------------------------
    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """Return up to k chunks most similar to the query vector, best first."""
        if not self._records:               # empty store -> []
            return []
        scored = [
            RetrievedChunk(chunk=self._to_chunk(r), score=_cosine(query_embedding, r["vector"]))
            for r in self._records.values()
        ]
        scored.sort(key=lambda h: h.score, reverse=True)  # highest cosine first
        return scored[:k]

    # ----- BM25 keyword search (used by hybrid retrieval, M7) -------------
    def _ensure_bm25(self) -> None:
        """Build the BM25 index over current chunk texts if it is stale."""
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._bm25_ids = list(self._records.keys())             # fix a row order
        corpus = [_tokenize(self._records[i]["text"]) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search_bm25(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to k chunks ranked by BM25 keyword overlap with the query text."""
        if not self._records:
            return []
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))        # one score per chunk
        ranked = sorted(zip(self._bm25_ids, scores, strict=False), key=lambda p: p[1], reverse=True)
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
