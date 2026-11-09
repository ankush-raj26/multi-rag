"""MILESTONE 7 — Retriever (hybrid RAG).

Turns a question into the most relevant chunks. Three modes (RETRIEVAL_MODE):
  dense   — semantic only (embed the query, cosine search)      = "traditional RAG"
  bm25    — keyword only (exact word overlap)
  hybrid  — run BOTH and fuse with Reciprocal Rank Fusion (RRF) = "hybrid RAG" (default)

Hybrid wins in practice: dense finds paraphrases ("vacation" ~ "annual leave"),
BM25 nails exact terms, IDs and numbers ("Form 16", "clause 4.2"). RRF needs no
score calibration — it just rewards chunks that rank high on either list.

`retrieve` is decorated with @traceable(run_type="retriever") so it shows up as a
proper "retriever" span in LangSmith (a requirement of M2 + M7).

RELEVANCE THRESHOLD (`retrieve_mode`, the path the chatbot uses):
Ranking always returns *something*, even for "trains from Varanasi to Jodhpur", and the
hybrid score is a rank-fusion number (~0.03) that says nothing about relevance. So every
routed hit is re-scored by its real cosine similarity to the question and dropped if it is
below the floor. Two numbers, calibrated on the live corpus:
  gate   the BEST passage must reach it, or the question isn't about the documents at all
         (stops vague near-misses, e.g. typo-heavy off-topic questions, at ~0.57)
  floor  each supporting passage must reach it (keeps useful context at ~0.55-0.6)
If nothing passes, the bot answers without documents and shows no sources.
`retrieve()` itself stays threshold-free: the milestone contract expects k results.
"""

from __future__ import annotations

from typing import Protocol

from langsmith import traceable

from app.models import RetrievedChunk
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore, _cosine

_RRF_K = 60  # standard RRF constant; dampens the weight of lower ranks


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        ...


def _rrf(result_lists: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score each chunk by sum of 1/(RRF_K + rank) across lists."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunks[cid] = hit
    fused = [RetrievedChunk(chunk=chunks[cid].chunk, score=score) for cid, score in scores.items()]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:k]


class HybridRetriever:
    """Embeds the query, searches the store, and (in hybrid mode) fuses with BM25."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        k: int,
        mode: str,
        min_score: float | None = None,
        min_top_score: float | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k
        self._mode = mode
        # None -> the embedder's own scale (bge-small ~0.55, hash ~0.1); 0 disables the filter
        self.min_score = min_score if min_score is not None else float(getattr(embedder, "relevance_floor", 0.0))
        gate = min_top_score if min_top_score is not None else getattr(embedder, "relevance_gate", None)
        self.min_top_score = max(float(gate), self.min_score) if gate is not None else self.min_score

    @traceable(run_type="retriever")  # <- makes this a "retriever" run in LangSmith
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        top_k = k or self._k
        # BM25 is only possible if the store exposes it (the test's fake store does not).
        can_bm25 = hasattr(self._store, "search_bm25")
        mode = self._mode if (self._mode != "hybrid" or can_bm25) else "dense"

        if mode == "bm25" and can_bm25:
            return self._store.search_bm25(query, top_k)  # type: ignore[attr-defined]

        # Pull a slightly wider pool than needed so fusion/ranking has room to work.
        pool = max(top_k * 4, 10)
        dense_hits = self._store.search(self._embedder.embed_query(query), pool)
        if mode == "dense":
            return dense_hits[:top_k]

        # hybrid: fuse dense + BM25
        bm25_hits = self._store.search_bm25(query, pool)  # type: ignore[attr-defined]
        return _rrf([dense_hits, bm25_hits], top_k)


    @traceable(run_type="retriever", name="retrieve_routed")
    def retrieve_mode(
        self, query: str, mode: str, k: int | None = None, page: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve with the mode the ROUTER picked (dense/bm25/hybrid/page).

        For 'page', run hybrid over a wider pool and move chunks from the asked-for
        page to the front, so "what's on page 3" answers from page 3.
        """
        top_k = k or self._k
        if mode not in ("dense", "bm25", "hybrid", "page"):
            mode = "hybrid"
        saved = self._mode
        self._mode = "hybrid" if mode == "page" else mode
        try:
            hits = self.retrieve(query, k=top_k * 4 if mode == "page" else top_k)
        finally:
            self._mode = saved
        if mode == "page" and page is not None:
            on_page = [h for h in hits if h.chunk.metadata.get("page") == page]
            others = [h for h in hits if h.chunk.metadata.get("page") != page]
            hits = on_page + others
        return self._keep_relevant(query, hits, page)[:top_k]

    def _keep_relevant(self, query: str, hits: list[RetrievedChunk], page: int | None) -> list[RetrievedChunk]:
        """Score each hit by cosine(question, chunk) and drop the ones below `min_score`.

        The score is stored on the chunk as metadata["relevance"] (shown in the UI);
        the hit's own rank score is left untouched. A chunk from the page the user
        explicitly asked for is always kept: "what is on page 4" names its answer.
        """
        if not hits:
            return []
        q = self._embedder.embed_query(query)
        ids = [h.chunk.id for h in hits]
        vectors = self._store.vectors_for(ids) if hasattr(self._store, "vectors_for") else {}
        missing = [h for h in hits if h.chunk.id not in vectors]
        if missing:   # stores without stored vectors: embed just those chunks, in one batch
            vectors.update(zip((h.chunk.id for h in missing),
                               self._embedder.embed_documents([h.chunk.text for h in missing]), strict=True))
        rels = {h.chunk.id: round(_cosine(q, vectors[h.chunk.id]), 4) for h in hits}
        named = {h.chunk.id for h in hits if page is not None and h.chunk.metadata.get("page") == page}
        # gate: unless the user named a page, the best passage must clearly be about the question
        if not named and max(rels.values()) < self.min_top_score:
            return []
        kept: list[RetrievedChunk] = []
        for h in hits:
            rel = rels[h.chunk.id]
            if rel >= self.min_score or h.chunk.id in named:
                chunk = h.chunk.model_copy(update={"metadata": {**h.chunk.metadata, "relevance": rel}})
                kept.append(RetrievedChunk(chunk=chunk, score=h.score))
        return kept


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
) -> Retriever:
    """Build a retriever from the store/embedder it is GIVEN (defaults from .env)."""
    from app.config import get_settings

    settings = get_settings()
    if store is None:
        from app.vectordb.store import get_vector_store

        store = get_vector_store()
    if embedder is None:
        from app.vectordb.embeddings import get_embedder

        embedder = get_embedder()
    mode = getattr(settings, "retrieval_mode", "hybrid")
    return HybridRetriever(
        store=store, embedder=embedder, k=k or settings.top_k, mode=mode,
        min_score=getattr(settings, "retrieval_min_score", None),
        min_top_score=getattr(settings, "retrieval_min_top_score", None),
    )
