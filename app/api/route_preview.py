"""POST /api/route — preview the router's decision for a question, without answering it.

Powers the landing page (/welcome): the visitor types a question and the page
draws where the REAL router sends it. Same `decide()` the chat uses, so the
drawing can never disagree with what the bot will actually do.

It never calls the LLM to answer. It may call it once for router layer 3 (the
LLM picker) when layers 1-2 are unsure, exactly like a chat turn would.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.chat.router import QueryRouter, RouteDecision
from app.deps import Components, get_components

router = APIRouter(prefix="/api", tags=["router"])


class RouteQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class PreviewHit(BaseModel):
    source: str
    chunk_id: str
    page: int | None = None
    score: float
    snippet: str


class RoutePreview(BaseModel):
    route: str             # none | dense | bm25 | hybrid | page | agentic
    layer: str             # regex | semantic | llm | fallback
    reason: str
    page: int | None = None
    picker_enabled: bool   # is router layer 3 switched on?
    semantic_enabled: bool  # is router layer 2 available (an embedder exists)?
    hits: list[PreviewHit] = Field(default_factory=list)  # what retrieval would find
    hits_mode: str = ""    # which retrieval mode produced `hits`
    index_chunks: int = 0  # chunks currently in the vector store


def _decide(components: Components, query: str) -> tuple[RouteDecision, bool]:
    """Use the live engine's router when it exists, else a router without the LLM."""
    engine = components.engine
    if hasattr(engine, "decide"):
        return engine.decide(query), bool(getattr(engine, "picker_enabled", False))
    fallback = QueryRouter(llm=None, embedder=components.embedder, use_llm=False)
    return fallback.route(query), False


@router.post("/route", response_model=RoutePreview)
def preview_route(body: RouteQuery, components: Components = Depends(get_components)) -> RoutePreview:
    decision, picker = _decide(components, body.query)
    preview = RoutePreview(
        route=decision.route,
        layer=decision.layer,
        reason=decision.reason,
        page=decision.page,
        picker_enabled=picker,
        semantic_enabled=components.embedder is not None,
        index_chunks=components.store.count() if components.store is not None else 0,
    )
    retriever = components.retriever
    if decision.route == "none" or retriever is None:
        return preview

    # The agent decides its own searches at answer time; show the hybrid hits it starts from.
    mode = decision.route if decision.route in ("dense", "bm25", "hybrid", "page") else "hybrid"
    if hasattr(retriever, "retrieve_mode"):
        hits = retriever.retrieve_mode(body.query, mode, k=3, page=decision.page)
    else:
        hits = retriever.retrieve(body.query, k=3)
    preview.hits_mode = mode
    preview.hits = [
        PreviewHit(
            source=h.chunk.source,
            chunk_id=h.chunk.id,
            page=h.chunk.metadata.get("page"),
            score=round(h.score, 4),
            snippet=h.chunk.text[:220],
        )
        for h in hits
    ]
    return preview
