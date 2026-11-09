"""Relevance threshold: off-topic questions must not drag in (or show) unrelated chunks."""

from __future__ import annotations

from app.chat.engine import get_engine
from app.models import Chunk
from app.vectordb.retriever import HybridRetriever
from tests.fakes import FakeChatModel, HashEmbedder, ListStore

TEXTS = [
    "Full-time employees receive 24 days of paid annual leave per year.",
    "Business expenses up to 50 USD may be claimed without a receipt.",
    "Employees may work remotely up to 3 days per week.",
]


def _retriever() -> HybridRetriever:
    store, emb = ListStore(), HashEmbedder()
    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=t, source="handbook.pdf", position=i, metadata={"page": i + 1})
        for i, t in enumerate(TEXTS)
    ]
    store.add(chunks, emb.embed_documents(TEXTS))
    return HybridRetriever(store=store, embedder=emb, k=3, mode="dense", min_score=0.35)


def test_off_topic_question_retrieves_nothing() -> None:
    hits = _retriever().retrieve_mode("show me trains from Varanasi to Jodhpur today", "hybrid")
    assert hits == []


def test_relevant_question_keeps_matches_and_reports_similarity() -> None:
    hits = _retriever().retrieve_mode("how many days of annual leave do employees get", "hybrid")
    assert hits and hits[0].chunk.text == TEXTS[0]
    assert all(h.chunk.metadata["relevance"] >= 0.35 for h in hits)


def test_named_page_is_kept_even_below_threshold() -> None:
    hits = _retriever().retrieve_mode("what is on page 3", "page", page=3)
    assert [h.chunk.metadata["page"] for h in hits] == [3]


def test_plain_retrieve_contract_is_unchanged() -> None:
    assert len(_retriever().retrieve("trains to Jodhpur", k=2)) == 2   # milestone contract: always k


def test_engine_reports_empty_search_and_shows_no_sources() -> None:
    engine = get_engine(llm=FakeChatModel(reply="I don't have that."), retriever=_retriever(), tools=[])
    result = engine.invoke(message="show me trains from Varanasi to Jodhpur today", history=[], session_id="s")
    assert result.sources == []
    assert result.route is not None and result.route.route != "none" and result.route.retrieved == 0


def test_greeting_has_no_search_count() -> None:
    engine = get_engine(llm=FakeChatModel(), retriever=_retriever(), tools=[])
    result = engine.invoke(message="hi", history=[], session_id="s")
    assert result.route is not None and result.route.route == "none" and result.route.retrieved is None
