"""Relevance threshold: off-topic questions must not drag in (or show) unrelated chunks."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.chat.engine import OUT_OF_SCOPE_REPLY, get_engine
from app.chat.router import QueryRouter
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


def test_gate_rejects_a_weak_best_match_that_clears_the_floor() -> None:
    r = _retriever()
    q = "how many days of annual leave do employees get"
    strict = HybridRetriever(store=r._store, embedder=r._embedder, k=3, mode="dense", min_score=0.1, min_top_score=0.9)
    assert strict.retrieve_mode(q, "hybrid") == []          # best match below the gate -> nothing
    normal = HybridRetriever(store=r._store, embedder=r._embedder, k=3, mode="dense", min_score=0.1, min_top_score=0.35)
    assert normal.retrieve_mode(q, "hybrid")                # best match clears it -> passages kept


def test_off_topic_question_gets_the_scope_reply_without_asking_the_model() -> None:
    llm = FakeChatModel(reply="Paris is the capital of France.")
    engine = get_engine(llm=llm, retriever=_retriever(), tools=[])
    result = engine.invoke(message="show me trains from Varanasi to Jodhpur today", history=[], session_id="s")
    assert result.answer == OUT_OF_SCOPE_REPLY and result.sources == []
    assert "Paris" not in result.answer


def test_follow_up_uses_the_previous_question_before_declining() -> None:
    from app.models import Message

    engine = get_engine(llm=FakeChatModel(reply="Part-timers get it pro rata."), retriever=_retriever(), tools=[])
    history = [Message(role="user", content="how many days of annual leave do employees get"),
               Message(role="assistant", content="24 days.")]
    result = engine.invoke(message="and for part timers?", history=history, session_id="s")
    assert result.answer != OUT_OF_SCOPE_REPLY and result.sources


def test_deep_agent_is_not_started_for_off_topic_questions() -> None:
    engine = get_engine(llm=FakeChatModel(), retriever=_retriever(), tools=[])
    out = engine._deep_agent({"messages": [HumanMessage(content="tell me the AQI of Delhi and its street food")]})
    assert out["out_of_scope"] is True and out["messages"][0].content == OUT_OF_SCOPE_REPLY


def test_math_and_clock_questions_skip_document_search() -> None:
    r = QueryRouter()
    tool_questions = ["what is 18 * 24?", "100/7", "tell me time right now in delhi", "what is today's date",
                      "what time is it"]
    for q in tool_questions:
        assert r.route(q).route == "none", q
    doc_questions = ["how many annual leave days do I get", "what is the time off policy",
                     "expense claims older than 90 days"]
    for q in doc_questions:
        assert r.route(q).route != "none", q
