"""Document picker: a question can be limited to the files the user ticked."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.chat.engine import get_engine
from app.deps import Components, get_components
from app.models import Chunk
from app.services import clean_docs
from app.vectordb.retriever import HybridRetriever
from app.vectordb.store import PersistentVectorStore
from tests.fakes import FakeChatModel, HashEmbedder, ListStore

DOCS = {
    "leave_policy.md": "Full-time employees receive 24 days of paid annual leave per year.",
    "remote_work_policy.md": "Employees may work remotely up to 3 days per week and get a 400 USD stipend.",
    "handbook.pdf": "Annual leave requests are approved by your manager in the HR tool.",
}


def _chunks() -> list[Chunk]:
    return [
        Chunk(id=f"{name}:0", doc_id=name, text=text, source=name, position=0)
        for name, text in DOCS.items()
    ]


def _retriever(store: object) -> HybridRetriever:
    emb = HashEmbedder()
    chunks = _chunks()
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))  # type: ignore[attr-defined]
    return HybridRetriever(store=store, embedder=emb, k=3, mode="hybrid", min_score=0.35)  # type: ignore[arg-type]


def test_store_search_filters_by_source(tmp_path: Path) -> None:
    emb = HashEmbedder()
    store = PersistentVectorStore(tmp_path)
    chunks = _chunks()
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    q = emb.embed_query("annual leave days")
    assert {h.chunk.source for h in store.search(q, 5, sources={"handbook.pdf"})} == {"handbook.pdf"}
    assert {h.chunk.source for h in store.search_bm25("annual leave", 5, sources={"leave_policy.md"})} == {
        "leave_policy.md"
    }
    assert store.sources() == sorted(DOCS)


def test_retriever_only_returns_picked_documents(tmp_path: Path) -> None:
    retriever = _retriever(PersistentVectorStore(tmp_path))
    hits = retriever.retrieve_mode("how many days of annual leave", "hybrid", sources=["handbook.pdf"])
    assert hits and {h.chunk.source for h in hits} == {"handbook.pdf"}


def test_picked_documents_skip_the_relevance_gate(tmp_path: Path) -> None:
    retriever = _retriever(PersistentVectorStore(tmp_path))
    assert retriever.retrieve_mode("summarize it", "hybrid") == []
    hits = retriever.retrieve_mode("summarize it", "hybrid", sources=["remote_work_policy.md"])
    assert [h.chunk.source for h in hits] == ["remote_work_policy.md"]


def test_store_without_filter_support_still_filters() -> None:
    retriever = _retriever(ListStore())
    hits = retriever.retrieve("annual leave", sources=["leave_policy.md"])
    assert hits and {h.chunk.source for h in hits} == {"leave_policy.md"}


def test_engine_tells_the_model_which_documents_were_picked(tmp_path: Path) -> None:
    llm = FakeChatModel(reply="From the remote work policy: 3 days.")
    engine = get_engine(llm=llm, retriever=_retriever(PersistentVectorStore(tmp_path)), tools=[])
    result = engine.invoke(
        message="how many days can I work remotely", history=[], session_id="s", docs=["remote_work_policy.md"]
    )
    assert [s.source for s in result.sources] == ["remote_work_policy.md"]
    assert "remote_work_policy.md" in llm.prompt_text()


def test_clean_docs_strips_paths_and_repeats() -> None:
    assert clean_docs(["../secret/a.md", "a.md", " ", "b.pdf"]) == ["a.md", "b.pdf"]


def test_api_and_stream_save_the_picked_documents(client: TestClient) -> None:
    engine = get_engine(llm=FakeChatModel(reply="ok"), retriever=None, tools=[])
    client.app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    sid = client.post("/api/sessions").json()["id"]

    turn = client.post(f"/api/sessions/{sid}/messages", json={"content": "hi", "docs": ["leave_policy.md"]}).json()
    assert turn["user_message"]["docs"] == ["leave_policy.md"]

    client.post(f"/ui/sessions/{sid}/stream", data={"content": "hi", "docs": ["a.md", "b.pdf"]})
    messages = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert messages[-2]["docs"] == ["a.md", "b.pdf"]
    page = client.get(f"/sessions/{sid}").text
    assert 'class="doc-tag"' in page and "b.pdf" in page
