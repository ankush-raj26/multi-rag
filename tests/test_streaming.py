from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.chat.engine import get_engine
from app.deps import Components, get_components
from app.models import EngineResult
from tests.fakes import FakeChatModel


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_engine_stream_yields_route_tokens_and_result() -> None:
    engine = get_engine(llm=FakeChatModel(reply="streamed answer"), retriever=None, tools=[])
    events = list(engine.stream(message="hi", history=[], session_id="s1"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "route" and kinds[-1] == "result"
    assert "".join(d for k, d in events if k == "token") == "streamed answer"
    result = events[-1][1]
    assert isinstance(result, EngineResult)
    assert result.answer == "streamed answer"
    assert result.route is not None and result.route.route == "none"


def test_stream_endpoint_sends_events_and_saves_turn(client: TestClient) -> None:
    engine = get_engine(llm=FakeChatModel(reply="hello back"), retriever=None, tools=[])
    client.app.dependency_overrides[get_components] = lambda: Components(engine=engine)
    sid = client.post("/api/sessions").json()["id"]

    response = client.post(f"/ui/sessions/{sid}/stream", data={"content": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "route" and kinds[-1] == "done"
    assert "".join(d["text"] for k, d in events if k == "token") == "hello back"
    assert "hello back" in events[-1][1]["html"]

    messages = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["content"] for m in messages] == ["hi", "hello back"]
    assert messages[1]["route"]["route"] == "none"


def test_stream_endpoint_works_with_engines_without_stream(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["id"]   # default components: placeholder engine
    events = _events(client.post(f"/ui/sessions/{sid}/stream", data={"content": "hi"}).text)
    assert events[-1][0] == "done"
    assert len(client.get(f"/api/sessions/{sid}").json()["messages"]) == 2


def test_stream_unknown_session_is_404(client: TestClient) -> None:
    assert client.post(f"/ui/sessions/{'0' * 32}/stream", data={"content": "x"}).status_code == 404
