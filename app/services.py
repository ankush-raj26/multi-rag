"""Use-cases shared by the JSON API and the HTML pages. Complete — no TODOs here."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.chat.engine import ChatEngine
from app.config import Settings
from app.deps import Components
from app.ingest.pipeline import ingest_file
from app.models import Message
from app.storage.json_store import SessionStore

logger = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def run_chat_turn(store: SessionStore, engine: ChatEngine, session_id: str, content: str) -> tuple[Message, Message]:
    """Send one user message through the engine and save both messages."""
    session = store.get(session_id)  # raises SessionNotFoundError before we call the engine
    user_message = Message(role="user", content=content)
    try:
        result = engine.invoke(message=content, history=session.messages, session_id=session_id)
        assistant_message = Message(
            role="assistant", content=result.answer, sources=result.sources, route=result.route
        )
    except Exception as exc:
        logger.exception("Engine failed for session %s", session_id)
        assistant_message = Message(role="assistant", content=f"Engine error: {type(exc).__name__}: {exc}")
    store.append_messages(session_id, user_message, assistant_message)
    return user_message, assistant_message


def _engine_events(
    engine: ChatEngine, content: str, history: list[Message], session_id: str
) -> Iterator[tuple[str, Any]]:
    """("route" | "token" | "result", data) from any engine; one without `stream()` answers in one piece."""
    stream = getattr(engine, "stream", None)
    if stream is not None:
        yield from stream(message=content, history=history, session_id=session_id)
        return
    result = engine.invoke(message=content, history=history, session_id=session_id)
    if result.route:
        yield "route", result.route
    yield "token", result.answer
    yield "result", result


def stream_chat_turn(
    store: SessionStore, engine: ChatEngine, session_id: str, content: str
) -> Generator[tuple[str, Any], None, None]:
    """Like `run_chat_turn`, but yields the answer while it is being written.

    Events: ("route", RouteInfo), ("token", str) ..., then ("done", (user_message, assistant_message))
    after both messages are saved. Raises SessionNotFoundError right away for an unknown session.
    """
    session = store.get(session_id)

    def events() -> Generator[tuple[str, Any], None, None]:
        user_message = Message(role="user", content=content)
        parts: list[str] = []
        route = None
        try:
            try:
                result = None
                for kind, data in _engine_events(engine, content, session.messages, session_id):
                    if kind == "result":
                        result = data
                        continue
                    if kind == "route":
                        route = data
                    else:
                        parts.append(data)
                    yield kind, data
                assistant_message = (
                    Message(role="assistant", content=result.answer, sources=result.sources, route=result.route)
                    if result is not None
                    else Message(role="assistant", content="".join(parts), route=route)
                )
            except Exception as exc:
                logger.exception("Engine failed for session %s", session_id)
                assistant_message = Message(role="assistant", content=f"Engine error: {type(exc).__name__}: {exc}")
        except GeneratorExit:
            # the reader stopped (Stop button, closed tab): keep what was written so far
            partial = "".join(parts).strip()
            stopped = (partial + "\n\n" if partial else "") + "(stopped)"
            store.append_messages(session_id, user_message, Message(role="assistant", content=stopped, route=route))
            raise
        store.append_messages(session_id, user_message, assistant_message)
        yield "done", (user_message, assistant_message)

    return events()


class UploadError(ValueError):
    pass


@dataclass
class UploadResult:
    filename: str
    indexed: bool
    chunks_indexed: int
    detail: str


def save_and_index_upload(settings: Settings, components: Components, filename: str, data: bytes) -> UploadResult:
    """Save an uploaded file into DOCUMENTS_DIR and index it if ingestion is available."""
    safe_name = _SAFE_NAME.sub("_", Path(filename or "").name).strip("._")
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or suffix not in settings.allowed_upload_suffixes:
        allowed = ", ".join(settings.allowed_upload_suffixes)
        raise UploadError(f"Unsupported file type. Allowed: {allowed}")
    if len(data) > settings.max_upload_bytes:
        raise UploadError("File too large")
    if not data.strip():
        raise UploadError("File is empty")

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    path = settings.documents_dir / safe_name
    path.write_bytes(data)

    if not components.can_index:
        missing = ", ".join(sorted(k for k in ("chunker", "embedder", "vector_store") if k in components.missing))
        return UploadResult(safe_name, False, 0, f"Saved. Indexing not available yet ({missing}).")
    assert components.chunker and components.embedder and components.store
    try:
        count = ingest_file(path, chunker=components.chunker, embedder=components.embedder, store=components.store)
    except NotImplementedError as exc:
        return UploadResult(safe_name, False, 0, f"Saved. Indexing not available yet ({exc}).")
    return UploadResult(safe_name, True, count, f"Saved and indexed {count} chunks.")
