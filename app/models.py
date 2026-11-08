"""Shared data types. These are part of the CONTRACT — do not rename fields.

The trainer's evaluator builds these objects directly, so every module you
write must accept and return exactly these shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- RAG types


class Document(BaseModel):
    """A whole source file after loading. `id` must be stable for the same file."""

    id: str
    text: str
    source: str  # file name, e.g. "leave_policy.md"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A piece of a Document. `position` is 0, 1, 2 ... in document order."""

    id: str  # must be unique and stable, e.g. f"{doc_id}:{position}"
    doc_id: str
    text: str
    source: str
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A search hit. Higher `score` = more relevant."""

    chunk: Chunk
    score: float


class Source(BaseModel):
    """What the UI shows under an answer so users can see where it came from."""

    source: str
    chunk_id: str
    score: float | None = None
    snippet: str = ""
    page: int | None = None  # PDF page number (page-index RAG), None for md/txt


class RouteInfo(BaseModel):
    """Which retrieval route the 3-layer router picked, and why (shown in the UI)."""

    route: str            # none | dense | bm25 | hybrid | page | agentic
    layer: str            # regex | semantic | llm | fallback
    reason: str = ""


class EngineResult(BaseModel):
    """What `ChatEngine.invoke` returns."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    route: RouteInfo | None = None  # optional extra: the router's decision


# --------------------------------------------------------------------------- chat sessions


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=utc_now)
    sources: list[Source] = Field(default_factory=list)
    route: RouteInfo | None = None  # saved so the badge still shows after a reload


class Session(BaseModel):
    id: str
    title: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    messages: list[Message] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: str
    message_count: int
