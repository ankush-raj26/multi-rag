"""FastAPI entry point.  Run with:  uv run uvicorn app.main:app --reload"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import web
from app.api import chat, documents, route_preview, sessions
from app.config import Settings, get_settings
from app.observability import configure_tracing
from app.storage.json_store import SessionStore

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own Settings that point at temp folders."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    application = FastAPI(title="RAG Chatbot (training scaffold)")
    application.state.settings = settings
    application.state.session_store = SessionStore(settings.sessions_dir)
    application.state.components = None  # built lazily by app.deps.get_components
    application.state.tracing_enabled = configure_tracing()

    application.include_router(sessions.router)
    application.include_router(chat.router)
    application.include_router(documents.router)
    application.include_router(route_preview.router)
    application.include_router(web.router)
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {"status": "ok", "tracing": application.state.tracing_enabled}

    return application


logging.basicConfig(level=logging.INFO)
app = create_app()
