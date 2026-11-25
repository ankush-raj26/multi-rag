"""FastAPI entry point.  Run with:  uv run uvicorn app.main:app --reload"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import web
from app.api import chat, documents, route_preview, sessions
from app.auth.router import COOKIE
from app.auth.router import router as auth_router
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
    application.state.components = None
    application.state.tracing_enabled = configure_tracing()
    application.state.auth = _build_auth(settings)

    if application.state.auth is not None:
        application.middleware("http")(_require_login)

    application.include_router(auth_router)
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


def _build_auth(settings: Settings):  # type: ignore[no-untyped-def]
    """Accounts need Postgres + Redis. Without them (tests, the old single-user mode) there is no auth."""
    if not settings.auth_enabled:
        return None
    from app.auth.email import get_mailer
    from app.auth.service import AuthService, PostgresUserRepo
    from app.db.kv import get_kv
    from app.db.pg import get_pool, migrate

    migrate(settings.database_url)
    return AuthService(
        users=PostgresUserRepo(get_pool(settings.database_url)),
        kv=get_kv(settings.redis_url),
        mailer=get_mailer(settings),
        base_url=settings.app_base_url,
        session_ttl_days=settings.session_ttl_days,
    )


# Paths anyone may open without an account. Everything else needs a login cookie.
PUBLIC_PREFIXES = ("/auth", "/static/", "/health", "/welcome", "/api/route", "/docs", "/openapi.json", "/redoc")


async def _require_login(request: Request, call_next):  # type: ignore[no-untyped-def]
    auth = request.app.state.auth
    request.state.user = auth.user_for_token(request.cookies.get(COOKIE))
    path = request.url.path
    if request.state.user is None and not any(path == p.rstrip("/") or path.startswith(p) for p in PUBLIC_PREFIXES):
        if path.startswith("/api/") or request.headers.get("HX-Request"):
            return JSONResponse({"detail": "Sign in required"}, status_code=401, headers={"HX-Redirect": "/auth"})
        return RedirectResponse(f"/auth?next={path}", status_code=303)
    return await call_next(request)


logging.basicConfig(level=logging.INFO)
app = create_app()
