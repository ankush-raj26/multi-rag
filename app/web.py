"""Server-rendered pages (Jinja2 + HTMX). Complete — no TODOs here."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import AsyncIterator, Generator
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.deps import Components, get_components, get_store
from app.services import UploadError, run_chat_turn, save_and_index_upload, stream_chat_turn
from app.storage.json_store import SessionNotFoundError, SessionStore

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)


def _redirect(request: Request, url: str) -> Response:
    if request.headers.get("HX-Request"):
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _render_page(request: Request, store: SessionStore, components: Components, session_id: str | None) -> HTMLResponse:
    current = None
    if session_id:
        try:
            current = store.get(session_id)
        except SessionNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    settings: Settings = request.app.state.settings
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": store.list(),
            "current": current,
            "missing": components.missing,
            "pdfs": sorted(p.name for p in settings.documents_dir.glob("*.pdf")),
            # all indexed files: the empty chat builds its starter questions from these
            "documents": sorted(
                p.name for p in settings.documents_dir.glob("*") if p.suffix.lower() in settings.allowed_upload_suffixes
            ),
        },
    )


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> Response:
    sessions = store.list()
    if sessions:
        return RedirectResponse(f"/sessions/{sessions[0].id}", status_code=status.HTTP_303_SEE_OTHER)
    return _render_page(request, store, components, None)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_page(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    return _render_page(request, store, components, session_id)


@router.get("/welcome", response_class=HTMLResponse)
def welcome(request: Request, components: Components = Depends(get_components)) -> HTMLResponse:
    """The scroll-driven landing page: shows the router deciding, ends in the live bot."""
    settings: Settings = request.app.state.settings
    return templates.TemplateResponse(
        request,
        "welcome.html",
        {
            "chunks": components.store.count() if components.store is not None else 0,
            "documents": sorted(
                p.name for p in settings.documents_dir.glob("*") if p.suffix.lower() in settings.allowed_upload_suffixes
            ),
            "ready": not components.missing,
        },
    )


@router.post("/ui/start")
def ui_start(
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> Response:
    """No-JavaScript path for the landing page's ask form: new chat, first turn, open it."""
    session = store.create()
    run_chat_turn(store, components.engine, session.id, content)
    return RedirectResponse(f"/sessions/{session.id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/files/{name}")
def serve_document(name: str, request: Request) -> FileResponse:
    """Serve a document inline so an <iframe ...#page=N> can display a chosen page."""
    settings: Settings = request.app.state.settings
    safe = Path(name).name  # strip any path parts -> no directory traversal
    path = settings.documents_dir / safe
    if not path.exists() or path.suffix.lower() not in settings.allowed_upload_suffixes:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return FileResponse(path, headers={"Content-Disposition": "inline"})


@router.post("/ui/sessions")
def ui_create_session(request: Request, store: SessionStore = Depends(get_store)) -> Response:
    session = store.create()
    return _redirect(request, f"/sessions/{session.id}")


@router.post("/ui/sessions/{session_id}/rename")
def ui_rename_session(
    session_id: str,
    request: Request,
    title: str = Form(...),
    store: SessionStore = Depends(get_store),
) -> Response:
    try:
        store.rename(session_id, title)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, f"/sessions/{session_id}")


@router.post("/ui/sessions/{session_id}/delete")
def ui_delete_session(session_id: str, request: Request, store: SessionStore = Depends(get_store)) -> Response:
    try:
        store.delete(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, "/")


@router.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse)
def ui_send_message(
    session_id: str,
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return templates.TemplateResponse(
        request, "partials/messages.html", {"messages": [user_message, assistant_message]}
    )


@router.post("/ui/sessions/{session_id}/stream")
def ui_stream_message(
    session_id: str,
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> StreamingResponse:
    """One chat turn as Server-Sent Events, so the page can show the answer as it is written.

    event: route  {"route", "layer", "reason"}   the router's decision (the panel animates it)
    event: token  {"text"}                       the next piece of the answer
    event: done   {"html"}                       both saved messages, rendered like the page does
    """
    try:
        events = stream_chat_turn(store, components.engine, session_id, content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    partial = templates.get_template("partials/messages.html")

    def sse(kind: str, data: Any) -> str:
        if kind == "route":
            payload = data.model_dump()
        elif kind == "token":
            payload = {"text": data}
        else:
            payload = {"html": partial.render(messages=list(data))}
        return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        _until_disconnect(request, events, sse),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _until_disconnect(
    request: Request, events: Generator[tuple[str, Any], None, None], fmt: Any
) -> AsyncIterator[str]:
    """Run the chat turn in a worker thread and stream its events.

    When the browser goes away (Stop button, closed tab) the worker stops at the next
    event and closes the turn itself, so the half-written answer is still saved.
    """
    out: queue.Queue[str | None] = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            for kind, data in events:
                if stop.is_set():
                    break
                out.put(fmt(kind, data))
        finally:
            events.close()   # after a stop: the turn saves what it has
            out.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def stream() -> AsyncIterator[str]:
        try:
            while (chunk := await anyio.to_thread.run_sync(out.get)) is not None:
                # uvicorn drops writes to a closed connection silently, so ask explicitly
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            stop.set()

    return stream()


@router.post("/ui/documents", response_class=HTMLResponse)
async def ui_upload_document(
    request: Request,
    file: UploadFile,
    components: Components = Depends(get_components),
) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        result = save_and_index_upload(settings, components, file.filename or "", data)
        message, ok = f"{result.filename}: {result.detail}", True
    except UploadError as exc:
        message, ok = str(exc), False
    return templates.TemplateResponse(request, "partials/upload_status.html", {"message": message, "ok": ok})
