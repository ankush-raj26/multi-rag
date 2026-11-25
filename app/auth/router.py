"""HTTP side of accounts: the /auth pages and the form/JSON endpoints behind them.

Every POST accepts a normal form post. When the request comes from the page's JavaScript
(`Accept: application/json`) it answers JSON {ok, next|error}; otherwise it redirects or
re-renders the page with the error, so the pages also work without JavaScript.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth.service import AuthError, AuthService, User

COOKIE = "rag_session"
router = APIRouter(include_in_schema=False)


def get_auth(request: Request) -> AuthService:
    service: AuthService = request.app.state.auth
    return service


def current_user(request: Request) -> User | None:
    return getattr(request.state, "user", None)


def _wants_json(request: Request) -> bool:
    return "application/json" in request.headers.get("accept", "")


def _set_cookie(response: Response, request: Request, token: str) -> None:
    ttl = request.app.state.settings.session_ttl_days * 24 * 3600
    response.set_cookie(
        COOKIE, token, max_age=ttl, httponly=True, samesite="lax", path="/", secure=request.url.scheme == "https"
    )


def _page(request: Request, mode: str, **ctx: Any) -> HTMLResponse:
    from app.web import templates

    return templates.TemplateResponse(request, "auth.html", {"mode": mode, **ctx})


def _done(request: Request, next_url: str, token: str | None = None, message: str = "") -> Response:
    response: Response
    if _wants_json(request):
        response = JSONResponse({"ok": True, "next": next_url, "message": message})
    else:
        response = RedirectResponse(next_url, status_code=status.HTTP_303_SEE_OTHER)
    if token:
        _set_cookie(response, request, token)
    return response


def _failed(request: Request, mode: str, error: str, **ctx: Any) -> Response:
    if _wants_json(request):
        return JSONResponse({"ok": False, "error": error}, status_code=status.HTTP_400_BAD_REQUEST)
    return _page(request, mode, error=error, **ctx)


@router.get("/auth", response_class=HTMLResponse)
def auth_page(request: Request, mode: str = "signin") -> Response:
    if current_user(request):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return _page(request, mode if mode in ("signin", "signup", "forgot", "sent") else "signin")


@router.post("/auth/signup")
def signup(
    request: Request,
    email: str = Form(...),
    name: str = Form(""),
    password: str = Form(...),
    auth: AuthService = Depends(get_auth),
) -> Response:
    try:
        user = auth.signup(email, name, password)
    except AuthError as exc:
        return _failed(request, "signup", str(exc), email=email, name=name)
    return _done(request, f"/auth/verify?u={user.id}", message="Code sent")


@router.get("/auth/verify", response_class=HTMLResponse)
def verify_page(request: Request, u: str = "") -> Response:
    return _page(request, "verify", user_id=u)


@router.post("/auth/verify")
def verify(
    request: Request, u: str = Form(...), code: str = Form(...), auth: AuthService = Depends(get_auth)
) -> Response:
    try:
        token = auth.verify(u, code)
    except AuthError as exc:
        return _failed(request, "verify", str(exc), user_id=u)
    return _done(request, "/", token)


@router.post("/auth/resend")
def resend(request: Request, u: str = Form(...), auth: AuthService = Depends(get_auth)) -> Response:
    try:
        auth.resend_code(u)
    except AuthError as exc:
        return _failed(request, "verify", str(exc), user_id=u)
    return _done(request, f"/auth/verify?u={u}", message="A new code is on its way")


@router.post("/auth/signin")
def signin(
    request: Request, email: str = Form(...), password: str = Form(...), auth: AuthService = Depends(get_auth)
) -> Response:
    try:
        user, token = auth.signin(email, password)
    except AuthError as exc:
        return _failed(request, "signin", str(exc), email=email)
    if token is None:
        return _done(request, f"/auth/verify?u={user.id}", message="Confirm your e-mail first")
    return _done(request, "/", token)


@router.post("/auth/forgot")
def forgot(request: Request, email: str = Form(...), auth: AuthService = Depends(get_auth)) -> Response:
    try:
        auth.forgot(email)
    except AuthError as exc:
        return _failed(request, "forgot", str(exc), email=email)
    return _done(request, "/auth?mode=sent", message="If that address has an account, a reset link is on its way")


@router.get("/auth/reset", response_class=HTMLResponse)
def reset_page(request: Request, token: str = "", auth: AuthService = Depends(get_auth)) -> Response:
    if not auth.reset_valid(token):
        return _page(request, "forgot", error="This reset link is invalid or has expired. Request a new one.")
    return _page(request, "reset", token=token)


@router.post("/auth/reset")
def reset(
    request: Request, token: str = Form(...), password: str = Form(...), auth: AuthService = Depends(get_auth)
) -> Response:
    try:
        session_token = auth.reset(token, password)
    except AuthError as exc:
        return _failed(request, "reset", str(exc), token=token)
    return _done(request, "/", session_token)


@router.post("/auth/logout")
def logout(request: Request, auth: AuthService = Depends(get_auth)) -> Response:
    auth.end_session(request.cookies.get(COOKIE))
    response = RedirectResponse("/auth", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(COOKIE, path="/")
    return response
