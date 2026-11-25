"""Account rules, tested without Postgres, Redis or Brevo (memory KV, dict user table, capturing mailer)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.auth.service import AuthError, AuthService, password_problem
from app.db.kv import MemoryKV


class DictUsers:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def by_email(self, email: str) -> dict[str, Any] | None:
        return next((r for r in self.rows.values() if r["email"] == email), None)

    def by_id(self, user_id: str) -> dict[str, Any] | None:
        return self.rows.get(user_id)

    def create(self, user_id: str, email: str, name: str, password_hash: str) -> None:
        self.rows[user_id] = {
            "id": user_id,
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "verified_at": None,
        }

    def set_password(self, user_id: str, password_hash: str) -> None:
        self.rows[user_id]["password_hash"] = password_hash

    def set_verified(self, user_id: str) -> None:
        self.rows[user_id]["verified_at"] = "now"


class CapturingMailer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to_email: str, to_name: str, subject: str, html: str) -> bool:
        self.sent.append((to_email, subject, html))
        return True


@pytest.fixture()
def auth() -> AuthService:
    return AuthService(DictUsers(), MemoryKV(), CapturingMailer(), "http://test")


def _code(mailer: CapturingMailer) -> str:
    match = re.search(r">(\d{6})<", mailer.sent[-1][2])
    assert match, "no 6-digit code in the last e-mail"
    return match.group(1)


def test_signup_verify_signin(auth: AuthService) -> None:
    mailer: CapturingMailer = auth.mailer  # type: ignore[assignment]
    user = auth.signup("Ada@Example.com", "Ada", "correct horse battery")
    assert user.email == "ada@example.com" and not user.verified
    assert mailer.sent[-1][0] == "ada@example.com"

    with pytest.raises(AuthError):
        auth.verify(user.id, "000000")
    token = auth.verify(user.id, _code(mailer))
    assert auth.user_for_token(token).email == "ada@example.com"  # type: ignore[union-attr]

    _, token2 = auth.signin("ada@example.com", "correct horse battery")
    assert token2 and auth.user_for_token(token2)
    auth.end_session(token2)
    assert auth.user_for_token(token2) is None


def test_signin_before_verify_resends_code(auth: AuthService) -> None:
    mailer: CapturingMailer = auth.mailer  # type: ignore[assignment]
    user = auth.signup("bob@example.com", "", "longenoughpw")
    assert len(mailer.sent) == 1
    same, token = auth.signin("bob@example.com", "longenoughpw")
    assert token is None and same.id == user.id and len(mailer.sent) == 2


def test_wrong_password_and_duplicate_signup(auth: AuthService) -> None:
    mailer: CapturingMailer = auth.mailer  # type: ignore[assignment]
    user = auth.signup("cat@example.com", "Cat", "longenoughpw")
    auth.verify(user.id, _code(mailer))
    with pytest.raises(AuthError, match="Wrong"):
        auth.signin("cat@example.com", "nope-nope-nope")
    with pytest.raises(AuthError, match="already"):
        auth.signup("cat@example.com", "Cat", "longenoughpw")


def test_forgot_and_reset(auth: AuthService) -> None:
    mailer: CapturingMailer = auth.mailer  # type: ignore[assignment]
    user = auth.signup("dan@example.com", "Dan", "longenoughpw")
    auth.verify(user.id, _code(mailer))
    auth.forgot("nobody@example.com")  # unknown address: no mail, no error
    assert len(mailer.sent) == 1
    auth.forgot("dan@example.com")
    link = re.search(r'href="([^"]+)"', mailer.sent[-1][2]).group(1)  # type: ignore[union-attr]
    token = link.split("token=")[1]
    assert auth.reset_valid(token)
    with pytest.raises(AuthError):
        auth.reset(token, "short")
    session = auth.reset(token, "a brand new password")
    assert auth.user_for_token(session)
    assert not auth.reset_valid(token), "reset links are single-use"
    _, ok = auth.signin("dan@example.com", "a brand new password")
    assert ok


def test_rate_limit(auth: AuthService) -> None:
    for _ in range(10):
        with pytest.raises(AuthError, match="Wrong"):
            auth.signin("eve@example.com", "x")
    with pytest.raises(AuthError, match="Too many"):
        auth.signin("eve@example.com", "x")


def test_password_policy_minimum() -> None:
    assert password_problem("1234567")
    assert password_problem("12345678") is None
