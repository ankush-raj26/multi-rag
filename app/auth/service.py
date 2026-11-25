"""The account rules, independent of HTTP. Tested with `MemoryKV` and a fake user table.

Flows
  sign up   -> user row (unverified) + 6-digit code in KV (15 min) + e-mail  -> verify -> signed in
  sign in   -> bcrypt check -> login session token in KV (SESSION_TTL_DAYS)  -> cookie
  forgot    -> one-time token in KV (30 min) + e-mail link                     -> reset -> signed in

Everything short-lived (codes, tokens, login sessions, rate limits) lives in Redis so it
expires on its own and a restart of the app never logs anyone out.
"""

from __future__ import annotations

import re
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import bcrypt

from app.auth.email import Mailer, code_email, reset_email
from app.db.kv import KV
from app.models import utc_now

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
CODE_TTL = 15 * 60
RESET_TTL = 30 * 60
ATTEMPT_WINDOW = 15 * 60
MAX_ATTEMPTS = 10


class AuthError(ValueError):
    """A message safe to show to the person at the keyboard."""


@dataclass
class User:
    id: str
    email: str
    name: str
    verified: bool


class UserRepo(Protocol):
    def by_email(self, email: str) -> dict[str, Any] | None: ...
    def by_id(self, user_id: str) -> dict[str, Any] | None: ...
    def create(self, user_id: str, email: str, name: str, password_hash: str) -> None: ...
    def set_password(self, user_id: str, password_hash: str) -> None: ...
    def set_verified(self, user_id: str) -> None: ...


class PostgresUserRepo:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def by_email(self, email: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()
        return dict(row) if row else None

    def by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        return dict(row) if row else None

    def create(self, user_id: str, email: str, name: str, password_hash: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash) VALUES (%s, %s, %s, %s)",
                (user_id, email, name, password_hash),
            )
            conn.commit()

    def set_password(self, user_id: str, password_hash: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (password_hash, user_id))
            conn.commit()

    def set_verified(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE users SET verified_at = now() WHERE id = %s AND verified_at IS NULL", (user_id,))
            conn.commit()


def password_problem(password: str) -> str | None:
    """Why `password` is not acceptable, or None when it is.

    TODO(human): decide the password policy. Right now only a minimum length is enforced.
    Think about: minimum length (NIST suggests 8+, length beats complexity rules), a maximum
    (bcrypt only hashes the first 72 bytes), rejecting the e-mail address or very common
    passwords ("password", "12345678"), and whether to demand digits/symbols at all (they
    mostly produce "Password1!"). Return a short sentence a user can act on, or None.
    """
    if len(password) < 8:
        return "Use at least 8 characters."
    return None


def _norm_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL.match(email):
        raise AuthError("Enter a valid e-mail address.")
    return email


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _check(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def _user(row: dict[str, Any]) -> User:
    return User(
        id=row["id"], email=row["email"], name=row.get("name") or "", verified=row.get("verified_at") is not None
    )


class AuthService:
    def __init__(self, users: UserRepo, kv: KV, mailer: Mailer, base_url: str, session_ttl_days: int = 14) -> None:
        self.users = users
        self.kv = kv
        self.mailer = mailer
        self.base_url = base_url.rstrip("/")
        self.session_ttl = session_ttl_days * 24 * 3600

    # --- rate limiting -------------------------------------------------------------
    def _throttle(self, bucket: str, key: str) -> None:
        if self.kv.incr(f"rl:{bucket}:{key}", ATTEMPT_WINDOW) > MAX_ATTEMPTS:
            raise AuthError("Too many attempts. Wait 15 minutes and try again.")

    # --- sign up + e-mail verification ---------------------------------------------
    def _send_code(self, user: User) -> None:
        code = f"{secrets.randbelow(1_000_000):06d}"
        self.kv.set(f"verify:{user.id}", code, CODE_TTL)
        subject, html = code_email(user.name, code)
        if not self.mailer.send(user.email, user.name, subject, html):
            raise AuthError("Could not send the confirmation e-mail. Check the address and try again.")

    def signup(self, email: str, name: str, password: str) -> User:
        """Create (or re-arm) an unverified account and e-mail a 6-digit code. Returns the user."""
        email = _norm_email(email)
        name = (name or "").strip()[:80]
        if problem := password_problem(password):
            raise AuthError(problem)
        self._throttle("signup", email)
        row = self.users.by_email(email)
        if row and row.get("verified_at"):
            raise AuthError("That e-mail already has an account. Sign in instead.")
        if row:  # signed up before but never confirmed: take the new password and send a fresh code
            self.users.set_password(row["id"], _hash(password))
            user = _user({**row, "name": name or row.get("name", "")})
        else:
            user = User(id=uuid.uuid4().hex, email=email, name=name, verified=False)
            self.users.create(user.id, email, name, _hash(password))
        self._send_code(user)
        return user

    def resend_code(self, user_id: str) -> None:
        row = self.users.by_id(user_id)
        if not row or row.get("verified_at"):
            raise AuthError("Nothing to confirm for this account.")
        self._throttle("resend", user_id)
        self._send_code(_user(row))

    def verify(self, user_id: str, code: str) -> str:
        """Check the e-mailed code; on success the account is verified and a login token is returned."""
        self._throttle("verify", user_id)
        expected = self.kv.get(f"verify:{user_id}")
        if not expected or not secrets.compare_digest(expected, (code or "").strip()):
            raise AuthError("That code is wrong or has expired. Request a new one.")
        self.users.set_verified(user_id)
        self.kv.delete(f"verify:{user_id}")
        return self.start_session(user_id)

    # --- sign in / out -------------------------------------------------------------
    def signin(self, email: str, password: str) -> tuple[User, str | None]:
        """(user, token). token is None when the account still needs its e-mail confirmed
        (a fresh code is sent in that case)."""
        email = _norm_email(email)
        self._throttle("signin", email)
        row = self.users.by_email(email)
        if not row or not _check(password or "", row["password_hash"]):
            raise AuthError("Wrong e-mail or password.")
        user = _user(row)
        if not user.verified:
            self._send_code(user)
            return user, None
        return user, self.start_session(user.id)

    def start_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.kv.set(f"sess:{token}", user_id, self.session_ttl)
        return token

    def user_for_token(self, token: str | None) -> User | None:
        if not token:
            return None
        user_id = self.kv.get(f"sess:{token}")
        if not user_id:
            return None
        row = self.users.by_id(user_id)
        return _user(row) if row and row.get("verified_at") else None

    def end_session(self, token: str | None) -> None:
        if token:
            self.kv.delete(f"sess:{token}")

    # --- forgot / reset ------------------------------------------------------------
    def forgot(self, email: str) -> None:
        """Always succeeds from the caller's point of view, so it cannot be used to probe e-mails."""
        email = _norm_email(email)
        self._throttle("forgot", email)
        row = self.users.by_email(email)
        if not row:
            return
        token = secrets.token_urlsafe(32)
        self.kv.set(f"reset:{token}", row["id"], RESET_TTL)
        subject, html = reset_email(row.get("name") or "", f"{self.base_url}/auth/reset?token={token}")
        self.mailer.send(email, row.get("name") or "", subject, html)

    def reset_valid(self, token: str) -> bool:
        return bool(token and self.kv.get(f"reset:{token}"))

    def reset(self, token: str, password: str) -> str:
        """Set the new password and sign the person in. Returns a login token."""
        user_id = self.kv.get(f"reset:{token}") if token else None
        if not user_id:
            raise AuthError("This reset link is invalid or has expired. Request a new one.")
        if problem := password_problem(password):
            raise AuthError(problem)
        self.users.set_password(user_id, _hash(password))
        self.users.set_verified(user_id)  # they proved they own the inbox
        self.kv.delete(f"reset:{token}")
        return self.start_session(user_id)


__all__ = ["AuthError", "AuthService", "PostgresUserRepo", "User", "password_problem", "utc_now"]
