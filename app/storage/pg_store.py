"""Chat sessions in Postgres, one store per signed-in user.

Same methods as the JSON `SessionStore` (list / create / get / rename / delete / append_messages),
so the web routes and the JSON API do not care which one they got. Every query is scoped by
`user_id`, so one user can never read or delete another user's chats.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.models import Message, Session, SessionSummary, utc_now
from app.storage.json_store import DEFAULT_TITLE, SessionNotFoundError

_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _iso(value: Any) -> str:
    return value.isoformat(timespec="seconds") if hasattr(value, "isoformat") else str(value)


class PostgresSessionStore:
    def __init__(self, pool: ConnectionPool, user_id: str) -> None:
        self._pool = pool
        self.user_id = user_id

    def _check(self, session_id: str) -> str:
        if not _ID_PATTERN.match(session_id):
            raise SessionNotFoundError(session_id)
        return session_id

    def _own(self, conn: Connection, session_id: str) -> dict[str, Any]:
        row = cast(
            dict[str, Any] | None,
            conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = %s AND user_id = %s",
                (self._check(session_id), self.user_id),
            ).fetchone(),
        )
        if row is None:
            raise SessionNotFoundError(session_id)
        return row

    def list(self) -> list[SessionSummary]:
        with self._pool.connection() as conn:
            rows: list[dict[str, Any]] = cast(
                list[dict[str, Any]],
                conn.execute(
                    """SELECT s.id, s.title, s.updated_at,
                          (SELECT count(*) FROM chat_messages m WHERE m.session_id = s.id) AS n
                   FROM chat_sessions s WHERE s.user_id = %s ORDER BY s.updated_at DESC""",
                    (self.user_id,),
                ).fetchall(),
            )
        return [
            SessionSummary(id=r["id"], title=r["title"], updated_at=_iso(r["updated_at"]), message_count=int(r["n"]))
            for r in rows
        ]

    def create(self, title: str | None = None) -> Session:
        session = Session(id=uuid.uuid4().hex, title=(title or "").strip() or DEFAULT_TITLE)
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
                (session.id, self.user_id, session.title, session.created_at, session.updated_at),
            )
            conn.commit()
        return session

    def get(self, session_id: str) -> Session:
        with self._pool.connection() as conn:
            row = self._own(conn, session_id)
            messages = conn.execute(
                "SELECT role, content, timestamp, sources, route, docs FROM chat_messages "
                "WHERE session_id = %s ORDER BY id",
                (session_id,),
            ).fetchall()
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
            messages=[Message.model_validate(dict(m)) for m in messages],
        )

    def rename(self, session_id: str, title: str) -> Session:
        with self._pool.connection() as conn:
            self._own(conn, session_id)
            conn.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE id = %s",
                (title.strip() or DEFAULT_TITLE, utc_now(), session_id),
            )
            conn.commit()
        return self.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            self._own(conn, session_id)
            conn.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            conn.commit()

    def append_messages(self, session_id: str, *messages: Message) -> Session:
        with self._pool.connection() as conn:
            row = self._own(conn, session_id)
            for m in messages:
                conn.execute(
                    """INSERT INTO chat_messages (session_id, role, content, timestamp, sources, route, docs)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        session_id,
                        m.role,
                        m.content,
                        m.timestamp,
                        Jsonb([s.model_dump() for s in m.sources]),
                        Jsonb(m.route.model_dump()) if m.route else None,
                        Jsonb(list(m.docs)),
                    ),
                )
            title = row["title"]
            if title == DEFAULT_TITLE:
                first_user = next((m for m in messages if m.role == "user"), None)
                if first_user:
                    title = first_user.content.strip()[:60] or DEFAULT_TITLE
            conn.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE id = %s", (title, utc_now(), session_id)
            )
            conn.commit()
        return self.get(session_id)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value
