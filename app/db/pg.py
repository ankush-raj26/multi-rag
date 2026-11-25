"""Postgres access: one connection pool per process, and a tiny SQL-file migration runner.

    uv run python -m app.db migrate      apply app/db/migrations/*.sql in order (idempotent)
    uv run python -m app.db check        JSON health report used by start.sh

Nothing here is ORM-based on purpose: the schema is three small tables and the
queries are short, so plain SQL stays readable for everyone on the team.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, cast

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_pools: dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def get_pool(database_url: str | None = None) -> ConnectionPool:
    """A shared pool for `database_url` (default: settings). Opened on first use."""
    url = database_url or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set (run docker compose up -d and see .env.example)")
    with _lock:
        pool = _pools.get(url)
        if pool is None:
            pool = ConnectionPool(url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True)
            _pools[url] = pool
    return pool


def migrate(database_url: str | None = None) -> list[str]:
    """Apply every migration file that is not yet recorded. Returns the names applied."""
    applied: list[str] = []
    pool = get_pool(database_url)
    with pool.connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
        )
        rows = cast(list[dict[str, Any]], conn.execute("SELECT name FROM schema_migrations").fetchall())
        done = {r["name"] for r in rows}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
        conn.commit()
    if applied:
        logger.info("Applied migrations: %s", ", ".join(applied))
    return applied


def ping(database_url: str | None = None) -> bool:
    try:
        with get_pool(database_url).connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def chunk_count(conn: Connection) -> int:
    row = cast(dict[str, Any] | None, conn.execute("SELECT count(*) AS n FROM chunks").fetchone())
    return int(row["n"]) if row else 0
