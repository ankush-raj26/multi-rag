"""Infrastructure commands used by start.sh:

uv run python -m app.db check     prints one JSON line: {"postgres": bool, "redis": bool, "chunks": int, ...}
uv run python -m app.db migrate   applies pending SQL migrations
"""

from __future__ import annotations

import json
import sys

from app.config import get_settings


def _check() -> dict[str, object]:
    settings = get_settings()
    report: dict[str, object] = {
        "database_url": bool(settings.database_url),
        "redis_url": bool(settings.redis_url),
        "postgres": False,
        "redis": False,
        "migrated": False,
        "chunks": 0,
        "brevo": "unset",
    }
    if settings.database_url:
        from app.db.pg import chunk_count, get_pool, ping

        report["postgres"] = ping()
        if report["postgres"]:
            try:
                with get_pool().connection() as conn:
                    conn.execute("SELECT 1 FROM schema_migrations LIMIT 1")
                    report["migrated"] = True
                    report["chunks"] = chunk_count(conn)
            except Exception:
                report["migrated"] = False
    if settings.redis_url:
        from app.db.kv import RedisKV

        report["redis"] = RedisKV(settings.redis_url).ping()
    if settings.brevo_api_key:
        from app.auth.email import BrevoMailer

        report["brevo"] = BrevoMailer.from_settings(settings).status()
    return report


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "check"
    if command == "check":
        print(json.dumps(_check()))
        return 0
    if command == "migrate":
        from app.db.pg import migrate

        applied = migrate()
        print("applied: " + (", ".join(applied) if applied else "nothing (schema up to date)"))
        return 0
    print(f"unknown command {command!r} (use: check | migrate)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
