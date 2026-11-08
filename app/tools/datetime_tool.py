"""The `current_datetime` tool — the model has no clock, so it asks this."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO 8601 format (e.g. 2026-11-08T21:30:00).

    Use this whenever the user asks about today's date, the current time, or 'now'.
    """
    return datetime.now().isoformat(timespec="seconds")
