"""MILESTONE 2 — Observability with LangSmith.

LangChain / LangGraph export traces to LangSmith automatically *once the right
environment variables are set*. So our only job is to turn that on safely from
`.env` and give the project a sensible name — and to NEVER crash the app if the
key is missing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# "1", "true", "yes", "on" (any case) all count as "switched on".
_TRUEY = {"1", "true", "yes", "on"}


def _is_on(name: str) -> bool:
    """True if env var `name` holds a truthy string."""
    return os.getenv(name, "").strip().lower() in _TRUEY


def configure_tracing() -> bool:
    """Turn LangSmith tracing on when BOTH a flag and an API key are present.

    OUT: True if tracing is now active, False if it is off or misconfigured.
    Called once at start-up (see app/main.py).
    """
    # Accept either the new name (LANGSMITH_TRACING) or the legacy one.
    wants_tracing = _is_on("LANGSMITH_TRACING") or _is_on("LANGCHAIN_TRACING_V2")
    # Accept either key name as well.
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    if not wants_tracing:
        return False  # tracing explicitly off — nothing to do

    if not api_key:
        # They asked for tracing but gave no key: warn, stay off, do NOT crash.
        logger.warning("LANGSMITH_TRACING is on but no LANGSMITH_API_KEY is set — tracing disabled.")
        return False

    # Set BOTH flag names so every langchain/langgraph version picks it up.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key      # normalise onto the new name
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)  # and keep the legacy one

    # Make sure runs land in a named project (default if the user gave none),
    # so you can actually find them in the LangSmith UI.
    project = os.getenv("LANGSMITH_PROJECT") or "rag-chatbot"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ.setdefault("LANGCHAIN_PROJECT", project)

    logger.info("LangSmith tracing enabled (project=%s).", project)
    return True
