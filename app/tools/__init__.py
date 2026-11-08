"""MILESTONE 8 — Tools the LLM can call.

Each tool lives in its own module and is listed here. The engine binds these to
the model (M8) and adds a ToolNode + tools_condition loop so the model can call a
tool and then answer using the result.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.calculator import calculator
from app.tools.datetime_tool import current_datetime
from app.tools.search_documents import search_documents


def get_tools() -> list[BaseTool]:
    """Every tool the engine should expose to the LLM."""
    return [calculator, current_datetime, search_documents]
