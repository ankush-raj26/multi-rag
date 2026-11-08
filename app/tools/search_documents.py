"""The `search_documents` tool — the 'agentic RAG' option.

In the default graph, retrieval runs on EVERY turn (M7). This tool lets the model
*decide* to search again — e.g. to look up a second thing mid-answer — which is
what "agentic RAG" means: the agent chooses when to use the knowledge base.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def search_documents(query: str) -> str:
    """Search the indexed company documents for a query and return the top passages.

    Use this to look up facts from the knowledge base when you need more context.
    """
    try:
        from app.vectordb.retriever import get_retriever

        hits = get_retriever().retrieve(query)
        if not hits:
            return "No matching passages found."
        return "\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)
    except Exception as exc:  # never crash the agent loop
        return f"Error: document search unavailable ({type(exc).__name__})"
