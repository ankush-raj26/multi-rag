"""3-layer query router — decides HOW (or whether) to do RAG for each message.

    Layer 1  REGEX      cheap pattern rules: "hi", "thanks", "page 4" ...      (~0 ms)
    Layer 2  SEMANTIC   embed the query, compare to example queries per route  (~ms)
    Layer 3  LLM PICKER ask the LLM to choose, only when layers 1-2 are unsure (1 call)

The cheaper layer answers first; we only pay for the next layer when the previous
one is not confident. Greetings never touch the vector store or the LLM picker.

Routes:
  none     no retrieval (small talk / greetings)
  dense    traditional semantic RAG   (conceptual "explain / why" questions)
  bm25     keyword RAG                (exact terms, emails, codes, numbers)
  hybrid   dense + BM25 fused (RRF)   (the safe default)
  page     page-index RAG             ("what's on page 3")
  agentic  deep agent with tools      (multi-step: compare, calculate from docs, plan)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

ROUTES = ("none", "dense", "bm25", "hybrid", "page", "agentic")

# ----------------------------------------------------------------------- layer 1: regex

_GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|namaste|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|thx|ok(ay)?|bye|goodbye|how\s+are\s+you|who\s+are\s+you)"
    r"[\s!.?,]*$",
    re.IGNORECASE,
)
_PAGE = re.compile(r"\bpage\s*(?:no\.?|number)?\s*(\d{1,4})\b", re.IGNORECASE)
_EXACT = re.compile(r"(\"[^\"]+\"|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b[A-Z]{2,}-?\d+\b|\bclause\s+\d)", re.IGNORECASE)


@dataclass
class RouteDecision:
    route: str            # one of ROUTES
    layer: str            # "regex" | "semantic" | "llm" | "fallback"
    reason: str           # human-readable why (shown in the UI)
    page: int | None = None


def _layer1_regex(query: str) -> RouteDecision | None:
    if _GREETING.match(query):
        return RouteDecision("none", "regex", "greeting / small talk — no retrieval needed")
    if m := _PAGE.search(query):
        return RouteDecision("page", "regex", f"mentions page {m.group(1)}", page=int(m.group(1)))
    if _EXACT.search(query):
        return RouteDecision("bm25", "regex", "exact term (quote / email / code) — keyword search")
    return None


# ----------------------------------------------------------------------- layer 2: semantic

# A few example questions per route. The query is routed to the closest examples.
_PROTOTYPES: dict[str, list[str]] = {
    "none": ["how are you doing today", "tell me a joke", "what can you do", "nice to meet you"],
    "dense": [
        "explain the policy in simple words", "why do we have this rule",
        "summarize the onboarding process", "what is the idea behind remote work",
    ],
    "bm25": [
        "what is the email address for HR", "find the exact clause number",
        "what is the helpdesk contact", "which form do I fill",
    ],
    "hybrid": [
        "how many sick days are allowed per year", "what is the limit for claiming expenses",
        "when does payroll run each month", "how long do I have to report a lost laptop",
    ],
    "agentic": [
        "compare the leave policy with the remote work policy",
        "calculate my total reimbursement from the expense rules",
        "make a step by step plan for my first week",
        "add the leave allowance and the remote stipend together",
    ],
}
_SEM_THRESHOLD = 0.55   # minimum cosine to trust a match
_SEM_MARGIN = 0.05      # best must beat the runner-up by this much


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticAnalyzer:
    """Embeds the prototype questions once, then scores each new query against them."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._labels: list[str] = []
        self._vectors: list[list[float]] = []

    def _ensure(self) -> None:
        if self._vectors:
            return
        for label, examples in _PROTOTYPES.items():
            for vec in self._embedder.embed_documents(examples):   # one batch per route
                self._labels.append(label)
                self._vectors.append(vec)

    def analyze(self, query: str) -> RouteDecision | None:
        self._ensure()
        q = self._embedder.embed_query(query)
        best: dict[str, float] = {}
        for label, vec in zip(self._labels, self._vectors, strict=True):
            best[label] = max(best.get(label, -1.0), _cos(q, vec))  # best example per route
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        (top, s1), (_, s2) = ranked[0], ranked[1]
        if s1 >= _SEM_THRESHOLD and s1 - s2 >= _SEM_MARGIN:
            return RouteDecision(top, "semantic", f"closest to '{top}' examples (cos {s1:.2f})")
        return None  # not confident -> let layer 3 decide


# ----------------------------------------------------------------------- layer 3: LLM picker

_PICKER_PROMPT = (
    "You route questions for a document Q&A bot. Pick exactly ONE label:\n"
    "none = greeting/small talk; dense = conceptual 'explain/why/summarize';\n"
    "bm25 = exact names, numbers, emails, codes; hybrid = normal factual question;\n"
    "agentic = multi-step (compare, combine, calculate from documents, plan).\n"
    "Reply with only the label.\nQuestion: {q}"
)


def _layer3_llm(llm: Any, query: str) -> RouteDecision | None:
    try:
        reply = llm.invoke(_PICKER_PROMPT.format(q=query))
        text = str(getattr(reply, "content", "") or "").strip().lower()
    except Exception:
        return None
    for label in ("agentic", "hybrid", "dense", "bm25", "none"):   # first valid label wins
        if re.search(rf"\b{label}\b", text):
            return RouteDecision(label, "llm", f"LLM picker chose '{label}'")
    return None  # unparseable answer -> fallback


# ----------------------------------------------------------------------- the router


class QueryRouter:
    """Runs the three layers in order and returns the first confident decision."""

    def __init__(self, llm: Any | None = None, embedder: Any | None = None, use_llm: bool = True) -> None:
        self._llm = llm
        self._semantic = SemanticAnalyzer(embedder) if embedder is not None else None
        self._use_llm = use_llm

    @property
    def use_llm(self) -> bool:
        """True when layer 3 can run (switched on AND a model was given)."""
        return self._use_llm and self._llm is not None

    def route(self, query: str) -> RouteDecision:
        if decision := _layer1_regex(query):                       # layer 1
            return decision
        if self._semantic is not None:                             # layer 2
            try:
                if decision := self._semantic.analyze(query):
                    return decision
            except Exception:
                pass  # an embedding hiccup must never break the chat
        if self._use_llm and self._llm is not None:                # layer 3
            if decision := _layer3_llm(self._llm, query):
                return decision
        return RouteDecision("hybrid", "fallback", "no layer was confident — safe default")
