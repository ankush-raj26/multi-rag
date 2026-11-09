"""MILESTONES 1, 7, 8 (+ router + deep agent) — the chat engine as a LangGraph graph.

    START -> route --none-------------------------------> generate -> END
                   --dense/bm25/hybrid/page--> retrieve -> generate <-> tools
                   --agentic---------------------------> deep_agent -> END

route      3-layer router (regex -> semantic analyzer -> LLM picker), app/chat/router.py
retrieve   RAG with the mode the router chose (M7)
generate   the LLM answers with system prompt + context (M1), may call tools (M8)
deep_agent a `deepagents` agent that plans (todo list), calls tools and searches the
           knowledge base itself — used for multi-step questions ("agentic RAG")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Protocol, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.chat.router import QueryRouter, RouteDecision
from app.models import EngineResult, Message, RetrievedChunk, RouteInfo, Source
from app.vectordb.retriever import Retriever

logger = logging.getLogger(__name__)
PLACEHOLDER_ANSWER = "Engine not implemented yet — see app/chat/engine.py"

SYSTEM_PROMPT = (
    "You are the assistant for this company's internal knowledge base. "
    "Only answer questions about the company and its documents (policies, leave, expenses, "
    "remote work, onboarding, IT, benefits, products), using the provided context. "
    "If a question, or part of one, is about anything else (general knowledge, news, weather, "
    "places, food, other companies), say briefly that you can only help with company topics, "
    "and answer only the company part if there is one. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Exception: you may use the calculator and current date/time tools for any question."
)
# Sent WITHOUT calling the model when a search finds nothing relevant: deterministic, free,
# and it can't drift into answering general-knowledge questions.
OUT_OF_SCOPE_REPLY = (
    "I can only help with questions about the company and its documents, such as leave, "
    "expenses, remote work, onboarding, IT support and benefits. I couldn't find anything "
    "about that in them, so I won't guess.\n\n"
    "I can also do calculations and tell you the current date and time."
)
AGENT_PROMPT = (
    "You are a research agent for a company knowledge base. For multi-step questions, "
    "plan first, use `search_knowledge_base` (more than once if needed) to gather facts, "
    "use `calculator` for any arithmetic, then give one clear final answer that names "
    "the documents you used. Never invent facts that are not in the documents."
)


class ChatEngine(Protocol):
    graph: Any

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        ...


class PlaceholderEngine:
    graph: Any = None

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        return EngineResult(answer=PLACEHOLDER_ANSWER, sources=[])


# --------------------------------------------------------------------------- state


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # nodes APPEND to this
    context: str            # retrieved text (M7)
    sources: list[Source]   # shown under the answer (M7)
    route: str              # router decision: none|dense|bm25|hybrid|page|agentic
    route_layer: str        # which layer decided: regex|semantic|llm|fallback
    route_reason: str       # why, in plain words (shown as a badge in the UI)
    page: int | None        # page number for the 'page' route
    retrieved: int          # passages that passed the relevance threshold (0 = nothing relevant)
    out_of_scope: bool      # searched, nothing relevant -> fixed reply, the model is not asked


def _to_langchain_messages(history: list[Message]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        else:
            out.append(SystemMessage(content=m.content))
    return out


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _previous_user_text(messages: list[BaseMessage]) -> str:
    """The user message before the latest one ('' if none): context for short follow-ups."""
    seen = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            seen += 1
            if seen == 2:
                return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _text(content: Any) -> str:
    """Message content as plain text (some providers send a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else str(p.get("text", "")) for p in content if isinstance(p, (str, dict))]
        return "".join(parts)
    return str(content)


def _to_sources(hits: list[RetrievedChunk]) -> list[Source]:
    """RetrievedChunks -> Sources for the UI (de-duplicated by chunk id, order kept)."""
    seen: set[str] = set()
    out: list[Source] = []
    for h in hits:
        if h.chunk.id in seen:
            continue
        seen.add(h.chunk.id)
        out.append(
            Source(
                source=h.chunk.source,
                chunk_id=h.chunk.id,
                # real cosine similarity when the retriever measured it, else the rank score
                score=h.chunk.metadata.get("relevance", h.score),
                snippet=h.chunk.text[:200],
                page=h.chunk.metadata.get("page"),  # set for PDF chunks -> UI can jump
            )
        )
    return out


class ChatEngineImpl:
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Retriever | None = None,
        tools: list[BaseTool] | None = None,
        embedder: Any | None = None,
        use_llm_picker: bool = True,
    ) -> None:
        self._raw_llm = llm                       # unbound model: used by router + deep agent
        self._retriever = retriever
        self._tools = tools or []
        self._llm = llm.bind_tools(self._tools) if self._tools else llm  # M8
        self._router = QueryRouter(llm=llm, embedder=embedder, use_llm=use_llm_picker)
        self._deep_agents_ok = self._check_deepagents()
        self.graph = self._build_graph()

    @staticmethod
    def _check_deepagents() -> bool:
        try:
            import deepagents  # noqa: F401
        except ImportError:
            return False
        return True

    # ----- nodes ----------------------------------------------------------
    @property
    def picker_enabled(self) -> bool:
        """Whether router layer 3 (the LLM picker) is switched on."""
        return self._router.use_llm

    def decide(self, query: str) -> RouteDecision:
        """The router's decision for `query`, exactly as a chat turn would make it.

        Public so the landing page preview (POST /api/route) shows the real
        decision instead of a re-implementation of it.
        """
        decision: RouteDecision = self._router.route(query)
        # Downgrade routes we can't serve with the pieces we were given.
        if decision.route == "agentic" and not (self._deep_agents_ok and self._retriever is not None):
            decision.route = "hybrid"
            decision.reason += " (agent unavailable -> hybrid)"
        return decision

    def _route(self, state: GraphState) -> dict[str, Any]:
        """Run the 3-layer router on the latest user message."""
        decision = self.decide(_last_user_text(state["messages"]))
        return {
            "route": decision.route,
            "route_layer": decision.layer,
            "route_reason": decision.reason,
            "page": decision.page,
        }

    def _next_after_route(self, state: GraphState) -> str:
        """Conditional edge: which node runs after `route`."""
        route = state.get("route", "hybrid")
        if route == "none" or self._retriever is None:
            return "generate"          # greetings (or no index yet): answer directly
        if route == "agentic":
            return "deep_agent"
        return "retrieve"

    def _retrieve(self, state: GraphState) -> dict[str, Any]:
        """RAG step with the mode the router picked."""
        query = _last_user_text(state["messages"])
        mode = state.get("route", "hybrid")
        if self._retriever is None:
            return {"context": "", "sources": []}
        if not hasattr(self._retriever, "retrieve_mode"):
            hits = self._retriever.retrieve(query)   # evaluator's fakes only have retrieve()
        else:
            hits = self._relevant(query, mode, state)
            if not hits:   # nothing in the documents is about this: decline, don't ask the model
                return {"context": "", "sources": [], "retrieved": 0, "out_of_scope": True}
        context = "\n\n".join(
            f"[{h.chunk.source}{' p.' + str(h.chunk.metadata['page']) if h.chunk.metadata.get('page') else ''}] "
            f"{h.chunk.text}"
            for h in hits
        )
        return {"context": context, "sources": _to_sources(hits), "retrieved": len(hits)}

    def _relevant(self, query: str, mode: str, state: GraphState) -> list[RetrievedChunk]:
        """Relevance-checked search. A short follow-up ("and for part-timers?") means little on
        its own, so if nothing passes, try once more with the previous question attached."""
        assert self._retriever is not None
        hits: list[RetrievedChunk] = self._retriever.retrieve_mode(query, mode, page=state.get("page"))  # type: ignore[attr-defined]
        previous = _previous_user_text(state["messages"])
        if not hits and previous:
            hits = self._retriever.retrieve_mode(f"{previous}\n{query}", mode, page=state.get("page"))  # type: ignore[attr-defined]
        return hits

    def _generate(self, state: GraphState) -> dict[str, Any]:
        if state.get("out_of_scope"):
            return {"messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)]}
        prompt: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if state.get("context"):
            prompt.append(SystemMessage(content=f"Context from documents:\n{state['context']}"))
        prompt.extend(state["messages"])
        return {"messages": [self._llm.invoke(prompt)]}

    def _deep_agent(self, state: GraphState) -> dict[str, Any]:
        """Agentic RAG: a deepagents agent plans, searches and calculates on its own."""
        # Never start an agent for a question the documents say nothing about.
        if self._retriever is not None and hasattr(self._retriever, "retrieve_mode"):
            if not self._relevant(_last_user_text(state["messages"]), "hybrid", state):
                return {"messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)], "sources": [], "retrieved": 0,
                        "out_of_scope": True}
        from deepagents import create_deep_agent

        collected: list[RetrievedChunk] = []   # every chunk the agent looked at -> sources
        retriever = self._retriever

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the indexed company documents and return the most relevant passages."""
            if retriever is None:
                hits = []
            elif hasattr(retriever, "retrieve_mode"):
                hits = retriever.retrieve_mode(query, "hybrid")   # same relevance threshold as RAG
            else:
                hits = retriever.retrieve(query)
            collected.extend(hits)
            if not hits:
                return "No matching passages found."
            return "\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)

        agent_tools = [search_knowledge_base, *[t for t in self._tools if t.name != "search_documents"]]
        try:
            # Building the agent sits inside the try too: a bad model/tool setup must
            # fall back to plain RAG, not crash the chat.
            agent: Any = create_deep_agent(model=self._raw_llm, tools=agent_tools, system_prompt=AGENT_PROMPT)
            result = agent.invoke({"messages": list(state["messages"])})
            answer_msg = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), None
            )
            reply = AIMessage(content=answer_msg.content if answer_msg else "The agent returned no answer.")
        except Exception as exc:  # never break the chat: fall back to a plain RAG answer
            logger.exception("deep agent failed, falling back to hybrid RAG")
            reason = f"{state.get('route_reason', '')} (agent error: {exc}) -> hybrid fallback"
            fallback = self._retrieve(cast(GraphState, {**state, "route": "hybrid"}))
            answer = self._generate(cast(GraphState, {**state, **fallback}))
            return {**fallback, **answer, "route": "hybrid", "route_reason": reason}
        sources = _to_sources(collected)
        return {"messages": [reply], "sources": sources, "retrieved": len(sources)}

    # ----- wiring ---------------------------------------------------------
    def _build_graph(self) -> Any:
        builder = StateGraph(GraphState)
        builder.add_node("route", self._route)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("deep_agent", self._deep_agent)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route", self._next_after_route,
            {"generate": "generate", "retrieve": "retrieve", "deep_agent": "deep_agent"},
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("deep_agent", END)

        if self._tools:
            builder.add_node("tools", ToolNode(self._tools))
            builder.add_conditional_edges("generate", tools_condition)  # tool call? -> tools
            builder.add_edge("tools", "generate")
        else:
            builder.add_edge("generate", END)
        return builder.compile()

    # ----- public API -----------------------------------------------------
    @staticmethod
    def _initial(message: str, history: list[Message]) -> GraphState:
        start = [*_to_langchain_messages(history), HumanMessage(content=message)]
        return {"messages": start, "context": "", "sources": []}

    @staticmethod
    def _result(final: dict[str, Any]) -> EngineResult:
        answer = ""
        for m in reversed(final.get("messages", [])):
            if isinstance(m, AIMessage) and m.content:
                answer = m.content if isinstance(m.content, str) else str(m.content)
                break
        route = None
        if final.get("route"):
            route = RouteInfo(
                route=final["route"], layer=final.get("route_layer", ""), reason=final.get("route_reason", ""),
                retrieved=final.get("retrieved") if final["route"] != "none" else None,
            )
        return EngineResult(answer=answer, sources=final.get("sources", []), route=route)

    def invoke(self, message: str, history: list[Message], session_id: str) -> EngineResult:
        final = self.graph.invoke(self._initial(message, history), config={"metadata": {"session_id": session_id}})
        return self._result(final)

    def stream(self, message: str, history: list[Message], session_id: str) -> Iterator[tuple[str, Any]]:
        """Run the same graph, yielding events as they happen:

            ("route", RouteInfo)    as soon as the router has decided
            ("token", str)          each piece of the answer from the `generate` node
            ("result", EngineResult) once, at the end (same shape `invoke` returns)

        Only `generate` streams tokens: the router's LLM picker and the deep agent's
        planning calls are internal, so the deep agent's answer arrives in `result`.
        """
        final: dict[str, Any] = {"messages": []}
        for mode, data in self.graph.stream(
            self._initial(message, history),
            config={"metadata": {"session_id": session_id}},
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                chunk, meta = data   # AIMessageChunk from streaming models, one AIMessage otherwise
                if meta.get("langgraph_node") == "generate" and isinstance(chunk, AIMessage) and chunk.content:
                    yield "token", _text(chunk.content)
                continue
            for node, update in data.items():   # mode == "updates": {node: what it returned}
                if not update:
                    continue
                for key, value in update.items():
                    if key == "messages":
                        final["messages"].extend(value)
                    else:
                        final[key] = value
                if node == "route":
                    yield "route", RouteInfo(
                        route=update["route"],
                        layer=update.get("route_layer", ""),
                        reason=update.get("route_reason", ""),
                    )
        yield "result", self._result(final)


def get_engine(
    llm: BaseChatModel | None = None,
    retriever: Retriever | None = None,
    tools: list[BaseTool] | None = None,
    embedder: Any | None = None,
) -> ChatEngine:
    """Build the engine from the pieces it is GIVEN.

    llm None -> get_llm();  tools None -> app.tools.get_tools();  tools=[] -> no tools.
    embedder (optional) powers layer 2 of the router; without it layer 2 is skipped.
    """
    from app.config import get_settings

    if llm is None:
        from app.chat.llm import get_llm

        llm = get_llm()
    if tools is None:
        try:
            from app.tools import get_tools

            tools = get_tools()
        except NotImplementedError:
            tools = []
    return ChatEngineImpl(
        llm, retriever=retriever, tools=tools, embedder=embedder,
        use_llm_picker=get_settings().router_llm_picker,
    )
