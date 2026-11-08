# delete_after_learning.md
*(Your cheat-sheet for the demo. Read it, understand it, then delete it before you hand the repo over.)*

This file explains **what was built, why, and exactly what to say to your mentor**.
Everything maps to the milestones in `TRAINEE_TASKS.md` (M1–M8), plus the extra RAG
flavours and the PDF page demo you asked for.

---

## 0. First, the important warnings (read these!)

1. **The two API keys could not be verified from here.** This machine's sandbox blocks
   outbound calls to Groq / Google / Hugging Face, so I could not make a live call to
   confirm the keys work. Also note: **both keys you gave me are identical**, and the value
   (`AQ.Ab8...`) does **not** look like a normal Groq key (`gsk_...`) or Gemini key (`AIza...`).
   **Before the demo, run one live test yourself** (see §7). If the chat says
   `Engine error: ... 401/invalid api key`, the key is the problem, not the code.
2. **Install the new dependencies on your Mac first:** `uv sync`. I added `deepagents`, `langchain-groq`,
   `langchain-google-genai`, `fastembed`, `numpy`, `rank-bm25`, `pypdf` to `pyproject.toml`.
3. **First run downloads the embedding model** (`BAAI/bge-small-en-v1.5`, ~130 MB) from the
   internet. After that it runs offline.
4. Your macOS `.venv` was **not touched**. I ran the tests in a throwaway Linux venv so I
   couldn't break your environment.
5. **Git history is already in the folder** — one commit per milestone (`git log --oneline`).

---

## 1. The 30-second summary (say this first)

> "It's a session-based RAG chatbot. FastAPI serves the UI and APIs; the AI pipeline is a
> **LangGraph** graph. Documents are **chunked**, **embedded** into vectors, and stored in a
> **persistent vector store** that also keeps a **BM25 keyword index**. On every question we
> **retrieve** the most relevant chunks (**hybrid**: semantic + keyword), feed them to the LLM,
> and show the **sources** under the answer. The LLM can also call **tools** (calculator, date,
> document search). Every run is traced in **LangSmith**."

---

## 2. Overall architecture

```
Browser (HTMX UI)
   │  POST /ui/sessions/{id}/messages
   ▼
FastAPI (app/web.py, app/api/)         ← plumbing, already provided
   │  run_chat_turn()  (app/services.py)
   ▼
ChatEngine  (app/chat/engine.py)       ← a compiled LangGraph graph   [M1,M7,M8]
   │
   ├── retrieve node ── Retriever (app/vectordb/retriever.py)         [M7]
   │                      │  embed query (app/vectordb/embeddings.py) [M4]
   │                      │  dense search + BM25 (app/vectordb/store.py) [M5]
   │                      ▼  hybrid fusion (RRF)
   │
   └── generate node ── LLM (app/chat/llm.py: Groq/Gemini/…)          [M1]
                          └── tools loop (app/tools/)                 [M8]

Ingestion (app/ingest/pipeline.py)  file → Document → Chunks → vectors → store   [M3,M4,M5,M6]
Observability (app/observability.py) turns LangSmith tracing on/off              [M2]
```

Data types in `app/models.py` are the **contract** (Document, Chunk, RetrievedChunk, Source,
EngineResult). I did not rename any field; I only **added** one optional field: `Source.page`
(for the PDF page jump).

---

## 3. What each milestone does (one line + why)

| M | File(s) | What I did | Why |
|---|---|---|---|
| M1 | `chat/llm.py`, `chat/engine.py` | `get_llm()` picks the provider from `.env`; `get_engine()` builds a `START→generate→END` LangGraph | A graph (not a plain function) so we can add retrieve + tools as nodes later |
| M2 | `observability.py` | `configure_tracing()` returns True only if flag **and** key set; never crashes | Safe to run with no LangSmith key |
| M3 | `vectordb/chunking.py` | 4 strategies registered; **recursive** is default | See §4 — the question your mentor WILL ask |
| M4 | `vectordb/embeddings.py` | `fastembed` local model (default) + `google` + `hash` fallback | Local = no API key, deterministic, good semantics |
| M5 | `vectordb/store.py` | JSON-persisted store: cosine search **+ BM25 index** | One store powers both dense and keyword retrieval |
| M6 | `ingest/pipeline.py` | load md/txt/**pdf**, idempotent ingest | Stable id + delete-before-add = no duplicates |
| M7 | `vectordb/retriever.py`, `chat/engine.py` | hybrid retriever (`@traceable`), `retrieve` node before `generate` | Context reaches the LLM every turn; sources shown |
| + | `chat/router.py` | 3-layer router + deep agent route, badge in UI | Cheap layers first; greetings skip RAG |
| M8 | `tools/` | calculator (AST, no `eval`), datetime, search_documents | Safe tool use + agentic retrieval |

---

## 4. Chunking — the "why this strategy" answer (memorise this)

**I chose `recursive` (paragraph-aware) as the default.** When the mentor asks *"why this and
not the others?"*, say:

- **Fixed-size (sliding window):** simplest, but it cuts sentences in half, so a chunk can start
  or end mid-idea. Retrieval quality suffers. Good only as a baseline.
- **Recursive (chosen):** splits on paragraph boundaries first, **merges** small paragraphs up
  to the size limit, and only falls back to word-splitting for oversized paragraphs. Result:
  each chunk is a **coherent passage**, which is what makes embeddings and retrieval accurate.
- **Markdown-heading:** great for structured docs, but our corpus mixes `.md`, `.txt` and `.pdf`,
  so heading-splitting isn't universal.
- **Page:** used **automatically for PDFs** so each chunk carries its page number (page-index RAG).

**The non-negotiable rules every strategy obeys** (the evaluator checks these):
no text is ever lost (I pack **whole words**, never mid-word), every chunk ≤ `chunk_size`
characters, deterministic output, unique/stable ids, and smaller size ⇒ more chunks.
`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` by default — overlap repeats ~100 chars across the cut so a
sentence split across two chunks still appears whole in one of them.

Switch strategy any time with `CHUNKING_STRATEGY=fixed|recursive|markdown|page` in `.env`.

---

## 5. The four RAG "flavours" you asked for (and where they live)

- **Traditional (dense) RAG** — embed the query, cosine-search the vectors. `RETRIEVAL_MODE=dense`.
- **Hybrid RAG with BM25 (default)** — run dense **and** BM25 keyword search, then fuse with
  **Reciprocal Rank Fusion (RRF)**. `RETRIEVAL_MODE=hybrid`. Dense finds paraphrases
  ("vacation" ≈ "annual leave"); BM25 nails exact words/numbers ("Form 16", "clause 4.2").
  RRF just rewards chunks that rank high on *either* list, so no score-scaling is needed.
  → `app/vectordb/retriever.py` (`_rrf`) + `store.search_bm25`.
- **Agentic RAG** — the LLM has a `search_documents` **tool** and decides *when* to query the
  knowledge base (on top of the always-on retrieve node). → `app/tools/search_documents.py`
  + the tool loop in the graph.
- **Page-index RAG** — PDFs are chunked per page, each chunk keeps `metadata["page"]`, and the
  UI shows a **"📄 page N"** button that jumps straight to that page. → `PageChunker`,
  `ingest/pipeline._read_pdf`, `Source.page`, `/files/{name}`, the viewer in `templates/index.html`.

---

## 6. The graph, in words (draw this on the whiteboard)

```
START -> route --none-------------------------------> generate -> END
               --dense / bm25 / hybrid / page--> retrieve -> generate <-> tools
               --agentic---------------------------> deep_agent -> END
```

- **State** = `{ messages, context, sources, route, route_layer, route_reason, page }`.
  `messages` uses the `add_messages` reducer so nodes **append** (needed for the tool loop).
- **route** runs the 3-layer router (section 6b) and writes its decision into the state.
- A **conditional edge** reads `route` and picks the next node:
  greetings go straight to `generate` (no retrieval at all), normal questions go to `retrieve`,
  multi-step questions go to `deep_agent`.
- **retrieve** calls the retriever **in the mode the router chose** (dense / BM25 / hybrid /
  page) and fills `context` + `sources`.
- **generate** gives the LLM `system prompt + context + conversation`. If it asks for a tool,
  `tools_condition` sends it to `tools`, which loops back to `generate`.
- **deep_agent** is a `deepagents` agent (LangChain's "deep agents"). It keeps a **todo list /
  plan**, calls `search_knowledge_base` as many times as it needs, uses the calculator, then
  answers. Every chunk it reads becomes a source. If the agent fails for any reason we
  **fall back to hybrid RAG**, so the chat never breaks.
- **Why a graph?** The control flow is explicit, every step is traced in LangSmith, and adding a
  step (reranker, guardrail) is just one more node.

## 6b. The 3-layer router: cheapest layer first (`app/chat/router.py`)

| Layer | How it decides | Cost | Example |
|---|---|---|---|
| 1. **Regex** | pattern rules | ~0 ms, no API | `hi`, `thanks` → **no RAG**; `page 3` → **page-index**; `"exact phrase"` / email / code → **BM25** |
| 2. **Semantic analyzer** | embeds the question and compares it (cosine) to a few example questions per route; trusts it only if the score is ≥ 0.55 **and** beats the runner-up by 0.05 | 1 embedding | "compare leave with remote work" → **agentic**; "email for HR" → **BM25** |
| 3. **LLM picker** | asks the LLM to reply with one label | 1 LLM call | only for questions layers 1–2 weren't sure about |
| fallback | nothing confident | none | **hybrid** (the safe default) |

**What to tell your mentor:** "Routing is ordered by cost. Greetings and obvious cases are
handled by a regex for free, the semantic layer handles intent with one cheap embedding, and we
only spend an LLM call when we really don't know. A 'hi' never touches the vector store."

The UI shows the decision as a **badge on every answer** (💬 no RAG · 🧠 dense · 🔤 BM25 ·
⚡ hybrid · 📄 page · 🤖 deep agent) plus "via regex / semantic / llm / fallback", with the reason
underneath. The sidebar has a "How questions are routed" panel explaining the three layers.

Turn layer 3 off with `ROUTER_LLM_PICKER=false` (saves an LLM call; unclear questions then go to hybrid).

## 7. How to run + the 2-minute demo script

**Quickest:** `./start.sh` does all four steps below (checks keys, indexes, starts, opens the browser).

```bash
cd rag-chatbot
uv sync                                         # install/refresh dependencies (macOS .venv)
uv run python scripts/check_setup.py --fix      # test every key for real, repair .env, run real chat turns
uv run python -m app.ingest                     # index data/documents (first run downloads the embedding model)
uv run uvicorn app.main:app --reload            # start the app
```
Then open **http://127.0.0.1:8000/welcome** (landing page) or **http://127.0.0.1:8000** (chat).
Traces appear at **smith.langchain.com**, project `rag-chatbot-spytro`, filterable by `session_id`.

Tests (keep tracing off so test runs don't clutter LangSmith):
```bash
LANGSMITH_TRACING=false uv run pytest                 # plumbing tests
LANGSMITH_TRACING=false uv run pytest -m contract     # milestone tests
LANGSMITH_TRACING=false uv run python self_check.py   # PASS/FAIL per milestone
uv run ruff check && uv run mypy app                  # lint + types
```

**About your keys:** Groq is the main provider (`openai/gpt-oss-120b`), Gemini (`gemini-3.8-flash`) is
the backup. `check_setup.py --fix` switches to Gemini automatically if Groq fails. Groq keys start with
`gsk_`, Google keys with `AQ.` (Google's auth-key format since May 2026), LangSmith keys with `lsv2_`.
Model names were also updated: Groq shut down `llama-3.3-70b-versatile` (Aug 2026), so Groq now uses
`openai/gpt-oss-120b`; Google shut down `gemini-2.0-flash` (Jun 2026).

Demo flow to show the mentor:
1. New chat, say **"hi"** → badge shows **💬 no RAG via regex** (no retrieval at all).
2. Ask **"How many annual leave days do I get?"** → **⚡ hybrid** badge, answer cites a source.
3. Click the **"📄 page N"** button under the answer → the handbook opens at the Leave page.
4. Ask **"what is on page 4?"** → **📄 page-index via regex**.
5. Ask **"compare the leave policy with the remote work policy"** → **🤖 deep agent** (plans, searches, answers).
6. Ask **"what is 18 * 24?"** → the **calculator tool** runs, model answers 432.
7. (If LangSmith key set) open smith.langchain.com → show the trace: route → retrieve → generate,
   filtered by `session_id`.

**Live key test (do before the demo):** `uv run python scripts/check_setup.py`

---

## 8. Likely mentor questions → quick answers

- **"How do you stop duplicate chunks on re-ingest?"** Stable doc id (hash of filename) +
  `store.delete_document(id)` before `add`.
- **"Why fastembed and not OpenAI embeddings?"** No API key, runs locally/offline, deterministic,
  and good enough quality for this corpus. Swappable via `EMBEDDING_MODEL`.
- **"Is the calculator safe?"** Yes — it parses an AST and only allows numbers/operators; it never
  calls `eval`. Bad input returns a string starting with `Error`.
- **"Where does persistence live?"** `data/vectorstore/store.json`; a new store on the same folder
  reloads it. Sessions are one JSON file each under `data/sessions/`.
- **"What changed vs the scaffold?"** Only the milestone files + I added `Source.page`, a `/files`
  route, the PDF viewer, PDF upload support, and the sample PDF. No contract fields renamed.

---

*Written for the BigStep training demo. Safe to delete once you're comfortable explaining the above.*

---

## 9. The landing page (`/welcome`): how it works and what to say

Open `http://127.0.0.1:8000/welcome`. The chat is still at `/`; its sidebar links to the tour.

**What it is.** A scroll-driven page drawn as an engineering drawing set (sheets R-1 to R-6). It
does not *describe* the router, it *runs* it: your question is sent to `POST /api/route`, which
calls the same `engine.decide()` a real chat turn uses, and the 3D drawing shows that decision.

| Sheet | What happens | How it's built |
|---|---|---|
| R-1 Assembly | Type a question; the 3D router assembly turns as you scroll | three.js line drawing, pinned section, layered planes (grid paper, 3D, dimension line) |
| R-2 Ledger | What a plain RAG bot does to "hi" vs this bot | plain HTML table, quiet on purpose |
| R-3 Drop test | Your question falls through the gates and stops where the real router put it; the log fills in | scroll progress drives the packet path; real `/api/route` data |
| R-4 Details | The five retrieval modes; yours is stamped; the real passages it found | horizontal rail (scroll moves it sideways) |
| R-5 Wiring | The LangGraph draws itself; your path is red | SVG lines drawn from scroll progress |
| R-6 Ask | Your question is already typed into the real bot | `POST /api/sessions` + `/messages`; works without JavaScript via `POST /ui/start` |

**Files:** `templates/welcome.html`, `static/welcome/welcome.js` (the 3D scene and scroll logic,
commented), `static/welcome/welcome.css`, `app/api/route_preview.py` (the preview endpoint).
three.js and the fonts are stored locally in `static/vendor` and `static/fonts`, so the demo works offline.

**What to tell your mentor:** "The landing page can't lie about the router, because it asks the
router. The packet's path is computed from the real decision: the layer that caught it, the reason,
and the retrieval preview. If the server is down the page says 'offline preview: regex layer only'
instead of pretending."

**Good demo questions:** `hi` (stops at L1, slides to "no retrieval"), `what is on page 4` (L1,
page-index), `what is the email for HR` (L2, BM25), `compare the leave policy with the remote work
policy` (L2, deep agent), `How many annual leave days do I get?` (hybrid).

**Safety nets:** no WebGL → a flat SVG version of the same drawing; reduced motion → the finished
state is shown without the drop animation; phones get their own layout.

**Router fix made while building this:** layer 2 had no examples for plain factual questions, so
"how many leave days" was being sent to the deep agent. I added a `hybrid` example cluster in
`app/chat/router.py` with generic factual questions (deliberately not the demo question).
