# RAG Chatbot — Training Project

A small chat app with a sessions sidebar (create, rename, delete) and a chat pane. The web pages, the JSON
session storage and the upload endpoint already work. **You build the AI parts** — the LLM, the LangGraph engine,
LangSmith tracing, chunking, embeddings, a vector store, ingestion, retrieval and tools — one milestone at a time.

Read [TRAINEE_TASKS.md](TRAINEE_TASKS.md) for the milestones (M1–M8, in order).

## 1. Install the tools (macOS)

```bash
brew install uv
```

`uv` installs Python 3.12 and every dependency for you. No separate `pip` or `venv` steps.

## 2. Set up the project

```bash
cd rag-chatbot
uv sync                  # creates .venv and installs dependencies
cp .env.example .env     # then edit .env: set SELF_CHECK_KEY to your name, add API keys later
```

## 3. Run the app

```bash
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Create a session and send a message. Until you finish Milestone 1 the reply is a
placeholder, and the sidebar lists the modules that are not built yet.

## 4. Test your work

| What | Command |
|---|---|
| App plumbing tests (always green) | `uv run pytest` |
| Milestone contract tests (pass = milestone done) | `uv run pytest -m contract` |
| One milestone only | `uv run pytest -m contract tests/contract/test_m3_chunking.py` |
| Pass/fail summary per milestone | `uv run python self_check.py` |
| Lint and types | `uv run ruff check` and `uv run mypy app` |

`self_check.py` prints PASS or FAIL per milestone and writes `self_check_report.json`. It needs `SELF_CHECK_KEY` in `.env`.
Passing it is a good sign, not a grade — your trainer runs a separate, stricter evaluation.

The fakes in `tests/fakes.py` (a fake chat model, a hash embedder, an in-memory store) let you test one module at a
time without any API key.

## 5. Index documents

```bash
uv run python -m app.ingest
```

Indexes everything in `data/documents/`. You can also upload `.md` / `.txt` files in the browser.

## Adding provider packages

```bash
uv add langchain-anthropic      # or langchain-openai
uv add chromadb                 # or faiss-cpu, lancedb ... any vector store you pick
```

Set `LLM_PROVIDER`, `LLM_MODEL` and the API key in `.env`. Never commit `.env`.

## Layout

```
app/
  main.py, config.py, deps.py, services.py, web.py   working plumbing — leave alone
  api/            JSON endpoints (sessions, chat, documents)
  storage/        one JSON file per session
  chat/llm.py            M1  get_llm
  chat/engine.py         M1, M7, M8  LangGraph engine
  observability.py       M2  LangSmith
  vectordb/chunking.py   M3
  vectordb/embeddings.py M4
  vectordb/store.py      M5
  ingest/                M6
  vectordb/retriever.py  M7
  tools/                 M8
data/documents/  sample documents      data/sessions/  chat history (JSON)
tests/           plumbing tests + tests/contract (one file per milestone)
templates/, static/   the server-rendered UI
```

Models in `app/models.py` are the contract between your modules and the trainer's evaluator — don't rename fields.
