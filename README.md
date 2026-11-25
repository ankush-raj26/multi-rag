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

## 2. Set up and run (one command)

You also need Docker Desktop running: chat history, accounts and the vector index live in Postgres,
and login sessions / e-mail codes live in Redis. Both run as containers.

```bash
cd rag-chatbot
./start.sh
```

The script checks `uv` and Docker, starts Postgres (with pgvector) and Redis via `docker compose`,
installs Python dependencies, creates `.env` from `.env.example` on the first run (add your keys, run it
again), applies the database migrations, runs a health check, indexes `data/documents/` when the vector
table is empty, and starts the app on <http://127.0.0.1:8000/auth>.

Useful variants:

| Command | What it does |
|---|---|
| `./start.sh --no-check` | skip the live LLM / LangSmith key test |
| `./start.sh --reset` | delete the Postgres + Redis data and set up again |
| `docker compose ps` | are the containers up? |
| `docker compose down` | stop them (data is kept in docker volumes) |
| `uv run python -m app.db check` | one-line JSON health report |
| `uv run python -m app.db migrate` | apply new SQL files from `app/db/migrations/` |

### Accounts and e-mail

Sign up asks for a 6-digit code sent by e-mail (Brevo). Forgot password sends a one-time link.
Set `BREVO_API_KEY` in `.env` (Brevo → SMTP & API → API keys). Without it the app still works:
the code is printed to the terminal running the server instead of being e-mailed.
`BREVO_SENDER_EMAIL` is optional; the first verified sender on the Brevo account is used.

### Uploaded documents (media) in S3

Uploads are stored in an S3 bucket; `data/documents/` is a local cache of it. Locally the bucket is
MinIO (started by `docker compose`, console at <http://127.0.0.1:9011>, user `rag` / `ragragrag`).
For AWS S3, Cloudflare R2 or any S3 API, change `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION` and the
keys in `.env`. At start-up and before indexing the folder and the bucket are synced both ways, so
the bundled sample documents end up in the bucket and a teammate pointing at the same bucket gets
every upload. Leave `S3_BUCKET` empty to keep files only on disk.

Ports: Postgres on `127.0.0.1:5434`, Redis on `127.0.0.1:6380`, MinIO on `127.0.0.1:9010` / `9011`
(not the defaults, so other local projects are never touched). Credentials are in `docker-compose.yml`
and `.env.example`.

### Running without Docker

Leave `DATABASE_URL` and `REDIS_URL` empty in `.env`. The app then runs in the original single-user
mode: no login page, chats in `data/sessions/*.json`, vectors in `data/vectorstore/store.json`.
This is also how the test-suite runs, so tests never need the containers.

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
  auth/           accounts: Brevo e-mail, rules (service.py), /auth pages (router.py)
  db/             Postgres pool + SQL migrations, Redis key-value, `python -m app.db check|migrate`
  storage/        chat sessions: pg_store.py (Postgres, per user) or json_store.py (no-Docker mode)
  storage/blob.py uploaded documents in S3 (MinIO locally), data/documents/ as the cache
  vectordb/pg_store.py   vectors + keyword search in Postgres (pgvector + tsvector)
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
templates/, static/   the server-rendered UI (templates/auth.html + static/auth/ = the access page)
docker-compose.yml    Postgres (pgvector) + Redis + MinIO for local development
```

Models in `app/models.py` are the contract between your modules and the trainer's evaluator — don't rename fields.
