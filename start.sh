#!/usr/bin/env bash
# One command to run the whole thing on a fresh machine or an existing one.
#
#   ./start.sh              check -> set up what is missing -> start everything
#   ./start.sh --no-check   skip the live LLM/LangSmith key test
#   ./start.sh --reset      wipe Postgres + Redis data (docker volumes) and set up again
#
# What it does, in order:
#   1. tool check        uv and docker must exist
#   2. infrastructure    docker compose up -d (Postgres+pgvector, Redis); waits until both are healthy
#   3. python deps       uv sync
#   4. .env              created from .env.example if missing (then you add keys)
#   5. database          applies SQL migrations (idempotent)
#   6. health check      python -m app.db check  -> JSON; aborts with a clear message if something is down
#   7. keys              scripts/check_setup.py --fix (unless --no-check)
#   8. documents         indexes data/documents only when the vector table is empty
#   9. app               uvicorn on http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\n!! %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is not installed. $2"; }

need uv     "Install with: brew install uv   (or see https://docs.astral.sh/uv/)"
need docker "Install Docker Desktop, then start it."
docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start Docker Desktop and run ./start.sh again."

if [[ "${1:-}" == "--reset" ]]; then
  say "Removing Postgres and Redis data (docker compose down -v)"
  docker compose down -v
  shift || true
fi

say "1/7 Infrastructure: Postgres (pgvector) + Redis"
docker compose up -d --wait --wait-timeout 90 || fail "Containers did not become healthy. Run: docker compose logs"

say "2/7 Python dependencies (uv sync)"
uv sync -q

if [[ ! -f .env ]]; then
  cp .env.example .env
  say ".env created from .env.example. Add your GROQ_API_KEY / GOOGLE_API_KEY (and BREVO_API_KEY for e-mail), then run ./start.sh again."
  exit 0
fi

say "3/7 Database schema (migrations)"
uv run python -m app.db migrate

say "4/7 Health check"
REPORT="$(uv run python -m app.db check)"
echo "    $REPORT"
echo "$REPORT" | grep -q '"postgres": true' || fail "Postgres is not reachable at DATABASE_URL. Check .env and: docker compose ps"
echo "$REPORT" | grep -q '"redis": true'    || fail "Redis is not reachable at REDIS_URL. Check .env and: docker compose ps"
echo "$REPORT" | grep -q '"brevo": "ok"'    || echo "    note: e-mail is not fully configured (see BREVO_* in .env); codes will be printed to this terminal instead."

if [[ "${1:-}" != "--no-check" ]]; then
  say "5/7 Testing Groq, Gemini and LangSmith keys with real calls"
  uv run python scripts/check_setup.py --fix || {
    echo; echo "Key check reported a failure (see above). Fix .env, or run ./start.sh --no-check to start anyway."
    exit 1
  }
fi

if echo "$REPORT" | grep -q '"chunks": 0'; then
  say "6/7 Vector index is empty: indexing data/documents (first run downloads the embedding model, ~130 MB)"
  uv run python -m app.ingest
else
  say "6/7 Vector index already has data: skipping indexing (upload files in the app to add more)"
fi

say "7/7 Starting the app on http://127.0.0.1:8000  (Ctrl+C to stop; containers keep running)"
( sleep 3; open "http://127.0.0.1:8000/auth" 2>/dev/null || xdg-open "http://127.0.0.1:8000/auth" 2>/dev/null || true ) &
exec uv run uvicorn app.main:app --reload
