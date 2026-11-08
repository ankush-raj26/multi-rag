#!/usr/bin/env bash
# One command to check everything and start the RAG chatbot.
#   ./start.sh            check keys, index documents, start the app, open the browser
#   ./start.sh --no-check skip the key check (faster restarts)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/4 Installing dependencies (uv sync)"
uv sync -q

if [[ "${1:-}" != "--no-check" ]]; then
  echo "==> 2/4 Testing Groq, Gemini and LangSmith keys with real calls"
  # --fix: if the configured provider fails but the other works, .env is switched to the working one
  uv run python scripts/check_setup.py --fix || {
    echo; echo "Key check reported a failure (see above). Fix .env, or run ./start.sh --no-check to start anyway."
    exit 1
  }
fi

echo "==> 3/4 Indexing data/documents (first run downloads the embedding model, ~130 MB)"
uv run python -m app.ingest

echo "==> 4/4 Starting the app on http://127.0.0.1:8000  (Ctrl+C to stop)"
( sleep 3; open "http://127.0.0.1:8000/welcome" 2>/dev/null || xdg-open "http://127.0.0.1:8000/welcome" 2>/dev/null || true ) &
exec uv run uvicorn app.main:app --reload
