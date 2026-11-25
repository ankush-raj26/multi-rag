-- 001: users, chat sessions + messages, vector chunks. Applied once by `python -m app.db migrate`.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,                 -- uuid4 hex
    email         TEXT NOT NULL UNIQUE,             -- stored lower-cased
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    verified_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         TEXT PRIMARY KEY,                    -- uuid4 hex (same shape as the old JSON files)
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_sessions_user_updated ON chat_sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    timestamp  TEXT NOT NULL,                       -- ISO string, as app.models.Message keeps it
    sources    JSONB NOT NULL DEFAULT '[]',
    route      JSONB,
    docs       JSONB NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS chat_messages_session ON chat_messages (session_id, id);

-- One row per chunk. `embedding` has no fixed dimension so any embedder works;
-- `tsv` is the keyword (BM25-style) side of hybrid retrieval, kept by Postgres itself.
CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    source    TEXT NOT NULL,
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}',
    embedding VECTOR,
    tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks (source);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING GIN (tsv);
