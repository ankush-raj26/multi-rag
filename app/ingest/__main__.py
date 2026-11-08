"""`uv run python -m app.ingest` — index every file in DOCUMENTS_DIR."""

from __future__ import annotations

from app.config import get_settings
from app.ingest.pipeline import ingest_directory
from app.vectordb.chunking import get_chunker
from app.vectordb.embeddings import get_embedder
from app.vectordb.store import get_vector_store


def main() -> None:
    settings = get_settings()
    total = ingest_directory(
        settings.documents_dir,
        chunker=get_chunker(),
        embedder=get_embedder(),
        store=get_vector_store(),
    )
    print(f"Indexed {total} chunks from {settings.documents_dir}")


if __name__ == "__main__":
    main()
