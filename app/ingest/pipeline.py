"""MILESTONE 6 — Ingestion: file -> Document -> Chunks -> vectors -> store.

Called from two places:
  - `uv run python -m app.ingest`      indexes every file in data/documents/
  - the upload endpoint                 indexes one uploaded file

Supports .md, .txt and .pdf. PDFs are read page by page and the pages are joined
with a form-feed ('\\f'); ingest_file then chunks them with the 'page' strategy so
every chunk carries metadata['page'] — the basis of page-index RAG and the
"jump to this page" feature in the UI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.models import Document
from app.vectordb.chunking import Chunker
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")
_PAGE_SEP = "\f"  # must match app/vectordb/chunking.PageChunker


def _stable_id(source_name: str) -> str:
    """A document id derived from the file NAME, so re-loading the same file
    always gives the same id (which is what makes re-ingestion idempotent)."""
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"


def _read_pdf(path: Path) -> tuple[str, int]:
    """Extract text from a PDF, one page at a time, joined by '\\f'. Returns (text, page_count)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return _PAGE_SEP.join(pages), len(pages)


def load_document(path: Path) -> Document:
    """Read a .md/.txt/.pdf file into a Document with a stable id.

    Raises ValueError for any other suffix.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    source = path.name
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r} (allowed: {', '.join(SUPPORTED_SUFFIXES)})")

    if suffix == ".pdf":
        text, page_count = _read_pdf(path)
        metadata = {"is_pdf": True, "pages": page_count}
    else:
        text = path.read_text(encoding="utf-8")  # plain text / markdown
        metadata = {}

    return Document(id=_stable_id(source), text=text, source=source, metadata=metadata)


def ingest_file(path: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """Load -> chunk -> embed -> (delete old copy) -> add. Returns chunks stored."""
    doc = load_document(path)

    # PDFs always use the page-aware chunker so page numbers survive, whatever the
    # global strategy is. Text/markdown use the chunker we were handed.
    active = chunker
    if doc.metadata.get("is_pdf"):
        from app.vectordb.chunking import get_chunker

        active = get_chunker("page")

    chunks = active.chunk(doc)

    # Remove any previous version of this document FIRST, so running ingestion
    # twice on the same file never creates duplicates.
    store.delete_document(doc.id)
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])  # one batched call
        store.add(chunks, vectors)
    return len(chunks)


def ingest_directory(directory: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """Ingest every supported file in `directory` (sorted by name). Returns total chunks."""
    directory = Path(directory)
    total = 0
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            total += ingest_file(path, chunker=chunker, embedder=embedder, store=store)
    return total
