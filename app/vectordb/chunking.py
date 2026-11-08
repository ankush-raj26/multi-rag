"""MILESTONE 3 — Chunking.

Split a Document into Chunks that are small enough to embed and retrieve.

Four strategies are registered and picked by name from CHUNKING_STRATEGY:
  recursive  (default) — keep paragraphs/headings together, split big ones by words
  fixed                — a sliding window of words with overlap
  markdown             — one section per heading, then recursive inside
  page                 — one unit per PDF page (page-index RAG), then recursive inside

Every strategy shares one word-safe packer (`_pack_words`) so ALL of them obey
the rules the evaluator checks:
  1. empty/whitespace doc -> []            5. doc_id & source copied onto each chunk
  2. no empty chunk                        6. positions 0..n-1, ids unique & stable
  3. len(text) <= chunk_size               7. deterministic (no randomness)
  4. no word is ever lost                  8. smaller chunk_size -> more chunks
"""

from __future__ import annotations

import re
from typing import Protocol

from app.models import Chunk, Document

_HEADING = re.compile(r"^#{1,6}\s")          # markdown heading line, e.g. "## Leave"
_BLANKLINE = re.compile(r"\n\s*\n")          # blank line = paragraph separator
_PAGE_SEP = "\f"                             # form-feed = page break (used by PDFs, M6)


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        ...


# --------------------------------------------------------------------------- shared helpers


def _pack_words(words: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack whole words into strings of at most `size` characters.

    Working at the WORD level (never mid-word) is what guarantees rule 4: every
    original token stays intact, so no word can be lost. `overlap` characters of
    trailing words are repeated at the start of the next chunk to preserve context
    across the cut (rule-friendly: overlap only ADDS copies, it never drops text).
    """
    chunks: list[str] = []
    cur: list[str] = []      # words in the chunk being built
    cur_len = 0              # its length in characters (incl. the single spaces)

    for word in words:
        wlen = len(word)
        if wlen > size:
            # Pathological token longer than a whole chunk: flush, then hard-split
            # it so rule 3 (size limit) always holds. Natural text never hits this.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.extend(word[i : i + size] for i in range(0, wlen, size))
            continue

        extra = wlen + (1 if cur else 0)  # +1 for the space before this word
        if cur_len + extra > size:
            # This word would overflow -> close the current chunk first.
            chunks.append(" ".join(cur))
            # Seed the next chunk with the last `overlap` characters of words.
            back: list[str] = []
            blen = 0
            for prev in reversed(cur):
                step = len(prev) + (1 if back else 0)
                if blen + step > overlap:
                    break
                back.insert(0, prev)
                blen += step
            # Make sure the overlap + new word still fit inside `size`.
            while back and blen + wlen + 1 > size:
                blen -= len(back[0]) + (1 if len(back) > 1 else 0)
                back.pop(0)
            cur, cur_len = back, blen
            extra = wlen + (1 if cur else 0)

        cur.append(word)
        cur_len += extra

    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]  # rule 2: never emit an empty chunk


def _merge_then_pack(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Combine small pieces (paragraphs/sections) up to `size`, split big ones by words.

    Keeping related small pieces together, and only falling back to word-splitting
    for oversized ones, is what makes chunks read as coherent passages.
    """
    out: list[str] = []
    cur = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_pack_words(piece.split(), size, overlap))  # too big: split by words
            continue
        candidate = f"{cur}\n\n{piece}" if cur else piece
        if len(candidate) <= size:
            cur = candidate                       # still fits: keep accumulating
        else:
            out.append(cur)                       # would overflow: start a new chunk
            cur = piece
    if cur:
        out.append(cur)
    return [c for c in out if c.strip()]


def _finalize(document: Document, texts: list[str], meta_each: list[dict] | None) -> list[Chunk]:
    """Turn a list of chunk texts into Chunk objects with stable ids/positions."""
    chunks: list[Chunk] = []
    for position, text in enumerate(texts):           # rule 6: 0,1,2,... in order
        meta = dict(meta_each[position]) if meta_each else {}
        chunks.append(
            Chunk(
                id=f"{document.id}:{position}",       # unique + stable (rule 6)
                doc_id=document.id,                   # rule 5
                text=text,
                source=document.source,               # rule 5
                position=position,
                metadata=meta,
            )
        )
    return chunks


# --------------------------------------------------------------------------- strategies


class FixedSizeChunker:
    """A sliding window of words: simplest possible, real character-overlap."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():             # rule 1
            return []
        texts = _pack_words(document.text.split(), self.size, self.overlap)
        return _finalize(document, texts, None)


class RecursiveChunker:
    """Split on paragraphs first, merge small ones, word-split big ones (default)."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        paragraphs = _BLANKLINE.split(document.text)
        texts = _merge_then_pack(paragraphs, self.size, self.overlap)
        return _finalize(document, texts, None)


class MarkdownChunker:
    """One section per heading; keeps the heading attached and records it in metadata."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def _sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        heading = ""
        buf: list[str] = []
        for line in text.splitlines():
            if _HEADING.match(line):              # a new heading starts a new section
                if buf:
                    sections.append((heading, "\n".join(buf)))
                heading = line.lstrip("#").strip()
                buf = [line]
            else:
                buf.append(line)
        if buf:
            sections.append((heading, "\n".join(buf)))
        return sections

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        texts: list[str] = []
        metas: list[dict] = []
        for heading, body in self._sections(document.text):
            for piece in _merge_then_pack([body], self.size, self.overlap):
                texts.append(piece)
                metas.append({"heading": heading} if heading else {})
        return _finalize(document, texts, metas)


class PageChunker:
    """One unit per page (pages separated by form-feed '\\f'); records the page number.

    This powers 'page-index RAG': PDFs are loaded (M6) with a '\\f' between pages,
    so each chunk carries metadata['page'] and the UI can jump straight to it.
    """

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        pages = document.text.split(_PAGE_SEP) or [document.text]
        texts: list[str] = []
        metas: list[dict] = []
        for page_no, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            for piece in _merge_then_pack(_BLANKLINE.split(page_text), self.size, self.overlap):
                texts.append(piece)
                metas.append({"page": page_no})
        return _finalize(document, texts, metas)


# name -> class, so new strategies are one line to add (and one word in .env to pick)
_REGISTRY: dict[str, type] = {
    "recursive": RecursiveChunker,
    "fixed": FixedSizeChunker,
    "markdown": MarkdownChunker,
    "page": PageChunker,
}


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Return the chunker named by `strategy` (default: settings.chunking_strategy)."""
    from app.config import get_settings

    settings = get_settings()
    name = (strategy or settings.chunking_strategy).strip().lower()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if name not in _REGISTRY:
        raise ValueError(f"Unknown CHUNKING_STRATEGY: {name!r} (choices: {', '.join(_REGISTRY)})")
    return _REGISTRY[name](size=size, overlap=overlap)
