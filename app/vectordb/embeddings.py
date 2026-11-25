"""MILESTONE 4 — Embeddings.

Turn text into vectors so that similar meaning -> nearby vectors (cosine).

Choose the backend with EMBEDDING_MODEL in .env:
  fastembed:<model>   local ONNX model, no API key, great quality (DEFAULT)
                      e.g. fastembed:BAAI/bge-small-en-v1.5  (384 dims)
  google:<model>      hosted Gemini embeddings (uses GOOGLE_API_KEY)
  hash                deterministic bag-of-words fallback (no model/network) —
                      handy for tests and fully offline runs

All backends expose the same tiny interface: dimension / embed_documents / embed_query.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or but of to in on at by for from with about as into over is are was were be been being "
    "do does did doing have has had i me my we our you your he she it its they them their this that these "
    "those what which who whom when where why how can could should would will shall may might must "
    "not no yes so if then than too very just also up down out off all any some such only own same "
    "show tell give get got let please".split()
)


class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class FastEmbedEmbedder:
    """Wraps a local fastembed model. The model file downloads once, then runs offline."""

    relevance_floor = 0.55
    relevance_gate = 0.60

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GoogleEmbedder:
    """Hosted Gemini embeddings (needs GOOGLE_API_KEY)."""

    relevance_floor = 0.55
    relevance_gate = 0.60

    def __init__(self, model_name: str) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._model = GoogleGenerativeAIEmbeddings(model=model_name)
        self._dim = len(self._model.embed_query("dimension probe"))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


class HashEmbedder:
    """Deterministic bag-of-words vectors via hashing: same text -> same vector.

    No model, no network — shared words make vectors point in similar directions,
    which is enough for tests and for a fully offline demo.
    """

    relevance_floor = 0.3

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder() -> Embedder:
    """Build the embedder named by EMBEDDING_MODEL."""
    from app.config import get_settings

    spec = (get_settings().embedding_model or "").strip()

    if spec == "" or spec.lower() == "hash":
        return HashEmbedder()
    if spec.lower().startswith("fastembed:"):
        return FastEmbedEmbedder(spec.split(":", 1)[1])
    if spec.lower().startswith("google:"):
        return GoogleEmbedder(spec.split(":", 1)[1])
    return FastEmbedEmbedder(spec)
