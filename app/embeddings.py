"""Embedding backends.

FastEmbedder — BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU, no torch -> fits free hosting).
HashEmbedder — deterministic feature hashing; no model download. Used in tests / offline smoke runs only.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol, Sequence

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        import os

        from fastembed import TextEmbedding

        self.name = model_name
        # FASTEMBED_CACHE_PATH lets the Docker image bake the model in at build time (no download at startup)
        self._model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        self._batch = batch_size
        self.dim = len(next(iter(self._model.embed(["probe"]))))
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        # cache query embeddings: repeated / multi-hop queries are common and embedding is the hot path
        self.embed_query = lru_cache(maxsize=2048)(self._embed_query)  # type: ignore[method-assign]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts), batch_size=self._batch)]

    def _embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._query_prefix + text]))).tolist()


_TOK = re.compile(r"\w+")


class HashEmbedder:
    """Bag of hashed unigrams+bigrams, L2-normalised. Lexical only — never use for real retrieval."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = _TOK.findall(text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embedder(backend: str, model_name: str, batch_size: int = 64) -> Embedder:
    if backend == "fastembed":
        return FastEmbedder(model_name, batch_size)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBED_BACKEND '{backend}'")
