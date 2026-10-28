from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.embeddings import HashEmbedder  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.loaders import SUPPORTED_EXTENSIONS, load_bytes  # noqa: E402
from app.store import VectorStore  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return replace(Settings(), qdrant_url=None, qdrant_path=":memory:", embed_backend="hash", llm_api_key=None,
                   admin_token=None, chunk_size=400, chunk_overlap=60, ask_rate_per_min=0)


@pytest.fixture
def engine(settings) -> Engine:
    emb = HashEmbedder()
    return Engine(settings, emb, store=VectorStore("test", emb.dim, path=":memory:"))


@pytest.fixture
def synthetic_docs():
    docs = []
    for p in sorted((ROOT / "data" / "synthetic").iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_bytes(p.name, p.read_bytes()))
    return docs


@pytest.fixture
def loaded_engine(engine, synthetic_docs) -> Engine:
    engine.ingest(synthetic_docs)
    return engine
