"""Runtime configuration. Everything comes from environment variables (.env in dev, secrets in prod)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Vector store. QDRANT_URL set -> Qdrant Cloud; else local on-disk mode at QDRANT_PATH; ":memory:" for tests.
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data"))
    collection: str = field(default_factory=lambda: os.getenv("COLLECTION", "docs"))

    # Embeddings: "fastembed" (real, ONNX, CPU) or "hash" (offline/tests only).
    embed_backend: str = field(default_factory=lambda: os.getenv("EMBED_BACKEND", "fastembed"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_batch: int = field(default_factory=lambda: _int("EMBED_BATCH", 64))

    # LLM: any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, ...).
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or None)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 60))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 3))

    # Chunking
    chunk_strategy: str = field(default_factory=lambda: os.getenv("CHUNK_STRATEGY", "recursive"))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))  # characters
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))

    # Retrieval
    top_k: int = field(default_factory=lambda: _int("TOP_K", 8))
    candidate_pool: int = field(default_factory=lambda: _int("CANDIDATE_POOL", 50))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    max_chunks_per_doc: int = field(default_factory=lambda: _int("MAX_CHUNKS_PER_DOC", 2))

    # Agent
    max_agent_steps: int = field(default_factory=lambda: _int("MAX_AGENT_STEPS", 4))

    # API protection
    admin_token: str | None = field(default_factory=lambda: os.getenv("ADMIN_TOKEN") or None)
    ask_rate_per_min: int = field(default_factory=lambda: _int("ASK_RATE_PER_MIN", 10))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 10))


@lru_cache
def get_settings() -> Settings:
    return Settings()
