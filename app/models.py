"""Core data types shared by every layer."""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_NS = uuid.UUID("6f1c2a52-8a9e-4d6e-9a43-0f6c2b7d1e10")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str  # original filename / dataset name
    fmt: str  # txt | md | pdf | json | html
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return content_hash(self.title + "\n" + self.text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int
    source: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for both embedding and BM25: title acts as a contextual header for the chunk."""
        return f"{self.title}\n{self.text}"

    def payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "position": self.position,
            "source": self.source,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    source: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)  # rank per retriever, for explainability


def chunk_uuid(doc_id: str, position: int) -> str:
    return str(uuid.uuid5(_NS, f"{doc_id}::{position}"))


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", text.lower())).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()
