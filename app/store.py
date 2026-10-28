"""Thin wrapper over Qdrant (cloud, local on-disk, or in-memory). The source of truth for chunks."""
from __future__ import annotations

import threading
from typing import Any, Iterator

from qdrant_client import QdrantClient, models as qm

from .models import Chunk


class _Serialized:
    """Runs every client call under one lock. Qdrant's local mode (in-memory / on-disk) is not thread-safe;
    the server (Qdrant Cloud) is, so the remote client is used directly."""

    def __init__(self, client: QdrantClient):
        self._client = client
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)
        return call


class VectorStore:
    def __init__(self, collection: str, dim: int, url: str | None = None, api_key: str | None = None,
                 path: str | None = None):
        self._remote = bool(url)
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif path == ":memory:" or not path:
            self.client = _Serialized(QdrantClient(location=":memory:"))
        else:
            self.client = _Serialized(QdrantClient(path=path))
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
            for field in ("doc_id", "content_hash"):
                try:
                    if self._remote:
                        self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
                except Exception:  # local mode does not need/support payload indexes
                    pass
        else:
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = getattr(existing, "size", None)
            if size is not None and size != self.dim:
                raise RuntimeError(
                    f"collection '{self.collection}' has dim {size} but embedder has {self.dim}; "
                    "use another COLLECTION name or re-ingest")

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            self.client.upsert(
                self.collection,
                points=[qm.PointStruct(id=c.chunk_id, vector=v, payload=c.payload())
                        for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])],
                wait=True,
            )

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])),
            wait=True,
        )

    def hash_owners(self, hashes: list[str]) -> dict[str, set[str]]:
        """content_hash -> ids of the documents that already store a chunk with that hash."""
        owners: dict[str, set[str]] = {}
        uniq = list(dict.fromkeys(hashes))
        for i in range(0, len(uniq), 256):
            flt = qm.Filter(must=[qm.FieldCondition(key="content_hash", match=qm.MatchAny(any=uniq[i:i + 256]))])
            offset = None
            while True:  # paginate: one hash can be owned by several documents
                points, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=512, offset=offset,
                                                    with_payload=["content_hash", "doc_id"], with_vectors=False)
                for p in points:
                    owners.setdefault(p.payload["content_hash"], set()).add(p.payload["doc_id"])
                if offset is None:
                    break
        return owners

    def delete_ids(self, ids: list[str]) -> None:
        self.client.delete(self.collection, points_selector=qm.PointIdsList(points=ids), wait=True)

    def search(self, vector: list[float], limit: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        flt = None
        if doc_ids:
            flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])
        res = self.client.query_points(self.collection, query=vector, limit=limit, query_filter=flt,
                                       with_payload=False)
        return [(str(p.id), float(p.score)) for p in res.points]

    def scroll_all(self, batch: int = 1000) -> Iterator[tuple[str, dict[str, Any]]]:
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=batch, offset=offset,
                                                with_payload=True, with_vectors=False)
            for p in points:
                yield str(p.id), p.payload or {}
            if offset is None:
                break

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
