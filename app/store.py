"""Thin wrapper over Qdrant (cloud, local on-disk, or in-memory). The source of truth for chunks."""
from __future__ import annotations

from typing import Any, Iterator

from qdrant_client import QdrantClient, models as qm

from .models import Chunk


class VectorStore:
    def __init__(self, collection: str, dim: int, url: str | None = None, api_key: str | None = None,
                 path: str | None = None):
        self._remote = bool(url)
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif path == ":memory:" or not path:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=path)
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

    def existing_hashes(self, hashes: list[str]) -> set[str]:
        found: set[str] = set()
        uniq = list(dict.fromkeys(hashes))
        for i in range(0, len(uniq), 256):
            part = uniq[i:i + 256]
            points, _ = self.client.scroll(
                self.collection,
                scroll_filter=qm.Filter(must=[qm.FieldCondition(key="content_hash", match=qm.MatchAny(any=part))]),
                limit=len(part) * 4, with_payload=["content_hash"], with_vectors=False,
            )
            found.update(p.payload["content_hash"] for p in points)
        return found

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
