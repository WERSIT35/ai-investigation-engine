"""Engine = ingestion pipeline + hybrid retrieval over one Qdrant collection.

Qdrant is the persistent source of truth (vectors + chunk payloads). On startup the engine scrolls the
collection once to rebuild the in-memory BM25 index and the document registry, so a restart of the
free-tier host loses nothing.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

from .bm25 import BM25Index
from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder
from .fusion import diversify_by_doc, reciprocal_rank_fusion
from .models import Chunk, Document, SearchHit
from .store import VectorStore

log = logging.getLogger(__name__)
Mode = Literal["hybrid", "dense", "bm25"]


@dataclass
class DocInfo:
    doc_id: str
    title: str
    source: str
    fmt: str
    n_chunks: int
    doc_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicate_docs: list[str] = field(default_factory=list)  # every chunk already stored under another document
    chunks_added: int = 0
    duplicate_chunks_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def merge(self, other: "IngestReport") -> None:
        for f in ("added", "updated", "unchanged", "duplicate_docs", "errors"):
            getattr(self, f).extend(getattr(other, f))
        self.chunks_added += other.chunks_added
        self.duplicate_chunks_skipped += other.duplicate_chunks_skipped
        self.seconds += other.seconds

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for f in ("added", "updated", "unchanged", "duplicate_docs"):  # keep API responses small for bulk loads
            if len(d[f]) > 50:
                d[f] = d[f][:50] + [f"... (+{len(d[f]) - 50} more)"]
        d["counts"] = {f: len(getattr(self, f)) for f in ("added", "updated", "unchanged", "duplicate_docs", "errors")}
        return d


class Engine:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore | None = None):
        self.s = settings
        self.embedder = embedder
        self.store = store or VectorStore(
            collection=settings.collection, dim=embedder.dim, url=settings.qdrant_url,
            api_key=settings.qdrant_api_key, path=settings.qdrant_path)
        self.bm25 = BM25Index()
        self.chunks: dict[str, dict[str, Any]] = {}  # chunk_id -> payload (cache of Qdrant payloads)
        self.docs: dict[str, DocInfo] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        # Two locks: writers (ingest/delete) are serialised by _write_lock, which is held across slow embedding
        # and Qdrant calls; the in-memory index (BM25, chunk/doc maps) is guarded by _index_lock, held only for
        # fast in-memory work. FastAPI runs requests in a thread pool, so searches must never iterate these maps
        # while an upload mutates them, but they also must not wait for a whole upload to embed.
        self._write_lock = threading.RLock()
        self._index_lock = threading.Lock()
        self.version = 0  # bumped on every content change; part of the answer-cache key

    # ------------------------------------------------------------------ startup
    def rebuild_from_store(self) -> int:
        t = time.time()
        with self._write_lock, self._index_lock:
            self.bm25 = BM25Index()
            self.chunks.clear()
            self.docs.clear()
            self._doc_chunks.clear()
            for cid, payload in self.store.scroll_all():
                self._register_chunk(cid, payload)
            for info in self.docs.values():
                info.n_chunks = len(self._doc_chunks[info.doc_id])
        log.info("rebuilt index: %d chunks / %d docs in %.1fs", len(self.chunks), len(self.docs), time.time() - t)
        return len(self.chunks)

    def _register_chunk(self, cid: str, payload: dict[str, Any]) -> None:  # caller holds _index_lock
        self.chunks[cid] = payload
        self.bm25.add(cid, f"{payload.get('title', '')}\n{payload.get('text', '')}")
        doc_id = payload["doc_id"]
        self._doc_chunks.setdefault(doc_id, []).append(cid)
        if doc_id not in self.docs:
            meta = dict(payload.get("metadata") or {})
            self.docs[doc_id] = DocInfo(doc_id=doc_id, title=payload.get("title", ""), source=payload.get("source", ""),
                                        fmt=meta.pop("fmt", ""), n_chunks=0,
                                        doc_hash=meta.pop("doc_hash", ""), metadata=meta)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, docs: Iterable[Document], strategy: str | None = None, size: int | None = None,
               overlap: int | None = None) -> IngestReport:
        """Add new documents, replace changed ones, skip unchanged ones.

        New chunks are embedded and written to Qdrant *before* the old version of an updated document is
        removed, so a failure while embedding never loses the existing copy.
        """
        t = time.time()
        rep = IngestReport()
        strategy = strategy or self.s.chunk_strategy
        size = size or self.s.chunk_size
        overlap = self.s.chunk_overlap if overlap is None else overlap
        with self._write_lock:
            new_chunks: list[Chunk] = []
            replaced: list[str] = []
            seen_hashes: set[str] = set()
            for doc in docs:
                doc_hash = doc.content_hash
                existing = self.docs.get(doc.doc_id)
                if existing and existing.doc_hash == doc_hash:
                    rep.unchanged.append(doc.doc_id)
                    continue
                if existing:
                    replaced.append(doc.doc_id)
                    rep.updated.append(doc.doc_id)
                else:
                    rep.added.append(doc.doc_id)
                doc.metadata = {**doc.metadata, "doc_hash": doc_hash}
                for ch in chunk_document(doc, strategy, size, overlap):
                    if ch.content_hash in seen_hashes:  # duplicate inside this batch
                        rep.duplicate_chunks_skipped += 1
                        continue
                    seen_hashes.add(ch.content_hash)
                    new_chunks.append(ch)
            # duplicates already stored under *other* documents (chunks of a document being replaced don't
            # count: they are about to be removed)
            if new_chunks:
                owners = self.store.hash_owners([c.content_hash for c in new_chunks])
                replacing = set(replaced)
                before = len(new_chunks)
                new_chunks = [c for c in new_chunks if not owners.get(c.content_hash, set()) - replacing]
                rep.duplicate_chunks_skipped += before - len(new_chunks)

            vectors: list[list[float]] = []
            for i in range(0, len(new_chunks), 512):
                vectors.extend(self.embedder.embed_documents([c.index_text for c in new_chunks[i:i + 512]]))
            # chunk ids are uuid5(doc_id, position): the upsert overwrites old points at the same positions,
            # then only old positions that no longer exist are deleted
            new_ids = {c.chunk_id for c in new_chunks}
            stale = [cid for d in replaced for cid in self._doc_chunks.get(d, []) if cid not in new_ids]
            self.store.upsert(new_chunks, vectors)
            if stale:
                self.store.delete_ids(stale)

            with self._index_lock:
                for doc_id in replaced:
                    self._forget_doc(doc_id)
                for c in new_chunks:
                    self._register_chunk(c.chunk_id, c.payload())
                for doc_id in rep.added + rep.updated:
                    if doc_id in self.docs:
                        self.docs[doc_id].n_chunks = len(self._doc_chunks.get(doc_id, []))
                # a document whose every chunk duplicates another document's is not stored: report it as such
                dup = [d for d in rep.added + rep.updated if d not in self.docs]
                if dup:
                    rep.duplicate_docs.extend(dup)
                    rep.added = [d for d in rep.added if d in self.docs]
                    rep.updated = [d for d in rep.updated if d in self.docs]
                if rep.added or rep.updated:
                    self.version += 1
            rep.chunks_added = len(new_chunks)
        rep.seconds = round(time.time() - t, 3)
        return rep

    def delete(self, doc_id: str) -> bool:
        with self._write_lock:
            if doc_id not in self.docs:
                return False
            self.store.delete_doc(doc_id)
            with self._index_lock:
                self._forget_doc(doc_id)
                self.version += 1
            return True

    def _forget_doc(self, doc_id: str) -> None:  # caller holds _index_lock
        for cid in self._doc_chunks.pop(doc_id, []):
            self.chunks.pop(cid, None)
            self.bm25.remove(cid)
        self.docs.pop(doc_id, None)

    # ------------------------------------------------------------------ retrieval
    def _hit(self, cid: str, score: float, ranks: dict[str, int] | None = None) -> SearchHit | None:
        p = self.chunks.get(cid)
        if p is None:
            return None
        return SearchHit(chunk_id=cid, doc_id=p["doc_id"], title=p.get("title", ""), text=p.get("text", ""),
                         score=score, source=p.get("source", ""), position=p.get("position", 0),
                         metadata=p.get("metadata") or {}, ranks=ranks or {})

    def search(self, query: str, top_k: int | None = None, mode: Mode = "hybrid",
               doc_ids: list[str] | None = None, diversify: bool = True) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        top_k = top_k or self.s.top_k
        pool = max(self.s.candidate_pool, top_k * 3)
        lists: dict[str, list[str]] = {}
        if mode in ("hybrid", "dense"):  # Qdrant round trip: outside the index lock
            lists["dense"] = [cid for cid, _ in self.store.search(self.embedder.embed_query(query), pool, doc_ids)]
        with self._index_lock:
            if mode in ("hybrid", "bm25"):
                allowed = {c for d in doc_ids for c in self._doc_chunks.get(d, [])} if doc_ids else None
                lists["bm25"] = [cid for cid, _ in self.bm25.search(query, pool, allowed)]
            fused = reciprocal_rank_fusion(lists, k=self.s.rrf_k)
            hits = [h for cid, score, ranks in fused if (h := self._hit(cid, score, ranks))]
        if diversify:
            hits = diversify_by_doc(hits, lambda h: h.doc_id, self.s.max_chunks_per_doc, top_k)
        return hits[:top_k]

    def get_document_chunks(self, doc_id: str) -> list[SearchHit]:
        with self._index_lock:
            hits = [h for c in self._doc_chunks.get(doc_id, []) if (h := self._hit(c, 1.0))]
        return sorted(hits, key=lambda h: h.position)

    def list_docs(self) -> list[DocInfo]:
        with self._index_lock:
            return list(self.docs.values())

    def find_docs(self, needle: str, limit: int = 5) -> list[DocInfo]:
        n = needle.lower().strip()
        docs = self.list_docs()
        exact = [d for d in docs if d.doc_id.lower() == n or d.title.lower() == n]
        if exact:
            return exact[:limit]
        return [d for d in docs if n in d.title.lower() or n in d.doc_id.lower()][:limit]

    def stats(self) -> dict[str, Any]:
        return {"documents": len(self.docs), "chunks": len(self.chunks), "bm25_terms": len(self.bm25.postings),
                "embedder": self.embedder.name, "dim": self.embedder.dim, "collection": self.s.collection}
