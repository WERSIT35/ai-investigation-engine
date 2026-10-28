"""FastAPI application: REST API + static single-page UI."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import InvestigationAgent
from .config import Settings, get_settings
from .embeddings import make_embedder
from .engine import Engine
from .llm import LLMError, OpenAICompatLLM
from .loaders import ParseError, UnsupportedFormatError, load_bytes
from .models import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------- request models
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


class TextDocument(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=2_000_000)
    doc_id: str | None = Field(default=None, max_length=300)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- helpers
class RateLimiter:
    """Sliding-window limiter per client IP. Protects the free LLM quota on the public demo."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, f"rate limit: {self.per_minute} questions per minute")
            q.append(now)


def client_ip(request: Request, trust_proxy: bool) -> str:
    """Key for per-client rate limiting. Without a proxy it is the TCP peer. Behind one, every visitor shares the
    proxy's address, so the client is read from X-Forwarded-For: the right-most entry is the one our own proxy
    appended; entries to its left come from the client and can be forged."""
    if trust_proxy:
        last = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


def _build_default(settings: Settings) -> tuple[Engine, InvestigationAgent]:
    engine = Engine(settings, make_embedder(settings.embed_backend, settings.embed_model, settings.embed_batch))
    engine.rebuild_from_store()
    try:
        llm = OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                              settings.llm_timeout, settings.llm_max_retries)
    except LLMError as exc:
        log.warning("LLM disabled: %s — /ask will return degraded answers", exc)
        llm = None
    return engine, InvestigationAgent(engine, llm, max_steps=settings.max_agent_steps)


def create_app(settings: Settings | None = None, engine: Engine | None = None,
               agent: InvestigationAgent | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, Any] = {"engine": engine, "agent": agent}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["engine"] is None:
            state["engine"], state["agent"] = _build_default(settings)
        elif state["agent"] is None:
            state["agent"] = InvestigationAgent(state["engine"], None, max_steps=settings.max_agent_steps)
        yield

    app = FastAPI(title="AI Investigation Engine", version="1.0.0", lifespan=lifespan)
    limiter = RateLimiter(settings.ask_rate_per_min)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    def eng() -> Engine:
        return state["engine"]

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if settings.admin_token and x_admin_token != settings.admin_token:
            raise HTTPException(401, "write operations need a valid X-Admin-Token header")

    async def _read_upload(f: UploadFile) -> list[Document]:
        data = await f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"{f.filename}: larger than {settings.max_upload_mb} MB")
        try:
            return load_bytes(f.filename or "upload.txt", data)
        except UnsupportedFormatError as exc:
            raise HTTPException(415, str(exc))
        except ParseError as exc:
            raise HTTPException(422, str(exc))

    # ------------------------------------------------------------ meta
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "llm": bool(state["agent"] and state["agent"].llm), **eng().stats()}

    # ------------------------------------------------------------ documents
    @app.post("/documents", dependencies=[Depends(require_admin)])
    async def upload_documents(files: list[UploadFile] = File(...),
                               chunk_strategy: Literal["recursive", "fixed"] | None = Form(default=None)):
        docs: list[Document] = []
        for f in files:
            docs.extend(await _read_upload(f))
        return eng().ingest(docs, strategy=chunk_strategy).to_dict()

    @app.post("/documents/text", dependencies=[Depends(require_admin)])
    def add_text_document(body: TextDocument):
        doc_id = body.doc_id or "text::" + body.title.strip().lower().replace(" ", "-")[:120]
        doc = Document(doc_id=doc_id, title=body.title.strip(), text=body.text, source="api", fmt="txt",
                       metadata=body.metadata)
        return eng().ingest([doc]).to_dict()

    @app.put("/documents/{doc_id:path}", dependencies=[Depends(require_admin)])
    async def replace_document(doc_id: str, file: UploadFile = File(...)):
        if doc_id not in eng().docs:
            raise HTTPException(404, f"document '{doc_id}' not found")
        docs = await _read_upload(file)
        if len(docs) != 1:
            raise HTTPException(422, "replacement file must contain exactly one document")
        docs[0].doc_id = doc_id
        return eng().ingest(docs).to_dict()

    @app.delete("/documents/{doc_id:path}", dependencies=[Depends(require_admin)])
    def delete_document(doc_id: str):
        if not eng().delete(doc_id):
            raise HTTPException(404, f"document '{doc_id}' not found")
        return {"deleted": doc_id}

    @app.get("/documents")
    def list_documents(q: str = "", offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
        docs = eng().list_docs()
        if q:
            ql = q.lower()
            docs = [d for d in docs if ql in d.title.lower() or ql in d.doc_id.lower()]
        docs.sort(key=lambda d: (not d.doc_id.startswith(("file::", "syn::", "text::")), d.title.lower()))
        return {"total": len(docs), "offset": offset,
                "items": [vars(d) for d in docs[offset:offset + limit]]}

    @app.get("/documents/{doc_id:path}")
    def get_document(doc_id: str):
        e = eng()
        if doc_id not in e.docs:
            raise HTTPException(404, f"document '{doc_id}' not found")
        return {**vars(e.docs[doc_id]),
                "chunks": [{"chunk_id": h.chunk_id, "position": h.position, "text": h.text}
                           for h in e.get_document_chunks(doc_id)]}

    # ------------------------------------------------------------ retrieval / QA
    @app.post("/search")
    def search(body: SearchRequest):
        t = time.time()
        hits = eng().search(body.query, top_k=body.top_k, mode=body.mode)
        return {"query": body.query, "mode": body.mode, "latency_s": round(time.time() - t, 3),
                "hits": [vars(h) for h in hits]}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        limiter.check(client_ip(request, settings.trust_proxy))
        try:
            return state["agent"].ask(body.question, mode=body.mode).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    # ------------------------------------------------------------ UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
