"""File parsing: bytes -> Document(s). Supports .txt .md .pdf .json .jsonl .html/.htm."""
from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import PurePath
from typing import Any

from .models import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".json", ".jsonl", ".html", ".htm"}


class UnsupportedFormatError(ValueError):
    pass


class ParseError(ValueError):
    pass


_WS_LINES = re.compile(r"[ \t]+")
_MANY_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(_WS_LINES.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NL.sub("\n\n", text).strip()


def _default_doc_id(filename: str) -> str:
    return "file::" + PurePath(filename).name.lower()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


# ---------- front matter (simple "key: value" block between --- lines) ----------
_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or fallback
        if s:
            return s[:120] if len(s) <= 120 else fallback
    return fallback


def _doc_from_meta(meta: dict[str, Any], body: str, filename: str, fmt: str) -> Document:
    meta = dict(meta)
    doc_id = str(meta.pop("doc_id", "") or _default_doc_id(filename))
    stem = PurePath(filename).stem
    title = str(meta.pop("title", "") or _title_from_text(body, stem))
    return Document(doc_id=doc_id, title=title, text=clean_text(body), source=PurePath(filename).name,
                    fmt=fmt, metadata=meta)


# ---------- per-format loaders ----------
def _load_text(filename: str, data: bytes, fmt: str) -> list[Document]:
    meta, body = _parse_front_matter(_decode(data))
    return [_doc_from_meta(meta, body, filename, fmt)]


def _load_pdf(filename: str, data: bytes) -> list[Document]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as exc:  # pypdf raises many types
        raise ParseError(f"cannot read PDF {filename}: {exc}") from exc
    body = "\n\n".join(pages)
    meta, body = _parse_front_matter(body)
    info = reader.metadata or {}
    if info.get("/Title") and "title" not in meta:
        meta["title"] = str(info["/Title"])
    meta["pages"] = len(pages)
    return [_doc_from_meta(meta, body, filename, "pdf")]


def _record_to_doc(rec: dict[str, Any], filename: str, idx: int, fmt: str) -> Document | None:
    text = rec.get("text") or rec.get("content") or rec.get("body")
    if isinstance(text, list):
        text = " ".join(map(str, text))
    if not text or not str(text).strip():
        return None
    meta = dict(rec.get("metadata") or {})
    for k, v in rec.items():
        if k not in {"text", "content", "body", "metadata", "doc_id", "id", "title"} and isinstance(v, (str, int, float)):
            meta.setdefault(k, v)
    doc_id = rec.get("doc_id") or rec.get("id") or f"{_default_doc_id(filename)}#{idx}"
    title = rec.get("title") or _title_from_text(str(text), f"{PurePath(filename).stem} #{idx}")
    return Document(doc_id=str(doc_id), title=str(title), text=clean_text(str(text)),
                    source=PurePath(filename).name, fmt=fmt, metadata=meta)


def _load_json(filename: str, data: bytes) -> list[Document]:
    try:
        obj = json.loads(_decode(data))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON in {filename}: {exc}") from exc
    records = obj if isinstance(obj, list) else (obj.get("documents") if isinstance(obj, dict) and "documents" in obj else [obj])
    docs = [_record_to_doc(r, filename, i, "json") for i, r in enumerate(records) if isinstance(r, dict)]
    docs = [d for d in docs if d]
    if not docs:
        raise ParseError(f"{filename}: no records with a text/content field")
    return docs


def _load_jsonl(filename: str, data: bytes) -> list[Document]:
    docs: list[Document] = []
    for i, line in enumerate(_decode(data).splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{filename}: bad JSON on line {i + 1}: {exc}") from exc
        d = _record_to_doc(rec, filename, i, "json") if isinstance(rec, dict) else None
        if d:
            docs.append(d)
    return docs


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "table"}
    SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if a.get("name") and a.get("content"):
                self.meta[a["name"].lower()] = a["content"]
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _load_html(filename: str, data: bytes) -> list[Document]:
    p = _HTMLText()
    p.feed(_decode(data))
    meta: dict[str, Any] = dict(p.meta)
    if p.title.strip():
        meta.setdefault("title", p.title.strip())
    return [_doc_from_meta(meta, "".join(p.parts), filename, "html")]


def load_bytes(filename: str, data: bytes) -> list[Document]:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if ext in {".txt"}:
        docs = _load_text(filename, data, "txt")
    elif ext in {".md", ".markdown"}:
        docs = _load_text(filename, data, "md")
    elif ext == ".pdf":
        docs = _load_pdf(filename, data)
    elif ext == ".json":
        docs = _load_json(filename, data)
    elif ext == ".jsonl":
        docs = _load_jsonl(filename, data)
    else:
        docs = _load_html(filename, data)
    docs = [d for d in docs if d.text]
    if not docs:
        raise ParseError(f"{filename}: no extractable text")
    return docs
