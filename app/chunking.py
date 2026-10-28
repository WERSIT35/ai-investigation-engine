"""Two chunking strategies, compared in eval/chunking_experiment.py.

fixed      — sliding window of ~size characters, cut on word boundaries, fixed overlap.
recursive  — structure-aware: split by paragraphs -> sentences -> words, then greedily pack
             whole units up to `size`; overlap carries the trailing sentence(s) of the previous chunk.
"""
from __future__ import annotations

import re

from .models import Chunk, Document, chunk_uuid, content_hash

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    overlap = max(0, min(overlap, size // 2))
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        length, end = 0, start
        while end < len(words) and (length + len(words[end]) + 1 <= size or end == start):
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # step back by ~overlap characters
        back, ov = end, 0
        while back > start + 1 and ov < overlap:
            back -= 1
            ov += len(words[back]) + 1
        start = back if back > start else end
    return chunks


def _split_units(text: str, size: int) -> list[str]:
    """Break text into units no longer than `size`, preferring paragraph > sentence > word boundaries."""
    units: list[str] = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if len(para) <= size:
            units.append(para)
            continue
        for sent in [s.strip() for s in _SENT.split(para) if s.strip()]:
            if len(sent) <= size:
                units.append(sent)
            else:
                units.extend(fixed_chunks(sent, size, 0))
    return units


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    units = _split_units(text, size)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            # overlap: keep trailing units of the finished chunk that fit in `overlap`
            tail: list[str] = []
            tlen = 0
            for prev in reversed(cur):
                if tlen + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tlen += len(prev) + 1
            cur, cur_len = tail, tlen
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        last = " ".join(cur)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "recursive": recursive_chunks}


def chunk_document(doc: Document, strategy: str = "recursive", size: int = 800, overlap: int = 120) -> list[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"unknown chunk strategy '{strategy}', use one of {list(STRATEGIES)}")
    pieces = fn(doc.text, size, overlap) if doc.text else []
    meta = {**doc.metadata, "fmt": doc.fmt, "chunk_strategy": strategy}
    return [
        Chunk(chunk_id=chunk_uuid(doc.doc_id, i), doc_id=doc.doc_id, title=doc.title, text=p,
              position=i, source=doc.source, content_hash=content_hash(p), metadata=meta)
        for i, p in enumerate(pieces)
    ]
