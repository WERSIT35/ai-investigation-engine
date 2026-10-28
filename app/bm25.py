"""Own BM25 (Okapi) implementation with an inverted index.

Supports incremental add/remove so the keyword index stays in sync with document CRUD.
score(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have he her his i in into is it its of on or our
    she that the their them they this to was were what when where which who whom why will with you your
    did does do how than then there these those can could would should also any all about after before""".split()
)


def _stem(tok: str) -> str:
    """Tiny suffix stripper: enough to match plural/verb forms without a dependency."""
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf) and not tok.endswith("ss"):
            return tok[: -len(suf)]
    return tok


_PARTS = re.compile(r"[.\-_]")


def tokenize(text: str) -> list[str]:
    """Words are stemmed. Identifiers like ledger-service / CHG-3301 / payments.settled are kept whole
    (exact-ID matches matter in incident data) and also split into stemmed parts for partial matches."""
    out: list[str] = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS:
            continue
        if _PARTS.search(t):
            out.append(t)
            out.extend(_stem(p) for p in _PARTS.split(t) if p and p not in STOPWORDS)
        else:
            out.append(_stem(t))
    return out


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_len: dict[str, int] = {}
        self._doc_terms: dict[str, tuple[str, ...]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def __contains__(self, key: str) -> bool:
        return key in self.doc_len

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self.doc_len) if self.doc_len else 0.0

    def add(self, key: str, text: str) -> None:
        if key in self.doc_len:
            self.remove(key)
        tf = Counter(tokenize(text))
        for term, n in tf.items():
            self.postings[term][key] = n
        length = sum(tf.values())
        self.doc_len[key] = length
        self._doc_terms[key] = tuple(tf)
        self._total_len += length

    def remove(self, key: str) -> None:
        if key not in self.doc_len:
            return
        for term in self._doc_terms.pop(key):
            plist = self.postings.get(term)
            if plist is not None:
                plist.pop(key, None)
                if not plist:
                    del self.postings[term]
        self._total_len -= self.doc_len.pop(key)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        n = len(self.doc_len)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.doc_len:
            return []
        avgdl = self.avgdl or 1.0
        scores: dict[str, float] = defaultdict(float)
        for term, qtf in Counter(tokenize(query)).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for key, tf in plist.items():
                if allowed is not None and key not in allowed:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[key] / avgdl)
                scores[key] += idf * tf * (self.k1 + 1) / denom * (1 + 0.1 * (qtf - 1))
        return heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
