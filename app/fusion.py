"""Result fusion and re-ranking — own algorithmic components.

reciprocal_rank_fusion  — merges ranked lists from different retrievers (dense, BM25) using only ranks,
                          so incomparable score scales (cosine vs BM25) don't matter.
                          RRF(d) = sum_r w_r / (k + rank_r(d))
diversify_by_doc        — caps chunks per document so the context window covers several documents,
                          which is what multi-hop questions need (otherwise one long doc floods top-k).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Sequence, TypeVar

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[K]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[K, float, dict[str, int]]]:
    """Returns [(key, fused_score, {retriever: 1-based rank})] sorted by score desc."""
    if k <= 0:
        raise ValueError("k must be > 0")
    weights = weights or {}
    scores: dict[K, float] = defaultdict(float)
    ranks: dict[K, dict[str, int]] = defaultdict(dict)
    for name, keys in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, key in enumerate(keys, start=1):
            if name in ranks[key]:  # duplicate within one list: keep best rank
                continue
            scores[key] += w / (k + rank)
            ranks[key][name] = rank
    # tie-break: items found by more retrievers first, then best single rank
    ordered = sorted(scores, key=lambda key: (-scores[key], -len(ranks[key]), min(ranks[key].values())))
    return [(key, scores[key], ranks[key]) for key in ordered]


def diversify_by_doc(items: Sequence[T], doc_of: Callable[[T], str], max_per_doc: int, limit: int) -> list[T]:
    """Keep ranking order but allow at most `max_per_doc` items per doc; backfill if not enough docs."""
    if max_per_doc <= 0:
        return list(items[:limit])
    picked: list[T] = []
    overflow: list[T] = []
    per_doc: dict[str, int] = defaultdict(int)
    for it in items:
        d = doc_of(it)
        if per_doc[d] < max_per_doc:
            picked.append(it)
            per_doc[d] += 1
        else:
            overflow.append(it)
        if len(picked) >= limit:
            return picked
    return (picked + overflow)[:limit]
