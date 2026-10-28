"""Evaluation metrics (retrieval + answer + citation + abstention)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


def unique_in_order(xs: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(xs))


def recall_at_k(retrieved_docs: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(unique_in_order(retrieved_docs)[:k])
    return len(top & set(gold)) / len(set(gold))


def all_gold_at_k(retrieved_docs: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 if every gold document is in the top-k docs (what a multi-hop question actually needs)."""
    if not gold:
        return float("nan")
    return float(set(gold) <= set(unique_in_order(retrieved_docs)[:k]))


def mrr(retrieved_docs: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(unique_in_order(retrieved_docs)[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


# ---- SQuAD/HotpotQA-style answer normalisation
def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"%", ".", ":"})
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)  # drop dots except decimal points
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def contains(text: str, needle: str) -> bool:
    n = normalize_answer(needle)
    if not n:
        return False
    t = normalize_answer(text)
    if n in {"yes", "no"}:  # avoid matching "no" inside "not" etc.
        return re.search(rf"\b{n}\b", t) is not None
    return n in t


def answer_correct(item: dict, answer: str, short: str) -> bool:
    """Lenient correctness used for the headline number (LLM answers are long, so EM alone undercounts)."""
    if item.get("must_include"):
        return all(contains(answer + " " + short, m) for m in item["must_include"])
    gold = item.get("answer") or ""
    if normalize_answer(gold) in {"yes", "no"}:
        return normalize_answer(short).startswith(normalize_answer(gold)) or (
            not short and contains(answer.split(".")[0], gold))
    return contains(short, gold) or contains(answer, gold) or token_f1(short, gold) >= 0.6


def citation_scores(cited_docs: Sequence[str], gold: Sequence[str]) -> tuple[float, float]:
    """(precision, recall) of cited documents vs gold supporting documents."""
    cited = set(cited_docs)
    if not gold:
        return float("nan"), float("nan")
    if not cited:
        return 0.0, 0.0
    hit = len(cited & set(gold))
    return hit / len(cited), hit / len(set(gold))


def mean(xs) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")
