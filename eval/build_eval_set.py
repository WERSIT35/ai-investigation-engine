"""Builds eval/eval_set.jsonl: 30 HotpotQA questions (15 bridge, 15 comparison) + the synthetic set.

Deterministic (seed 42) so results are reproducible. Requires data/hotpot/questions.jsonl
(scripts/build_hotpot_corpus.py). Without it, only the synthetic questions are written.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(n_bridge: int = 15, n_comparison: int = 15, seed: int = 42) -> None:
    items = json.loads((ROOT / "eval" / "synthetic_questions.json").read_text(encoding="utf-8"))
    qpath = ROOT / "data" / "hotpot" / "questions.jsonl"
    if qpath.exists():
        qs = [json.loads(l) for l in qpath.open(encoding="utf-8")]
        rng = random.Random(seed)
        picked = []
        for typ, n in (("bridge", n_bridge), ("comparison", n_comparison)):
            pool = [q for q in qs if q["type"] == typ and len(q["answer"]) <= 60]
            picked += rng.sample(pool, n)
        for q in picked:
            items.append({"id": f"hp-{q['id']}", "type": "multi_hop" if q["type"] == "bridge" else "comparison",
                          "question": q["question"], "answer": q["answer"],
                          "gold_docs": [f"hp::{t}" for t in q["supporting"]], "source": "hotpotqa"})
    else:
        print("data/hotpot/questions.jsonl missing -> synthetic questions only")
    out = ROOT / "eval" / "eval_set.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    by_type: dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    print(f"{len(items)} questions -> {out}  {by_type}")


if __name__ == "__main__":
    main()
