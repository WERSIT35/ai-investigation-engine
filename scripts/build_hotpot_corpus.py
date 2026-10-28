"""Builds the 50k+ document corpus from HotpotQA (distractor setting).

Every unique Wikipedia paragraph in the HotpotQA validation contexts becomes one document
(doc_id = "hp::<title>"). Validation alone gives ~66k unique paragraphs; if --target is higher, train
examples are added. Questions (with gold supporting titles) are saved for the evaluation set.

Usage:  python scripts/build_hotpot_corpus.py --target 60000
Output: data/hotpot/corpus.jsonl, data/hotpot/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hotpot"
RAW_URLS = {
    "validation": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
}


def _normalise(ex: dict) -> dict:
    """Accept both the HF datasets layout and the original JSON layout."""
    ctx = ex["context"]
    if isinstance(ctx, dict):  # HF: {"title": [...], "sentences": [[...]]}
        paragraphs = list(zip(ctx["title"], ctx["sentences"]))
        sf = ex["supporting_facts"]
        supporting = list(dict.fromkeys(sf["title"]))
    else:  # raw: [[title, [sent, ...]], ...]
        paragraphs = [(t, s) for t, s in ctx]
        supporting = list(dict.fromkeys(t for t, _ in ex["supporting_facts"]))
    return {"id": ex.get("id") or ex.get("_id"), "question": ex["question"], "answer": ex["answer"],
            "type": ex["type"], "level": ex["level"], "paragraphs": paragraphs, "supporting": supporting}


def load_split(split: str):
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        print(f"loaded {split} from Hugging Face datasets ({len(ds)} examples)")
        return (_normalise(ex) for ex in ds)
    except Exception as exc:  # no `datasets`, no HF access, ...
        print(f"HF datasets unavailable ({exc}); downloading raw JSON")
        cache = OUT / f"raw_{split}.json"
        if not cache.exists():
            urllib.request.urlretrieve(RAW_URLS[split], cache)
        return (_normalise(ex) for ex in json.loads(cache.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60000, help="minimum number of documents")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_docs = 0
    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as corpus, \
            (OUT / "questions.jsonl").open("w", encoding="utf-8") as qf:
        for split in ("validation", "train"):
            if split == "train" and n_docs >= args.target:
                break
            for ex in load_split(split):
                for title, sents in ex["paragraphs"]:
                    if title in seen:
                        continue
                    seen.add(title)
                    text = "".join(sents).strip()
                    if not text:
                        continue
                    corpus.write(json.dumps({"doc_id": f"hp::{title}", "title": title, "text": text,
                                             "metadata": {"dataset": "hotpotqa", "split": split}}) + "\n")
                    n_docs += 1
                if split == "validation":
                    qf.write(json.dumps({k: ex[k] for k in ("id", "question", "answer", "type", "level", "supporting")}) + "\n")
                if split == "train" and n_docs >= args.target:
                    break
    print(f"corpus: {n_docs} documents -> {OUT / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
