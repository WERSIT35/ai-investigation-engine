"""Chunking strategy comparison (no LLM, in-memory Qdrant, runs in ~1 minute).

Indexes the synthetic incident corpus (multi-paragraph docs, where chunking matters; HotpotQA paragraphs
are single short chunks) with each strategy/size and measures, over the answerable synthetic questions:
  - answer-hit@3 : the gold answer string appears inside one of the top-3 retrieved chunks
  - doc recall@3 : share of gold documents among the top-3 chunks' documents
  - chunk count / mean chunk length (index cost)

Usage: python eval/chunking_experiment.py
Writes eval/results/chunking.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.loaders import SUPPORTED_EXTENSIONS, load_bytes  # noqa: E402
from app.store import VectorStore  # noqa: E402
from eval.metrics import contains, mean, recall_at_k, unique_in_order  # noqa: E402

CONFIGS = [("fixed", 300, 50), ("recursive", 300, 50), ("fixed", 800, 120), ("recursive", 800, 120)]


def main() -> None:
    base = get_settings()
    embedder = make_embedder(base.embed_backend, base.embed_model, base.embed_batch)
    docs = []
    for p in sorted((ROOT / "data" / "synthetic").iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_bytes(p.name, p.read_bytes()))
    qs = [q for q in json.loads((ROOT / "eval" / "synthetic_questions.json").read_text())
          if q["type"] != "unanswerable" and q.get("must_include")]

    lines = ["# Chunking strategy comparison", "",
             f"Corpus: {len(docs)} synthetic documents · {len(qs)} answerable questions · embedder `{embedder.name}` · "
             "hybrid retrieval, top-3 chunks, no per-doc cap", "",
             "| Strategy | Size / overlap (chars) | Chunks | Mean chunk len | Answer-hit@3 | Doc recall@3 |",
             "|---|---|---|---|---|---|"]
    for strategy, size, overlap in CONFIGS:
        s = replace(base, chunk_strategy=strategy, chunk_size=size, chunk_overlap=overlap)
        store = VectorStore(collection=f"chunk_{strategy}_{size}", dim=embedder.dim, path=":memory:")
        engine = Engine(s, embedder, store=store)
        engine.ingest([replace(d) for d in docs])
        lens = [len(p["text"]) for p in engine.chunks.values()]
        hit, rec = [], []
        for q in qs:
            hits = engine.search(q["question"], top_k=3, diversify=False)
            text = " ".join(h.text for h in hits)
            hit.append(float(all(contains(text, m) for m in q["must_include"])))
            rec.append(recall_at_k(unique_in_order([h.doc_id for h in hits]), q["gold_docs"], 3))
        row = (f"| {strategy} | {size} / {overlap} | {len(lens)} | {mean(lens):.0f} | {mean(hit):.3f} | "
               f"{mean(rec):.3f} |")
        print(row)
        lines.append(row)
    out = ROOT / "eval" / "results" / "chunking.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
