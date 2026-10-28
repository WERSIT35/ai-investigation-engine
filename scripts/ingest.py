"""Bulk ingestion into whatever Qdrant the .env points at (run locally once, the deployed app reuses it).

Usage:
  python scripts/ingest.py --synthetic                      # 9 synthetic incident docs
  python scripts/ingest.py --corpus data/hotpot/corpus.jsonl --limit 60000
  python scripts/ingest.py --synthetic --corpus data/hotpot/corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.engine import Engine, IngestReport  # noqa: E402
from app.loaders import SUPPORTED_EXTENSIONS, load_bytes  # noqa: E402
from app.models import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def iter_corpus(path: Path, limit: int | None):
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            yield Document(doc_id=r["doc_id"], title=r["title"], text=r["text"], source=path.name, fmt="json",
                           metadata=r.get("metadata", {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--strategy", choices=["recursive", "fixed"])
    args = ap.parse_args()

    s = get_settings()
    engine = Engine(s, make_embedder(s.embed_backend, s.embed_model, s.embed_batch))
    engine.rebuild_from_store()
    total = IngestReport()

    if args.synthetic:
        docs = []
        for p in sorted((ROOT / "data" / "synthetic").iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.extend(load_bytes(p.name, p.read_bytes()))
        rep = engine.ingest(docs, strategy=args.strategy)
        print(f"synthetic: {rep.to_dict()['counts']} chunks={rep.chunks_added}")
        total.merge(rep)

    if args.corpus:
        t0, batch, done = time.time(), [], 0
        for doc in iter_corpus(args.corpus, args.limit):
            batch.append(doc)
            if len(batch) >= args.batch:
                rep = engine.ingest(batch, strategy=args.strategy)
                total.merge(rep)
                done += len(batch)
                batch = []
                rate = done / (time.time() - t0)
                print(f"{done:>7} docs  {rate:6.1f} docs/s  chunks={total.chunks_added} dup={total.duplicate_chunks_skipped}",
                      flush=True)
        if batch:
            total.merge(engine.ingest(batch, strategy=args.strategy))

    print("TOTAL", json.dumps(total.to_dict()["counts"]), "chunks:", total.chunks_added,
          "duplicates skipped:", total.duplicate_chunks_skipped, "seconds:", round(total.seconds, 1))
    print("index:", engine.stats())


if __name__ == "__main__":
    main()
