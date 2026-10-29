"""Runs the evaluation and writes eval/results/<timestamp>.json + eval/results/report.md.

Part A (retrieval, no LLM, cheap): compares retrieval configs (hybrid / dense / bm25) on Recall@5, Recall@10,
         All-gold@10 (every supporting doc found) and MRR@10 for every answerable question.
Part B (end-to-end, LLM): runs the agent for the configs in --e2e and scores answer correctness, EM/F1,
         citation precision/recall, abstention on unanswerable questions, conflict detection,
         evidence recall of the whole investigation, latency and tokens.

Usage:
  python eval/run_eval.py                          # retrieval for 3 configs + agent with hybrid
  python eval/run_eval.py --e2e hybrid,bm25        # two end-to-end configs (2x LLM calls)
  python eval/run_eval.py --e2e none               # retrieval only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import InvestigationAgent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.llm import LLMError, OpenAICompatLLM  # noqa: E402
from eval.metrics import (all_gold_at_k, answer_correct, citation_scores, exact_match, mean, mrr,  # noqa: E402
                          recall_at_k, token_f1, unique_in_order)

RESULTS = ROOT / "eval" / "results"


def load_items(limit: int | None) -> list[dict]:
    path = ROOT / "eval" / "eval_set.jsonl"
    if not path.exists():
        sys.exit("eval/eval_set.jsonl missing: run `python eval/build_eval_set.py` first")
    items = [json.loads(l) for l in path.open(encoding="utf-8")]
    return items[:limit] if limit else items


def retrieval_eval(engine: Engine, items: list[dict], modes: list[str]) -> dict:
    out: dict[str, dict] = {}
    for mode in modes:
        rows = []
        for it in items:
            if not it.get("gold_docs"):
                continue
            t = time.time()
            hits = engine.search(it["question"], top_k=10, mode=mode)
            docs = unique_in_order([h.doc_id for h in hits])
            rows.append({"id": it["id"], "type": it["type"], "latency": time.time() - t,
                         "r5": recall_at_k(docs, it["gold_docs"], 5), "r10": recall_at_k(docs, it["gold_docs"], 10),
                         "all10": all_gold_at_k(docs, it["gold_docs"], 10), "mrr": mrr(docs, it["gold_docs"], 10),
                         "top_docs": docs[:5]})
        out[mode] = {"summary": _summ_retrieval(rows), "by_type": _by_type(rows, _summ_retrieval), "rows": rows}
        s = out[mode]["summary"]
        print(f"[retrieval:{mode:6}] R@5={s['recall@5']:.3f} R@10={s['recall@10']:.3f} "
              f"All@10={s['all_gold@10']:.3f} MRR={s['mrr@10']:.3f} lat={s['latency_ms']:.0f}ms")
    return out


def _summ_retrieval(rows: list[dict]) -> dict:
    return {"n": len(rows), "recall@5": mean(r["r5"] for r in rows), "recall@10": mean(r["r10"] for r in rows),
            "all_gold@10": mean(r["all10"] for r in rows), "mrr@10": mean(r["mrr"] for r in rows),
            "latency_ms": 1000 * mean(r["latency"] for r in rows)}


def _by_type(rows: list[dict], summ) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    return {t: summ(g) for t, g in sorted(groups.items())}


def e2e_eval(agent: InvestigationAgent, items: list[dict], mode: str, sleep: float) -> dict:
    rows = []
    for i, it in enumerate(items, 1):
        a = agent.ask(it["question"], mode=mode, use_cache=False)
        cited_docs = unique_in_order([c.doc_id for c in a.citations])
        seen_docs = unique_in_order([r["doc_id"] for r in a.retrieved])
        unanswerable = it["type"] == "unanswerable"
        row = {"id": it["id"], "type": it["type"], "question": it["question"], "gold": it.get("answer"),
               "answer": a.answer, "short_answer": a.short_answer, "status": a.status,
               "insufficient": a.insufficient, "n_conflicts": len(a.conflicts), "cited_docs": cited_docs,
               "invalid_citations": len(a.invalid_citations), "latency": a.latency_s,
               "llm_calls": a.usage.get("llm_calls", 0),
               "tokens": a.usage.get("prompt_tokens", 0) + a.usage.get("completion_tokens", 0),
               "tool_calls": sum(1 for t in a.trace if t.get("tool") and t["step"] not in (0, "final"))}
        if unanswerable:
            row["abstained"] = float(a.insufficient)
        else:
            row["correct"] = float(not a.insufficient and answer_correct(it, a.answer, a.short_answer))
            row["false_abstention"] = float(a.insufficient)
            if it.get("source") == "hotpotqa":
                row["em"] = exact_match(a.short_answer, it["answer"])
                row["f1"] = token_f1(a.short_answer, it["answer"])
            p, r = citation_scores(cited_docs, it["gold_docs"])
            row["cit_precision"], row["cit_recall"] = p, r
            row["evidence_recall"] = recall_at_k(seen_docs, it["gold_docs"], 10_000)
        if it["type"] == "conflict":
            row["conflict_detected"] = float(len(a.conflicts) > 0)
        rows.append(row)
        flag = "OK " if row.get("correct", row.get("abstained", 0)) else "-- "
        print(f"[e2e:{mode}] {i:>2}/{len(items)} {flag}{it['id']:<14} {a.latency_s:>5.1f}s  {a.short_answer[:60]!r}")
        if a.status != "ok":
            print(f"    status={a.status}: {a.answer[:120]}")
        time.sleep(sleep)
    return {"summary": _summ_e2e(rows), "by_type": _by_type(rows, _summ_e2e), "rows": rows}


def _summ_e2e(rows: list[dict]) -> dict:
    g = lambda k: mean(r[k] for r in rows if k in r)  # noqa: E731
    return {"n": len(rows), "answer_correctness": g("correct"), "hotpot_em": g("em"), "hotpot_f1": g("f1"),
            "citation_precision": g("cit_precision"), "citation_recall": g("cit_recall"),
            "evidence_recall": g("evidence_recall"), "abstention_on_unanswerable": g("abstained"),
            "false_abstention": g("false_abstention"), "conflict_detection": g("conflict_detected"),
            "invalid_citations_total": sum(r["invalid_citations"] for r in rows),
            "degraded": sum(1 for r in rows if r["status"] != "ok"),
            "latency_s_mean": g("latency"), "latency_s_p95": _p95([r["latency"] for r in rows]),
            "llm_calls_mean": g("llm_calls"), "tokens_mean": g("tokens"), "tool_calls_mean": g("tool_calls")}


def _p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def _f(x) -> str:
    return "—" if x is None or x != x else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_report(res: dict, path: Path) -> None:
    L = [f"# Evaluation results — {res['timestamp']}", "",
         f"Index: {res['index']['documents']} documents / {res['index']['chunks']} chunks · embedder "
         f"`{res['index']['embedder']}` · LLM `{res.get('llm_model')}` · {res['n_questions']} questions", "",
         "## A. Retrieval (single-shot, top-10, answerable questions only)", "",
         "| Config | Recall@5 | Recall@10 | All-gold@10 | MRR@10 | Latency (ms) |", "|---|---|---|---|---|---|"]
    for mode, r in res["retrieval"].items():
        s = r["summary"]
        L.append(f"| {mode} | {_f(s['recall@5'])} | {_f(s['recall@10'])} | {_f(s['all_gold@10'])} | "
                 f"{_f(s['mrr@10'])} | {s['latency_ms']:.0f} |")
    L += ["", "### Retrieval by question type (Recall@10 / All-gold@10)", "",
          "| Type | " + " | ".join(res["retrieval"]) + " |", "|---|" + "---|" * len(res["retrieval"])]
    types = sorted({t for r in res["retrieval"].values() for t in r["by_type"]})
    for t in types:
        cells = []
        for r in res["retrieval"].values():
            b = r["by_type"].get(t)
            cells.append(f"{_f(b['recall@10'])} / {_f(b['all_gold@10'])}" if b else "—")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    if res["e2e"]:
        keys = ["answer_correctness", "hotpot_em", "hotpot_f1", "citation_precision", "citation_recall",
                "evidence_recall", "abstention_on_unanswerable", "false_abstention", "conflict_detection",
                "invalid_citations_total", "degraded", "latency_s_mean", "latency_s_p95", "llm_calls_mean",
                "tokens_mean", "tool_calls_mean"]
        L += ["", "## B. End-to-end agent", "", "| Metric | " + " | ".join(res["e2e"]) + " |",
              "|---|" + "---|" * len(res["e2e"])]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(_f(v["summary"][k]) for v in res["e2e"].values()) + " |")
        for mode, v in res["e2e"].items():
            L += ["", f"### Answer correctness by type ({mode})", "", "| Type | n | Correct | Abstained | Conflict found |",
                  "|---|---|---|---|---|"]
            for t, b in v["by_type"].items():
                L.append(f"| {t} | {b['n']} | {_f(b['answer_correctness'])} | {_f(b['abstention_on_unanswerable'])} | "
                         f"{_f(b['conflict_detection'])} |")
            fails = [r for r in v["rows"] if r.get("correct") == 0.0 or r.get("abstained") == 0.0
                     or r.get("conflict_detected") == 0.0]
            L += ["", f"### Failures ({mode}) — {len(fails)}", ""]
            for r in fails:
                L.append(f"- **{r['id']}** ({r['type']}) Q: {r['question']}  \n  gold: `{r['gold']}` · got: "
                         f"`{(r['short_answer'] or r['answer'])[:160]}` · insufficient={r['insufficient']} · "
                         f"evidence_recall={_f(r.get('evidence_recall'))}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="hybrid,dense,bm25")
    ap.add_argument("--e2e", default="hybrid", help="comma list of modes, or 'none'")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between questions (free-tier RPM limits)")
    args = ap.parse_args()
    # Model answers contain non-ASCII (→, Georgian names); Windows consoles/redirects default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = get_settings()
    engine = Engine(s, make_embedder(s.embed_backend, s.embed_model, s.embed_batch))
    engine.rebuild_from_store()
    items = load_items(args.limit)
    res: dict = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_questions": len(items),
                 "index": engine.stats(), "settings": {k: v for k, v in vars(s).items() if "key" not in k and "token" not in k}}
    res["retrieval"] = retrieval_eval(engine, items, [m for m in args.retrieval.split(",") if m])
    res["e2e"] = {}
    if args.e2e != "none":
        try:
            llm = OpenAICompatLLM(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout, s.llm_max_retries)
        except LLMError as exc:
            sys.exit(f"end-to-end eval needs an LLM: {exc}")
        res["llm_model"] = s.llm_model
        agent = InvestigationAgent(engine, llm, max_steps=s.max_agent_steps)
        for mode in args.e2e.split(","):
            res["e2e"][mode] = e2e_eval(agent, items, mode, args.sleep)
            print(json.dumps(res["e2e"][mode]["summary"], indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_report(res, RESULTS / "report.md")
    print(f"\nwrote {RESULTS / (stamp + '.json')} and {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
