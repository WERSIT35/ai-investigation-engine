# Evaluation Report

> Numbers below come from `eval/results/report.md` and `eval/results/chunking.md`. Every `TODO` cell must be filled
> from a real run — do not estimate.

## 1. Methodology

**Question set** (`eval/eval_set.jsonl`, built deterministically with seed 42 by `eval/build_eval_set.py`): 47 questions.

| Type | n | Source | What it tests |
|---|---|---|---|
| simple | 5 | synthetic | single-document fact lookup (incl. the PDF) |
| multi_hop | 4 + 15 | synthetic + HotpotQA *bridge* | chaining facts across documents |
| comparison | 15 | HotpotQA *comparison* | comparing two entities from two documents |
| unanswerable | 5 | synthetic | answer is not in the corpus → must abstain |
| conflict | 3 | synthetic | documents disagree → must report both versions |

Gold labels: HotpotQA supporting-fact titles map 1:1 to documents (`hp::<title>`); synthetic questions have hand-labelled gold documents and required answer strings.

**Configurations compared**

- Retrieval: `hybrid` (BM25 + dense, RRF, diversity cap) vs `dense` only vs `bm25` only.
- End-to-end: agent with `hybrid` vs agent with `bm25` retrieval.
- Chunking: `fixed` vs `recursive`, at 300 and 800 characters.

**Metrics**

| Metric | Definition |
|---|---|
| Recall@K | share of gold documents among the top-K retrieved documents |
| All-gold@10 | 1 if *every* gold document is in the top 10 (what a multi-hop question actually needs) |
| MRR@10 | 1 / rank of the first gold document |
| Answer correctness | answer contains all required strings (synthetic) or the gold answer / token-F1 ≥ 0.6 vs the short answer (HotpotQA); abstentions count as wrong |
| HotpotQA EM / F1 | standard SQuAD-normalised exact match and token F1 on the short answer |
| Citation precision / recall | cited documents vs gold documents |
| Evidence recall | gold documents seen anywhere during the agent's investigation (seed + tool calls) |
| Abstention | unanswerable questions answered with `insufficient = true` |
| False abstention | answerable questions where the system refused |
| Conflict detection | conflict questions with at least one structured conflict reported |

## 2. Results

Index: TODO documents / TODO chunks · `BAAI/bge-small-en-v1.5` · LLM: TODO

### 2.1 Retrieval

| Config | Recall@5 | Recall@10 | All-gold@10 | MRR@10 | Latency |
|---|---|---|---|---|---|
| hybrid | TODO | TODO | TODO | TODO | TODO |
| dense | TODO | TODO | TODO | TODO | TODO |
| bm25 | TODO | TODO | TODO | TODO | TODO |

### 2.2 End-to-end

| Metric | hybrid | bm25 |
|---|---|---|
| Answer correctness | TODO | TODO |
| HotpotQA EM / F1 | TODO | TODO |
| Citation precision / recall | TODO | TODO |
| Evidence recall (agent) | TODO | TODO |
| Abstention on unanswerable | TODO | TODO |
| False abstention | TODO | TODO |
| Conflict detection | TODO | TODO |
| Mean latency / p95 | TODO | TODO |
| Mean LLM calls / tokens per question | TODO | TODO |

### 2.3 Chunking

Paste the table from `eval/results/chunking.md`.

## 3. Analysis

Fill from the *Failures* list in `eval/results/report.md`. Questions to answer:

- **Where does hybrid beat dense/BM25?** Expect BM25 to win on identifiers (`CHG-3301`, `INC-1042`) and rare names,
  dense on paraphrased questions; check the per-type table.
- **Single-shot vs agent:** compare All-gold@10 (single search) with agent evidence recall. The gap is what the
  second-hop tool calls buy on bridge questions.
- **Failure categories:** (a) retrieval miss — gold doc never seen; (b) reasoning miss — evidence seen, wrong answer;
  (c) format/grading — correct answer phrased differently from gold (e.g. "yes" vs a full sentence);
  (d) over-abstention.
- **Improvements** (ranked by expected impact / cost): cross-encoder re-ranking of the top 30; query decomposition
  before the first search; LLM-as-judge grading for long answers; larger embedding model for HotpotQA entity questions.
