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

Index: 66,563 documents / 78,009 chunks (Qdrant Cloud) · `BAAI/bge-small-en-v1.5` · LLM: `gemini-3.1-flash-lite` · run 2026-10-29

### 2.1 Retrieval

| Config | Recall@5 | Recall@10 | All-gold@10 | MRR@10 | Latency |
|---|---|---|---|---|---|
| **hybrid** | **0.889** | **0.944** | **0.881** | **0.948** | 278 ms |
| dense | 0.861 | 0.893 | 0.833 | 0.940 | 262 ms |
| bm25 | 0.794 | 0.889 | 0.786 | 0.873 | 6 ms |

42 answerable questions. Latency is the full search call (embedding the query on CPU dominates; BM25 alone is in-memory).

Recall@10 / All-gold@10 by question type:

| Type | n | hybrid | dense | bm25 |
|---|---|---|---|---|
| simple | 5 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| multi_hop | 19 | **0.904 / 0.789** | 0.816 / 0.737 | 0.833 / 0.684 |
| comparison | 15 | **0.967 / 0.933** | 0.933 / 0.867 | 0.900 / 0.800 |
| conflict | 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

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

Corpus: the 11 multi-paragraph synthetic documents (HotpotQA paragraphs are single short chunks, so chunking does not
change them) · 12 answerable questions · hybrid retrieval, top-3 chunks.

| Strategy | Size / overlap (chars) | Chunks | Mean chunk len | Answer-hit@3 | Doc recall@3 |
|---|---|---|---|---|---|
| fixed | 300 / 50 | 29 | 250 | **1.000** | 0.861 |
| recursive | 300 / 50 | 31 | 205 | 0.833 | 0.861 |
| fixed | 800 / 120 | 13 | 504 | 0.917 | 0.861 |
| recursive | 800 / 120 | 13 | 484 | 0.917 | 0.861 |

Doc recall is identical across all four, so chunking only changes whether the *exact answer* lands in the top-3
passages. Fixed/300 wins here, but 12 questions is a small sample (one question = 0.083). The default stays
recursive/800: it keeps paragraphs and table rows whole for the LLM, and on this set it ties fixed/800.

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
