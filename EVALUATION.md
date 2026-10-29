# Evaluation Report

> Every number below is copied from a real run: `eval/results/report.md` (+ the timestamped JSON next to it) and
> `eval/results/chunking.md`. Re-run with the commands in the README to reproduce.

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
- End-to-end: the agent with `hybrid` retrieval (one LLM config; see 2.2 for why a second end-to-end config was not run).
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

All 47 questions, agent with hybrid retrieval, `gemini-3.1-flash-lite`, free tier, run 2026-10-29
(`eval/results/20261029-163949.json`, `eval/results/report.md`).

| Metric | Value |
|---|---|
| Answer correctness (42 answerable) | **0.810** |
| HotpotQA EM / F1 (34 questions) | 0.400 / 0.563 |
| Citation precision / recall | 0.782 / 0.865 |
| Evidence recall (whole investigation) | 0.948 |
| Abstention on unanswerable (5) | **1.000** |
| False abstention | 0.095 (4 of 42) |
| Conflict detection (3) | **1.000** |
| Invalid (hallucinated) citations | 0 |
| Degraded answers (LLM failures) | 0 |
| Mean latency / p95 | 10.7 s / 38.2 s |
| Mean LLM calls / tokens / tool calls per question | 1.66 / 4,866 / 0.85 |

| Type | n | Correct |
|---|---|---|
| simple | 5 | 1.000 |
| multi_hop | 19 | 0.789 |
| comparison | 15 | 0.733 |
| conflict | 3 | 1.000 (both versions reported every time) |
| unanswerable | 5 | 1.000 abstained |

Only one end-to-end configuration was run, to stay inside free LLM quotas: the three retrieval configurations are
compared in 2.1 without any LLM calls. A second LLM was tried: Groq `qwen/qwen3.8-27b` (free tier) could not complete the
run — the free tier caps input at 7,000 tokens/minute, so multi-step requests (~7.6k tokens) were refused with HTTP 413,
and the model twice called tools that do not exist (`search_DOCUMENTS`, `search-treated`). That run led to a fix: a
rejected tool call now ends tool use and the agent answers from the evidence already gathered instead of failing.
Latency p95 includes Gemini timeouts that were retried.

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

**Hybrid vs dense vs BM25.** Hybrid is best on every quality metric (Recall@10 0.944 vs 0.893 / 0.889, All-gold@10
0.881 vs 0.833 / 0.786). The gain is concentrated where it matters: multi-hop All-gold@10 0.789 vs 0.737 (dense) and
0.684 (BM25), comparison 0.933 vs 0.867 / 0.800. Simple and conflict questions are already at 1.0 for all three. The
two retrievers fail on different questions (BM25 misses paraphrases, dense misses rare names), which is exactly the
case where rank fusion helps. Cost: dense and hybrid both pay for the query embedding (~260–280 ms on CPU); BM25 alone
is 6 ms.

**Single search vs agent.** One hybrid search finds every gold document for 88.1 % of questions (All-gold@10). The
agent's investigation (seed search + its own tool calls) reaches 94.8 % evidence recall. Those extra hops are what
answer bridge questions whose second entity is only named inside the first document.

**Failures (8 of 42 answerable).** Every unanswerable question was refused and every conflict was reported, so the
errors are all on HotpotQA:

| Category | n | Questions | What happened |
|---|---|---|---|
| Over-abstention | 4 | 4 comparison questions (The Bomb / Rolling Papers, two airports, Lysiloma, two bands) | Both gold documents were retrieved (evidence recall 1.0), but the answer needs a small inference the documents do not state directly — e.g. which airport is further north from the cities they are in — and the prompt's "do not guess" rule made the model refuse. |
| Retrieval miss on the second hop | 2 | Houston coach → *Texas A&M Aggies football*; Avery Jennings sitcom end date | Evidence recall 0.5: the second document was never found, so the model answered from the first one (the university instead of the team; a wrong date). |
| Reasoning / ambiguous gold | 1 | second-largest city in Vietnam in the First Indochina War | Evidence complete; the model picked a different role of the city than the gold label. |
| Grading artifact | 1 | Liu Ailing | Answered "FIFA Women's World Cup" (correct and more specific); the HotpotQA gold label is the generic "international football competition". |

**What I would change next (by expected gain per cost).**
1. Let the model make explicit, labelled inferences on comparison questions ("X is in Colorado, Y in Texas, so…")
   instead of refusing — targets the 4 over-abstentions (≈ +10 points of correctness) for a prompt change only.
2. Query decomposition / a forced second search on the bridge entity before answering — targets the 2 retrieval misses.
3. Cross-encoder re-ranking of the top 30 fused results to push the second-hop document into the top 10.
4. LLM-as-judge grading next to the string match, so answers like the Liu Ailing one are not counted as wrong.
5. Run the same 47 questions on a second LLM with a higher token allowance to compare models.
