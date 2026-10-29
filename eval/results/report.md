# Evaluation results — 2026-10-29T16:24:45

Index: 66563 documents / 78009 chunks · embedder `BAAI/bge-small-en-v1.5` · LLM `gemini-3.1-flash-lite` · 47 questions

## A. Retrieval (single-shot, top-10, answerable questions only)

| Config | Recall@5 | Recall@10 | All-gold@10 | MRR@10 | Latency (ms) |
|---|---|---|---|---|---|
| hybrid | 0.889 | 0.944 | 0.881 | 0.948 | 278 |
| dense | 0.861 | 0.893 | 0.833 | 0.940 | 262 |
| bm25 | 0.794 | 0.889 | 0.786 | 0.873 | 6 |

### Retrieval by question type (Recall@10 / All-gold@10)

| Type | hybrid | dense | bm25 |
|---|---|---|---|
| comparison | 0.967 / 0.933 | 0.933 / 0.867 | 0.900 / 0.800 |
| conflict | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| multi_hop | 0.904 / 0.789 | 0.816 / 0.737 | 0.833 / 0.684 |
| simple | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

## B. End-to-end agent

| Metric | hybrid |
|---|---|
| answer_correctness | 0.810 |
| hotpot_em | 0.400 |
| hotpot_f1 | 0.563 |
| citation_precision | 0.782 |
| citation_recall | 0.865 |
| evidence_recall | 0.948 |
| abstention_on_unanswerable | 1.000 |
| false_abstention | 0.095 |
| conflict_detection | 1.000 |
| invalid_citations_total | 0 |
| degraded | 0 |
| latency_s_mean | 10.740 |
| latency_s_p95 | 38.200 |
| llm_calls_mean | 1.660 |
| tokens_mean | 4865.872 |
| tool_calls_mean | 0.851 |

### Answer correctness by type (hybrid)

| Type | n | Correct | Abstained | Conflict found |
|---|---|---|---|---|
| comparison | 15 | 0.733 | — | — |
| conflict | 3 | 1.000 | — | 1.000 |
| multi_hop | 19 | 0.789 | — | — |
| simple | 5 | 1.000 | — | — |
| unanswerable | 5 | — | 1.000 | — |

### Failures (hybrid) — 8

- **hp-5a74547755429979e2882900** (multi_hop) Q: the head football coach at the University of Houston from 2007 to 2011, is the current team coach of which football team ?  
  gold: `Texas A&M Aggies football` · got: `Texas A&M University` · insufficient=False · evidence_recall=0.500
- **hp-5ac275f155429921a00aaf82** (multi_hop) Q: what did the second largest city by population in vietnam also serve as in the first indochina war?   
  gold: `Viet Minh's base` · got: `capital of French Indochina` · insufficient=False · evidence_recall=1.000
- **hp-5a84b4005542991dd0999d99** (multi_hop) Q: When did the Disney sitcom in which Genevieve Knight "G" Hannelius starred as Avery Jennings end?  
  gold: `October 16, 2015` · got: `September 25, 2015` · insufficient=False · evidence_recall=0.500
- **hp-5a886886554299206df2b258** (multi_hop) Q: What is the international football competition for women that Liu Ailing played in the 1991,1995, 1999 editions?  
  gold: `international football competition` · got: `FIFA Women's World Cup` · insufficient=False · evidence_recall=1.000
- **hp-5a8c30f6554299240d9c20d1** (comparison) Q: Were The Bomb and Rolling Papers released in the same year?  
  gold: `yes` · got: `The evidence does not contain information about an album titled "The Bomb." Consequently, it is impossible to determine if such an album was released in the sam` · insufficient=True · evidence_recall=1.000
- **hp-5ae1e9635542997f29b3c19e** (comparison) Q: Which airport is located further north in the United States, San Angelo Regional Airport or Grand Junction Regional Airport?  
  gold: `Grand Junction Regional Airport` · got: `The provided documents do not contain the specific latitude coordinates for San Angelo Regional Airport or Grand Junction Regional Airport, nor do they provide ` · insufficient=True · evidence_recall=1.000
- **hp-5a875fee5542996e4f3087ab** (comparison) Q: Which has more species, Lysiloma or Hydrocotyle umbellata?  
  gold: `Lysiloma` · got: `The provided documents do not contain information regarding the total number of species in the genera *Lysiloma* or *Hydrocotyle*. While *Hydrocotyle umbellata*` · insufficient=True · evidence_recall=1.000
- **hp-5a7a2be25542996a35c17118** (comparison) Q: Which band has had more members, The Operation M.D. or Del Amitri?  
  gold: `Del Amitri` · got: `The provided evidence does not contain sufficient information to determine the total number of members for either The Operation M.D. or Del Amitri. While Del Am` · insufficient=True · evidence_recall=1.000
