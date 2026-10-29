# AI Usage Report

## Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic, incl. Claude Code) | Requirement breakdown, time plan, initial code scaffold for all modules, tests, eval scripts, README drafts |
| Gemini 3.1 Flash-Lite (runtime) | The LLM inside the product (answer generation + tool calling) and the end-to-end eval |
| Groq `qwen/qwen3.8-27b` (tried) | First end-to-end eval attempt; dropped because the free tier caps input at 7k tokens/min and the model invented tool names |

## What AI helped with

- Turning the task brief into a requirement checklist and a build order for a 1-day deadline.
- First drafts of: loaders, two chunking strategies, BM25, RRF fusion, Qdrant wrapper, the agent loop, FastAPI routes,
  the HTML UI, the eval runner and metrics, and the pytest suite (36 tests now).
- The synthetic incident corpus with deliberate contradictions.

## What I changed / decided myself

TODO(oto) — be specific and honest; the interview will check this. For example:
- which parts you rewrote after reviewing them and why
- bugs you found when running on the real corpus / Qdrant Cloud
- parameters you tuned from eval results (chunk size, `RRF_K`, `MAX_CHUNKS_PER_DOC`, `TOP_K`)
- deployment issues you solved

## How I verified AI-written code

- Read every module; can explain the RAG flow step by step (see README "Request flow").
- `pytest -q` (36 tests) covering parsers, chunking, BM25, RRF, dedup/CRUD, citation validation, LLM failure, API auth/limits.
- Ran the evaluation on the full index and inspected the failure list manually.
