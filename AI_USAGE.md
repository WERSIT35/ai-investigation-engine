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

Most code was written with Claude Code; my part was choosing the direction, running it on the real setup and
deciding what to do when something broke. Concretely:

**Decisions I made**
- Dataset: HotpotQA validation paragraphs (66,563 documents after dedup) plus a small synthetic incident corpus with
  deliberate contradictions, so the conflict and abstention requirements can actually be tested.
- Stack: Qdrant Cloud free tier for vectors, FastEmbed `bge-small-en-v1.5` on CPU (no embedding API cost), an
  OpenAI-compatible LLM client so the provider is a config change.
- UI: kept the working tool layout and asked only for a restyle, not a redesign of the flow.
- Evaluation model: first tried Groq (`qwen/qwen3.8-27b`). It failed for reasons unrelated to my system (7k input
  tokens/min cap returns 413 on multi-step questions; the model invented tool names), so I stopped that run to save
  quota and switched the eval to Gemini 3.1 Flash-Lite, the same model the demo uses.
- Hosting: Hugging Face now requires PRO for Docker Spaces (got HTTP 402) and the app needs ~760 MB RAM, more than
  512 MB free tiers. I chose a Cloudflare quick tunnel from my machine with write endpoints locked by an admin token.
- Free-tier budget: I asked for every run to minimise LLM calls (retrieval eval needs none; only one end-to-end config).

**Problems found by running it for real (fixed with Claude Code, reviewed by me)**
- Gemini 3 rejected follow-up requests with HTTP 400 until each tool call's `thought_signature` was echoed back.
- `gemini-2.5-flash` returned 404 on my key; the default model became `gemini-3.1-flash-lite`.
- Concurrent uploads could corrupt the in-memory BM25 index → write/index locks; updates now upsert first and delete
  stale chunks after, so a failed update never loses the old version.
- A model passing `top_k="six"` crashed `/ask` → tool arguments are coerced and validated.
- Behind a proxy every visitor shared one rate-limit bucket → `TRUST_PROXY` reads the client from `X-Forwarded-For`.
- Groq 400 `tool_use_failed` degraded the whole answer → the agent now answers from the evidence already gathered.
- Eval scripts crashed on Windows cp1252 output (`→` in an answer) → UTF-8 stdout.

## How I verified AI-written code

- Read every module; can explain the RAG flow step by step (see README "Request flow").
- `pytest -q` (36 tests) covering parsers, chunking, BM25, RRF, dedup/CRUD, citation validation, LLM failure, API auth/limits.
- Ran the evaluation on the full index and inspected the failure list manually.
