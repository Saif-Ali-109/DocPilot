---
title: DocPilot — README
role: human-facing overview — keep claims honest, no contracts
authority: none — informational
load: not needed for agent sessions
last_updated: 2026-09-11
---

# DocPilot

An evidence-driven agentic RAG system for technical documentation.

DocPilot ingests Markdown/MDX documentation (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. Rather than naive retrieve-and-answer, it evaluates whether its evidence is sufficient, retries searches when it isn't, and says **"I don't know"** rather than hallucinating. Every retrieval and tool decision is loggable and inspectable.

> **Status:** Phases 1–6 complete (Classic RAG → Agentic Retrieval → GitHub tooling → Evaluation → API/UI → Code generation/validation), plus the pre-Phase-6 hardening batch (Plan §H): cross-encoder reranking, hybrid vector+FTS retrieval, gate-knob wiring, judge score-floor backstop, and a benchmark expansion to 30 questions. Phase 6 (code) is an **explicit opt-in** route — generate → validate against the retrieved docs → return only validated code, otherwise refuse with sources; it never changes the plain ask path. Levers ship default-OFF; the gate benchmark decides whether to flip them on. See [PLAN.md](PLAN.md) for the build plan and [SPEC.md](SPEC.md) for the authoritative specification.

## Current scope

- **Ingestion:** FastAPI docs corpus → Markdown-aware semantic chunking → BGE-small embeddings → pgvector
- **Retrieval:** exact cosine search (top-k), language-filtered (English default; `--lang <tag>`, `any` disables)
- **Agentic retrieval:** a heuristic gate routes multi-hop questions through a LangGraph loop — retrieve → sufficiency judge → reformulate/retry (max 2 retries → honest refusal). Simple questions keep the classic fast path. Optional judge-skip (disabled by default) answers directly when the top retrieval score clears `AGENT_JUDGE_SKIP_MIN_SCORE`
- **GitHub tool (Phase 3):** judge-gated — the loop reaches GitHub (live issues, repo commits) via a plain REST `Tool` only when static docs are demonstrably insufficient; each call is traced and cited (`github:#issue` / `@commit`)
- **Generation:** Groq, temperature 0, retry/backoff; token usage recorded per answer
- **Output:** cited answers with inline `[1]` source markers + footer (sources tagged by kind — tutorial/advanced/reference/… — to steer the LLM to the most specific file); per-step agent trace in `--debug` / `--json`
- **API + UI (Phase 5, complete):** async FastAPI backend with SSE token streaming; `POST /api/v1/chat` returns a `gate → search → token… → answer → done` event stream; SQLite session/chat history (`/api/v1/sessions`); Chainlit UI renders the live trace, a retrieval debug panel (chunks + scores) and cited sources
- **Pre-Phase-6 hardening (Plan §H):** optional retrieval levers (all default OFF — enabled when benchmark justifies)
  - **Cross-encoder reranking** (`RERANK_ENABLED=1`): fetches a wider candidate window (`RERANK_CANDIDATES=20`), re-scores with `BAAI/bge-reranker-base` via sentence-transformers `CrossEncoder`, keeps the top-k by reranked relevance. Gate verdict: implemented and tested but **not** enabled — it loses to plain cosine at retrieval (recall 0.900→0.850) and costs ~85 s/predict on CPU (PLAN §6.5.2). Kept as the `Reranker` interface implementation for corpus-specific A/B.
  - **Hybrid retrieval** (`HYBRID_ENABLED=1`): fuses the dense vector path with a Postgres full-text search (`to_tsquery` OR-semantics + `ts_rank`, GIN index on `chunks.content`) via weighted Reciprocal Rank Fusion — rescues exact API identifiers and error codes the dense index under-weights. Gate verdict (PLAN §6.5.2): hybrid **alone** at the 2:1 vector:lexical weights is a strict retrieval winner (recall parity 0.900, MRR 0.717→0.783, near-zero latency); the cross-encoder reranker **hurts** retrieval on this corpus (0.850 recall in every variant, ~85 s/predict on CPU) and stays off. `HYBRID_ENABLED` ships `0` until the answer-level gate confirms the retrieval-level gain.
  - **Judge score-floor backstop** (`AGENT_JUDGE_SCORE_FLOOR` > 0): forces `insufficient` when top retrieval score falls below the floor, preserving `needs_tool`/`tool_request` for the GitHub path
- **Code generation/validation (Phase 6):** an opt-in code route — `POST /api/v1/code`, a Chainlit *Generate validated code* toggle, or per-call `explicit_code` — runs the T4 loop: retrieve → generate (`ask_code`) → validate every emitted block structurally against the retrieved docs (parse → imports grounded → symbols grounded; no LLM judge, `SKIP` ≠ pass) → on failure, reformulate with the failure reasons (up to `CODE_VALIDATE_MAX_TURNS`) → return the code + cited sources only when fully validated, otherwise refuse with **"I couldn't validate the generated code against the retrieved documentation, so I'm not returning code."** plus the retrieved sources. Unvalidated code is never returned; non-code asks are byte-identical to the standard `ask()` path (`CODE_ROUTE_ENABLED` ships `0`). SSE emission: gate step → per-attempt `search` + a `code` validation-verdict event (checks with PASS/FAIL/SKIP + reasons) → `answer` → `done`.

## Setup

```bash
uv sync --extra dev   # install dependencies (incl. test extra) into .venv
```

Create a `.env` file from the template: `cp .env.example .env`, then fill in the required values (`GROQ_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — and optionally `GITHUB_PAT` to enable the Phase 3 GitHub tool). Keys are listed in **SPEC.md §3.16**. In this project's dev environment the keys live in the user's shell profile; run live commands with `bash -ic`.

## Usage

### CLI (Phases 1–4)

```bash
python -m docpilot ingest            # ingest the corpus into pgvector
python -m docpilot ask "How do I install FastAPI?"
python -m docpilot ask --json "..."  # structured output
python -m docpilot ask --debug "..." # inspect retrieval + prompt details
python -m docpilot ask --strategy agentic "Combine path, query, and body parameters in one endpoint..." # force the agentic loop
python -m docpilot ask --strategy direct "How do I install FastAPI?"                                 # force the classic fast path
python -m docpilot ask --lang ja "..."                                                              # language-filtered retrieval
```

### API (Phase 5)

```bash
uvicorn docpilot.api.app:app --port 8000
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"question": "How do I install FastAPI?", "session_id": "<id>"}'
```

SSE events: `step` (gate/trace), `search` (retrieved chunks + scores), `token` (streamed deltas), `answer` (citation-formatted display + sources), `done` (trace, latency, usage). Session CRUD at `/api/v1/sessions` (SQLite, `data/docpilot.sqlite3`, git-ignored).

### UI (Phase 5)

```bash
chainlit run src/docpilot/ui/chainlit_app.py   # from the repo root
```

Broken down development is documented in PLAN.md §6.

## Phases

| Phase | Status |
|-------|--------|
| 1. Classic RAG | ✅ Complete |
| 2. Agentic retrieval | ✅ Complete |
| 3. GitHub tooling (plain REST, no MCP) | ✅ Complete (benchmarked `phase-3`) |
| 4. Evaluation | ✅ Complete (benchmarked `phase-4`; 30-question suite post-H-expansion, classic-vs-agentic comparison) |
| 5. API + UI | ✅ Complete (backend + Chainlit UI; hardening after-run recorded, gate stamp `20260910_201739`, tagged `phase-5`) |
| 5.5 Pre-Phase-6 hardening (Plan §H) | ✅ Implemented (reranker, hybrid-FTS, gate wiring, judge score-floor, eval expansion to 30 rows). Levers default OFF: reranker loses retrieval gate; hybrid 2:1 wins retrieval (MRR +8.3%, recall parity) but `HYBRID_ENABLED` stays 0 until the answer-level classic gate (pending). Expanded baseline complete (PLAN §6.5.3): agentic 0.85 vs classic 0.817 at 2.2× latency — agentic justified only for hard questions, never as default path. |
| 6. Code generation/validation | ✅ Complete, tagged `phase-6` (2026-09-12). Opt-in code route (`POST /api/v1/code`, Chainlit *Generate validated code* toggle, per-call `explicit_code`): retrieve → generate → validate structurally against retrieved docs (parse → imports grounded → symbols grounded; no LLM judge, `SKIP` ≠ pass) → return only fully-validated code with cited sources, else refuse with sources. Code eval `20260912_codefix` (6 rows, `src/docpilot/eval/reports/`): validation-pass **0.75** (3/4 validation-attempted; 1 clean validation refusal), compiles 1.0, imports/symbols grounded 0.75, citation-gold **0.333** (1/3 marker rows; c02/c04 marker→source misses mirror the main benchmark's citation weakness), refusal accuracy 0.833 (one false coverage refusal c03), avg 1.7 validation turns, ~11.4 s/row. Honest boundary: code is a dedicated opt-in route — it never changes the plain ask path. |
| Framework extraction | Post-Phase 6 |

See [SPEC.md](SPEC.md) for full details.