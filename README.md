# DocPilot

An evidence-driven agentic RAG system for technical documentation.

DocPilot ingests Markdown/MDX documentation (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. Rather than naive retrieve-and-answer, it evaluates whether its evidence is sufficient, retries searches when it isn't, and says **"I don't know"** rather than hallucinating. Every retrieval and tool decision is loggable and inspectable.

> **Status:** Phases 1–4 complete (Classic RAG → Agentic Retrieval → GitHub tooling → Evaluation, benchmarked `phase-4`). Phase 5 (API + UI) **in progress** — async FastAPI backend, SSE streaming, citations + retrieved-context debug panel, SQLite session history, Chainlit UI. See [PLAN.md](PLAN.md) for the build plan and [SPEC.md](SPEC.md) for the authoritative specification.

## Current scope

- **Ingestion:** FastAPI docs corpus → Markdown-aware semantic chunking → BGE-small embeddings → pgvector
- **Retrieval:** exact cosine search (top-k), language-filtered (English default; `--lang <tag>`, `any` disables)
- **Agentic retrieval:** a heuristic gate routes multi-hop questions through a LangGraph loop — retrieve → sufficiency judge → reformulate/retry (max 2 retries → honest refusal). Simple questions keep the classic fast path. Optional judge-skip (disabled by default) answers directly when the top retrieval score clears `AGENT_JUDGE_SKIP_MIN_SCORE`
- **GitHub tool (Phase 3):** judge-gated — the loop reaches GitHub (live issues, repo commits) via a plain REST `Tool` only when static docs are demonstrably insufficient; each call is traced and cited (`github:#issue` / `@commit`)
- **Generation:** Groq, temperature 0, retry/backoff; token usage recorded per answer
- **Output:** cited answers with inline `[1]` source markers + footer (sources tagged by kind — tutorial/advanced/reference/… — to steer the LLM to the most specific file); per-step agent trace in `--debug` / `--json`
- **API + UI (Phase 5, in progress):** async FastAPI backend with SSE token streaming; `POST /api/v1/chat` returns a `gate → search → token… → answer → done` event stream; SQLite session/chat history (`/api/v1/sessions`); Chainlit UI renders the live trace, a retrieval debug panel (chunks + scores) and cited sources

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
| 4. Evaluation | ✅ Complete (benchmarked `phase-4`; 15-question suite, classic-vs-agentic comparison) |
| 5. API + UI | 🚧 In progress (backend + Chainlit UI built; milestone gate = §6.1 before/after benchmark) |
| 6. Code generation/validation | Planned (not implemented — do not treat as available) |
| Framework extraction | Post-Phase 6 |

See [SPEC.md](SPEC.md) for full details.