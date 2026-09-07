# DocPilot

An evidence-driven agentic RAG system for technical documentation.

DocPilot ingests Markdown/MDX documentation (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. Rather than naive retrieve-and-answer, it evaluates whether its evidence is sufficient, retries searches when it isn't, and says **"I don't know"** rather than hallucinating. Every retrieval and tool decision is loggable and inspectable.

> **Status:** Phases 1–2 complete (Classic RAG + Agentic Retrieval). Phase 3 (MCP/GitHub) in progress. See [PLAN.md](PLAN.md) for the build plan and [SPEC.md](SPEC.md) for the authoritative specification.

## Current scope (Phases 1–2)

- Ingestion: FastAPI docs corpus → Markdown-aware semantic chunking → BGE-small embeddings → pgvector
- Retrieval: exact cosine search (top-k), language-filtered (English default; `--lang <tag>`, `any` disables)
- Agentic retrieval: a heuristic gate routes multi-hop questions through a LangGraph loop — retrieve → sufficiency judge → reformulate/retry (max 2 retries → honest refusal). Simple questions keep the classic fast path
- Generation: Groq `openai/gpt-oss-20b`, temperature 0, retry/backoff
- Output: cited answers with inline `[1]` source markers + footer; per-step agent trace in `--debug` / `--json`
- CLI: `python -m docpilot ingest [--debug]` · `python -m docpilot ask "..." [--debug] [--json] [--strategy auto|direct|agentic] [--lang <tag>]`

## Setup

```bash
uv sync            # install dependencies into .venv
```

Create a `.env` file from the template: `cp .env.example .env`, then fill in the required values (`GROQ_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — and optionally `GITHUB_PAT` to enable the Phase 3 GitHub tool). Keys are listed in **SPEC.md §3.16**.

## Usage

```bash
python -m docpilot ingest            # ingest the corpus into pgvector
python -m docpilot ask "How do I install FastAPI?"
python -m docpilot ask --json "..."  # structured output
python -m docpilot ask --debug "..." # inspect retrieval + prompt details
python -m docpilot ask --strategy agentic "Combine path, query, and body parameters in one endpoint..." # force the agentic loop
python -m docpilot ask --strategy direct "How do I install FastAPI?"                                 # force the classic fast path
python -m docpilot ask --lang ja "..."                                                              # language-filtered retrieval
```

## Phases

| Phase | Status |
|-------|--------|
| 1. Classic RAG | ✅ Complete |
| 2. Agentic retrieval | ✅ Complete |
| 3. MCP / GitHub tooling | 🚧 In progress (current) |
| 4. Evaluation | Planned |
| 5. API + UI | Planned |
| 6. Code generation/validation | Planned |
| Framework extraction | Post-Phase 6 |

See [SPEC.md](SPEC.md) for full details.