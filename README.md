# DocPilot

An evidence-driven agentic RAG system for technical documentation.

DocPilot ingests Markdown/MDX documentation (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. Rather than naive retrieve-and-answer, it evaluates whether its evidence is sufficient, retries searches when it isn't, and says **"I don't know"** rather than hallucinating. Every retrieval and tool decision is loggable and inspectable.

> **Status:** Phase 1 (Classic RAG) in progress. See [PLAN.md](PLAN.md) for the build plan and [SPEC.md](SPEC.md) for the authoritative specification.

## Current scope (Phase 1)

- Ingestion: FastAPI docs corpus → Markdown-aware semantic chunking → BGE-small embeddings → pgvector
- Retrieval: exact cosine search (top-k)
- Generation: Groq `openai/gpt-oss-20b`, temperature 0, retry/backoff
- Output: cited answers with inline `[1]` source markers + footer
- CLI: `python -m docpilot ingest [--debug]` · `python -m docpilot ask "..." [--debug] [--json]`

## Setup

```bash
uv sync            # install dependencies into .venv
cp .env.example    # (no — .env keys are documented in SPEC.md §3.16)
```

Create a `.env` file with the keys listed in **SPEC.md §3.16** (`GROQ_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, etc.).

## Usage

```bash
python -m docpilot ingest            # ingest the corpus into pgvector
python -m docpilot ask "How do I install FastAPI?"
python -m docpilot ask --json "..."  # structured output
python -m docpilot ask --debug "..." # inspect retrieval + prompt details
```

## Phases

| Phase | Status |
|-------|--------|
| 1. Classic RAG | **In progress** |
| 2. Agentic retrieval | Planned |
| 3. MCP / GitHub tooling | Planned |
| 4. Evaluation | Planned |
| 5. API + UI | Planned |
| 6. Code generation/validation | Planned |
| Framework extraction | Post-Phase 6 |

See [SPEC.md](SPEC.md) for full details.