# deps.md — dependency requests (coordinator-owned)

Per PLAN.md §2.3, agents declare `deps_needed` here and AGENT A installs them.
All packages are already declared in `pyproject.toml` (AGENT A) and installed.

| Agent | Requested deps | Status |
|-------|----------------|--------|
| B | markdown parsing | Not needed — implemented with stdlib (heading/fence/table line parsing) |
| C | psycopg[binary], pgvector, sentence-transformers | Already in pyproject.toml |
| D | groq | Already in pyproject.toml |
| E | python-dotenv, pytest | Already in pyproject.toml (+dev extra) |

### Phase 2 (agentic retrieval)

| Agent | Requested deps | Status |
|-------|----------------|--------|
| F, G | `langgraph` (StateGraph for the conditional agentic loop) | Declared in `pyproject.toml` by AGENT A; `uv sync` installs it |

**Phase 1 note:** no pyproject/uv.lock changes were required for Phase 1.