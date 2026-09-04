# deps.md — dependency requests (coordinator-owned)

Per PLAN.md §2.3, agents declare `deps_needed` here and AGENT A installs them.
All packages are already declared in `pyproject.toml` (AGENT A) and installed.

| Agent | Requested deps | Status |
|-------|----------------|--------|
| B | markdown parsing | Not needed — implemented with stdlib (heading/fence/table line parsing) |
| C | psycopg[binary], pgvector, sentence-transformers | Already in pyproject.toml |
| D | groq | Already in pyproject.toml |
| E | python-dotenv, pytest | Already in pyproject.toml (+dev extra) |

**No changes to pyproject.toml / uv.lock were required.**