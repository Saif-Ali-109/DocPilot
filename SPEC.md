# DocPilot — Specification

**Version:** 0.1.0 (Phase 1)
**Status:** Approved
**Last updated:** 2026-09-05 (§7 amended: Chainlit replaces Streamlit — see rationale in §7)

---

## 1. Project Overview

**DocPilot** is an evidence-driven agentic RAG system for technical documentation. It ingests Markdown/MDX docs (code blocks, nested headings, cross-references) and answers questions using retrieved evidence.

Key behaviours:
- Judges whether retrieved evidence is sufficient to answer a question.
- Retries and reformulates searches when evidence is insufficient (Phase 2+).
- Calls out to GitHub via MCP when static docs can't answer (Phase 3+).
- Returns "I don't know" rather than hallucinating.
- Every retrieval and tool decision is loggable and inspectable.

**Purpose:** DocPilot is the learning/validation project for a future reusable RAG framework. DocPilot comes first; the framework is extracted from it later.

---

## 2. Phased Build Order

```
Phase 1: Classic RAG → Phase 2: Agentic retrieval → Phase 3: MCP tools →
Phase 4: Evaluation → Phase 5: API/UI → Phase 6: Code gen/validation →
Extract reusable framework
```

**Hard rules:**
1. Never implement a later phase's functionality to unblock an earlier phase.
2. Never skip Phase 4 to reach Phase 5 faster.
3. Do not add Phase 6 code generation into the core loop unless explicitly told the project is at that phase.
4. Do not build the reusable framework before DocPilot itself works end-to-end and has been evaluated.

---

## 3. Phase 1 — Classic RAG (Detailed)

### 3.1 Pipeline

```
Docs → Parsing → Code-aware chunking → Local embeddings → pgvector → Retrieval → Groq LLM → Cited answer
```

### 3.2 Corpus

| Property | Value |
|----------|-------|
| Source | FastAPI documentation (official repo) |
| Format | `.md` / `.mdx` source files |
| Location | `docs/` directory |
| Versioning | Pin to a specific commit; record commit hash in `docs/CORPUS.md` |
| Preprocessing | None at ingest time — raw files stored as-is |
| Re-ingestion | Idempotent — see §3.17 and the note in §3.4 |

### 3.3 Technology Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Python | `>=3.13, <3.14` | Pin in `pyproject.toml` |
| Environment | `uv` venv | Managed via `pyproject.toml` + `uv.lock`; no `requirements.txt` |
| Config | `.env` file | Secrets never hardcoded or committed |
| Embeddings | `BAAI/bge-small-en-v1.5` | 384 dimensions, local via `sentence-transformers` |
| Vector DB | pgvector (PostgreSQL extension) | Running locally, `localhost:5432` |
| LLM | Groq `openai/gpt-oss-20b` | *Was* `llama-3.1-8b-instant`, retired from Groq (returns 404 for all keys, Sep 2026). Replacement chosen after live probing of available models for citation discipline (answered with correct `[1]` markers) and instruction-following. Model ID configurable via `.env` (`GROQ_MODEL`), never hardcoded in business logic |
| Async | None | Synchronous pipeline; async introduced at Phase 5 API boundary |

### 3.4 Database

**Connection:**
- Host: `localhost`
- Port: `5432`
- Database: `docpilot`
- Credentials: via `.env` (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`)

**Schema — `chunks` table:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- required for gen_random_uuid()

CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content       TEXT NOT NULL,
    heading_path  TEXT,                -- e.g. "/Getting Started/Installation/"
    source_file   TEXT NOT NULL,       -- relative path to .md/.mdx file
    chunk_index   INT NOT NULL,        -- order within source file
    metadata      JSONB DEFAULT '{}',  -- code_block, table, heading_level, etc.
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Note: NO ANN index in Phase 1. Use exact (brute-force) cosine search.
-- FastAPI docs chunk to roughly 1,000–3,000 rows — too small for ANN to pay
-- off, and a wrong-sized IVFFlat index (e.g. lists=100) actively hurts recall
-- at this scale. Add an HNSW/IVFFlat index later if the corpus grows, with
-- lists calculated from the actual row count (lists ≈ rows / 1000).
```

Keep the schema simple for Phase 1. Add `document_id` or versioning fields only if re-ingestion requires it.

**Re-ingestion / idempotency:** Because the corpus is pinned to a commit hash, `ingest` is deterministic for a given corpus version. To keep it idempotent, `ingest` **deletes all existing rows for `source_file`s it is about to (re)insert** (i.e. deletes by `source_file` set before inserting). This prevents duplicate chunks on re-run after chunker or parser tweaks, without needing a version-tracking table in Phase 1. A future `document_id`/`corpus_version` column can replace this when Phase 4 versioning requires it.

### 3.5 Chunking Strategy

**Locked: Markdown-aware semantic chunking** (behind a `Chunker` interface).

| Parameter | Value |
|-----------|-------|
| Split boundary | Markdown headings (H1→H2→H3 as logical boundaries) |
| Code blocks | Kept intact — never split mid-block |
| Tables | Kept intact |
| Target chunk size | ~500–800 tokens |
| Overlap | ~50–100 tokens where useful |
| Metadata preserved | Heading hierarchy, source file, chunk index, code block flag, table flag, heading level |

The `Chunker` interface allows swapping strategies later for benchmarking.

### 3.6 Embeddings

- Model: `BAAI/bge-small-en-v1.5`
- Dimension: 384
- Runtime: local (no API call)
- Provider interface: `EmbeddingProvider`

### 3.7 Retrieval

| Parameter | Value |
|-----------|-------|
| Default `top_k` | 5 |
| Configurable | Yes, via CLI flag or `.env` (`RETRIEVAL_TOP_K`) |
| Reranker | None in Phase 1 |
| Search type | Pure vector cosine similarity (pgvector) |
| Retriever interface | `Retriever` |

### 3.8 LLM Generation

| Property | Value |
|----------|-------|
| Provider | Groq |
| Model | `openai/gpt-oss-20b` (model ID from `.env` `GROQ_MODEL`, default in code) |
| API key | Via `.env` (`GROQ_API_KEY`) |
| Retry/backoff | Yes — retry transient failures (network, 429 rate-limit, 5xx) with exponential backoff + jitter; fail the request only after `GROQ_MAX_RETRIES` attempts. Required so eval latency/failure metrics reflect RAG behaviour, not flaky HTTP |
| Temperature | 0 (deterministic, repeatable answers for evaluation) |
| Interface | `Generator` |

### 3.9 System Prompt Template (Phase 1 Default)

The following prompt is the fixed Phase 1 baseline. It is configurable (can be overridden via config), but this default is used for evaluation unless explicitly changed.

```
You are DocPilot, an evidence-driven assistant for technical documentation.

RULES:
1. Answer using ONLY the provided context. Do not use outside knowledge.
2. If the context does not contain enough evidence to answer the question,
   say exactly: "I don't know — the available documentation does not cover
   this question."
3. Cite every claim using [1], [2], etc., matching the numbered sources
   provided below the context.
4. Never invent citations or reference sources that were not provided.
5. Prefer concise, technically accurate answers. Show code examples only
   when present in the context.
6. Clearly distinguish what is stated in the retrieved docs vs. what is
   your interpretation.

CONTEXT:
{context}

SOURCES:
{sources}

QUESTION:
{question}
```

**Template variables:**
- `{context}` — concatenated retrieved chunk contents, each prefixed with its reference number
- `{sources}` — numbered source list: `[1] source_file → heading_path`
- `{question}` — the user's question

### 3.10 Citation Format

Inline markers in the answer body:
```
To install FastAPI, run `pip install fastapi`. [1]
```

Footer at the end of the answer:
```
Sources:
[1] docs/getting-started.md → Installation
[2] docs/tutorial/first-steps.md → First Steps
```

### 3.11 CLI Interface

```bash
# Ingest docs into pgvector
python -m docpilot ingest [--debug]

# Ask a question
python -m docpilot ask "How do I install FastAPI?" [--debug] [--json]
```

**Flags:**
- `--debug`: enables detailed logging (retrieval scores, prompts, raw LLM response for `ask`; chunk stats for `ingest`)
- `--json`: outputs structured JSON instead of plain text (only for `ask`)

**Output — plain text (default):**
```
To install FastAPI, run `pip install fastapi`.

Sources:
[1] docs/getting-started.md → Installation
```

**Output — JSON (`--json`):**
```json
{
  "answer": "To install FastAPI, run `pip install fastapi`.",
  "sources": [
    {"ref": 1, "file": "docs/getting-started.md", "heading": "Installation"}
  ]
}
```

### 3.12 Logging

- Module: Python `logging` (no external dependencies)
- Output: **stderr** (stdout reserved for answers)
- Levels:
  - `INFO`: ingestion progress, retrieval summary, answer generation timing
  - `DEBUG`: individual chunk scores, full prompts sent to LLM, raw LLM responses
  - `WARNING`: fallback behaviour (empty retrieval, retry needed)
  - `ERROR`: connection failures, missing API keys
- **Never** log secrets or API keys, even at DEBUG level.

### 3.13 Debug Mode (`--debug`)

**`ingest --debug`:**
- Number of files processed
- Total chunks created
- Average/min/max chunk size (tokens)
- Chunks per source file
- Time taken per stage

**`ask --debug`:**
- Query embedding dimension
- Retrieved chunk IDs, cosine scores, source file, heading path
- Full prompt sent to the LLM (system prompt + context + question)
- Raw LLM response before post-processing
- Total latency (retrieval + generation)

### 3.14 Interfaces

All components are behind interfaces (abstract base classes or protocols). Even though only one implementation exists in Phase 1, this allows swapping internals later without rewriting call sites.

| Interface | Phase 1 Implementation |
|-----------|----------------------|
| `DocumentLoader` | FastAPI docs loader (fetch from pinned commit) |
| `Parser` | Markdown/MDX parser |
| `Chunker` | Markdown-aware semantic chunker |
| `EmbeddingProvider` | BGE-small-en-v1.5 (local) |
| `VectorStore` | pgvector |
| `Retriever` | Vector similarity search (top-k) |
| `Reranker` | Not implemented in Phase 1 |
| `Tool` | Not implemented in Phase 1 |
| `Agent` | Not implemented in Phase 1 |
| `Generator` | Groq openai/gpt-oss-20b (via .env GROQ_MODEL) |
| `CitationEngine` | Inline [1] marker + source footer |
| `Evaluator` | Not implemented in Phase 1 |

Do not hard-code `groq.chat(...)` or `qdrant_client.search(...)` calls through business logic — always go through the relevant interface.

### 3.15 Directory Layout (Phase 1)

Proposed structure (not locked — will evolve):

```
docpilot/
├── src/docpilot/              # core library
│   ├── __init__.py
│   ├── __main__.py            # CLI entrypoint
│   ├── cli.py                 # argparse CLI
│   ├── config.py              # .env loading, settings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # DocumentLoader
│   │   ├── parser.py          # Parser
│   │   └── chunker.py         # Chunker interface + impl
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── provider.py        # EmbeddingProvider interface + impl
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # VectorStore interface + pgvector impl
│   │   └── retriever.py       # Retriever interface + impl
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py       # Generator interface + Groq impl
│   │   └── prompts.py         # prompt templates
│   ├── citations/
│   │   ├── __init__.py
│   │   └── engine.py          # CitationEngine interface + impl
│   └── db/
│       ├── __init__.py
│       └── schema.sql         # pgvector table creation
├── docs/                      # raw FastAPI documentation corpus
├── tests/
├── data/                      # vector DB state, eval datasets (Phase 4)
├── notebooks/                 # optional exploration/eval notebooks
├── .env                       # secrets (git-ignored)
├── .gitignore
├── pyproject.toml
└── uv.lock
```

### 3.16 `.env` Keys

```
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MAX_RETRIES=3
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=docpilot
POSTGRES_USER=
POSTGRES_PASSWORD=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE_TARGET=650
CHUNK_OVERLAP=75
RETRIEVAL_TOP_K=5
```

No `.env.example` file — document keys in SPEC.md and README only.

### 3.17 Test Strategy (Phase 1)

Because Phase 4 evaluation must trust "answer correctness," Phase 1 needs unit-level tests per interface implementation so a broken chunker/retriever doesn't surface later as a mysterious eval regression.

**Unit tests (one per interface implementation):**
- `Chunker`: splits code blocks correctly (never mid-block); preserves heading hierarchy metadata; handles tables as atomic units; respects target size + overlap bounds; handles empty/edge inputs.
- `Parser`: correctly extracts source_file, headings, code blocks, tables from sample Markdown/MDX.
- `EmbeddingProvider`: returns correct dimension (384); is deterministic for identical input.
- `VectorStore`/`Retriever`: returns exactly `top_k` results ordered by score; empty result handling; filters by source correctly.
- `CitationEngine`: maps answer `[n]` markers to the correct source indices; refuses/invalidates out-of-range indices.
- `config`: loads `.env` correctly; no secrets leaked.

**Integration tests:**
- Embed → store → retrieve round-trip on a tiny fixture corpus (no LLM call).
- End-to-end test (also in exit criteria): ingest small fixture → ask → answer + citations verified, sources match retrieved chunk `source_file`/`heading_path`. Use a mock/fake `Generator` so tests are deterministic and don't hit the Groq API.

**Test runner:** `pytest` (dev dependency). Tests must not require a live Postgres/Groq connection to run the fast unit suite — use fakes/in-memory stores for interface tests; a separate opt-in/labeled integration suite covers real pgvector.

### 3.18 Exit Criteria (Phase 1 Done)

- [ ] Ingestion pipeline runs end-to-end: fetch FastAPI docs → parse → chunk → embed → store in pgvector
- [ ] Corpus version pinned and recorded (`docs/CORPUS.md` with commit hash)
- [ ] Ingest is idempotent — re-running does not duplicate chunks (delete-by-`source_file` before insert)
- [ ] `python -m docpilot ask "..."` returns a cited answer with correct source references
- [ ] `python -m docpilot ask --json "..."` returns structured JSON output
- [ ] `--debug` flag works on both `ingest` and `ask`
- [ ] "I don't know" response works when context is insufficient
- [ ] Debug view shows retrieved chunks, scores, and full prompt
- [ ] All components behind interfaces (no direct vendor calls in business logic)
- [ ] Groq calls have retry/backoff; temperature = 0
- [ ] Retrieval quality is measurable (manual review of top-k results for sample queries)
- [ ] Answer quality is measurable (manual review with citation checking)
- [ ] Logging to stderr, no secrets exposed
- [ ] Unit tests per interface implementation pass (§3.17)
- [ ] End-to-end test passes: ingest fixture → ask → answer + citations verified (§3.17)
- [ ] Exact (non-ANN) vector search confirmed working; no IVFFlat index in place

---

## 4. Phase 2 — Agentic Retrieval (Outline)

Add LangGraph and build a conditional agentic loop:

1. Analyze the question.
2. Search the documentation.
3. Evaluate whether retrieved evidence is sufficient.
4. Reformulate/retry the search if not sufficient.
5. Stop and answer when evidence is sufficient.
6. Refuse with "I don't know" when evidence stays insufficient.

**Guardrail:** Simple questions route straight through classic RAG (fast, cheap). Agentic looping is conditional — only for multi-hop or ambiguous queries. Never mandatory for every query.

**Exit criteria (Phase 2):** The agent correctly decides when to loop vs. when to answer directly. Measurable improvement over Phase 1 baseline on multi-hop questions (tracked in Phase 4).

---

## 5. Phase 3 — MCP / GitHub Tooling (Outline)

Add one meaningful MCP integration: GitHub.

The agent reaches for GitHub only when static docs are demonstrably insufficient — e.g. live issues, repo state, recent PRs.

**Example:**
> "Does this repo currently have an issue related to OAuth token expiration?"
> Doc search → insufficient → GitHub MCP tool → inspect issues → answer with evidence.

**Guardrail:** MCP solves a real problem (docs can't answer this), not a checkbox. If no natural example exists where it's needed, don't wire it in.

---

## 6. Phase 4 — Evaluation (Outline)

Build a benchmark dataset and track metrics before polishing anything.

**Metrics tracked:**
- Retrieval quality (precision/recall of top-k)
- Answer correctness
- Citation correctness
- Groundedness / hallucination rate
- "I don't know" accuracy
- Latency
- Number of retrieval/tool calls per query

**Required comparison:** Classic RAG (Phase 1) vs. agentic RAG (Phase 2), on the same benchmark. Claims of improvement must be backed by this data.

---

## 7. Phase 5 — API + UI (Outline)

Only after the RAG/agent core is working and evaluated.

- FastAPI backend, **async** (async introduced here for the first time)
- Streaming responses
- Source citations surfaced in the UI
- Retrieved-context debug panel
- Basic session/chat history (SQLite)
- **Chainlit** first; React/Next.js only if time allows

**Amendment (2026-09-05):** Originally Streamlit. Swapped to Chainlit — it's built directly on FastAPI/Starlette (matching the async backend introduced at this phase rather than sitting beside it), has native token streaming, built-in citation elements, a built-in `Step` UI for showing intermediate tool/retrieval/agent steps (a natural fit for the Phase 2 `LoopTraceStep` trace), and built-in chat-history persistence (SQLite/Postgres) instead of hand-rolled session state. No functional requirement above changed — only the framework used to satisfy them.

**Guardrail:** Frontend polish never delays or distorts the RAG/agent core. This amendment does not pull Phase 5 work forward — Phase 2/3/4 still come first per §2's hard rules.

---

## 8. Phase 6 — Code Generation / Validation (Outline)

Only after the core system is reliable:

```
Documentation retrieval → generate code → validate against retrieved
API/schema/examples → return code + sources.
```

Do not pull this into the pitch as a core differentiator until it's actually built.

---

## 9. Framework Extraction (Post-Phase 6)

Extract reusable components (interfaces, orchestration, evaluation harness) into a standalone SDK. This happens explicitly and only after DocPilot itself works end-to-end.

---

## 10. Architectural Rules (All Phases)

1. All components behind interfaces — never hard-code vendor/library calls in business logic.
2. Every retrieval/tool decision is loggable and inspectable.
3. Prefer fewer, well-scoped interfaces over premature generalization.
4. Keep README/pitch language honest — don't claim capabilities that aren't built yet.

---

## 11. Conflict Resolution

- Later-phase feature requested early: implement behind interface if truly needed, but state plainly that it pulls forward work from Phase N.
- "Run agentic loop on every query for consistency": push back — violates Phase 2 guardrail and the cost/latency argument.
- Skip evaluation to polish faster: implement what's asked, note that Phase 4 is still owed.
- Unsure which phase a task belongs to: ask before restructuring code.
