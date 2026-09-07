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
    language      TEXT NOT NULL DEFAULT 'en',  -- derived from source_file first path segment
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
7. Never include code that does not appear in the provided context. Do not
   reconstruct, extend, or embellish code examples from outside the context.
8. Cite only the source that actually backs each claim; never cite a source
   merely because it is present in the context.

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
python -m docpilot ask "How do I install FastAPI?" [--debug] [--json] [--lang <tag>]
```

**Flags:**
- `--debug`: enables detailed logging (retrieval scores, prompts, raw LLM response for `ask`; chunk stats for `ingest`)
- `--json`: outputs structured JSON instead of plain text (only for `ask`)
- `--lang <tag>`: retrieval language filter (default `RETRIEVAL_LANGUAGE`, i.e. `en`; `any` disables filtering and returns chunks from every corpus language) — added 2026-09-06 to fix mixed-language retrieval

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

All components are behind interfaces (abstract base classes or protocols). Even during Phase 1, when only one implementation existed per interface, this design lets internals be swapped later without rewriting call sites.

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

**Phase 2 update (2026-09-06):** `Agent` is implemented — `AgenticAgent` (LangGraph loop) behind `Agent(ABC)` with `agentic_ask()` (§4); `QueryClassifier` (`HeuristicQueryClassifier`) and `SufficiencyJudge` (`LLMSufficiencyJudge`) were introduced in Phase 2 and are not part of the Phase 1 table above. `Tool` becomes Phase 3; `Reranker` and `Evaluator` are Phase 4. The table above is the Phase 1 snapshot.

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

A full template ships at the repo root as `.env.example` — copy it to `.env`
(`cp .env.example .env`) and fill in the values. `.env` is git-ignored; never
commit secrets. Every key:

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
RETRIEVAL_LANGUAGE=en
AGENT_MAX_RETRIES=2
AGENT_LOOP_TOP_K=8
AGENT_DEFAULT_STRATEGY=auto
AGENT_JUDGE_MODEL=
GITHUB_PAT=
GITHUB_API_BASE=https://api.github.com
GITHUB_OWNER=fastapi
GITHUB_REPO=fastapi
```

`GITHUB_OWNER`/`GITHUB_REPO` are the default repository the GitHub tool
queries when a request doesn't name one (code default: empty — the tool
request must then carry `owner`/`repo`).

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

### 3.18 Exit Criteria (Phase 1 Done) — verified 2026-09-06

- [x] Ingestion pipeline runs end-to-end: fetch FastAPI docs → parse → chunk → embed → store in pgvector
- [x] Corpus version pinned and recorded (`docs/CORPUS.md` with commit hash)
- [x] Ingest is idempotent — re-running does not duplicate chunks (delete-by-`source_file` before insert)
- [x] `python -m docpilot ask "..."` returns a cited answer with correct source references
- [x] `python -m docpilot ask --json "..."` returns structured JSON output
- [x] `--debug` flag works on both `ingest` and `ask`
- [x] "I don't know" response works when context is insufficient
- [x] Debug view shows retrieved chunks, scores, and full prompt
- [x] All components behind interfaces (no direct vendor calls in business logic)
- [x] Groq calls have retry/backoff; temperature = 0
- [x] Retrieval quality is measurable (manual review of top-k results for sample queries)
- [x] Answer quality is measurable (manual review with citation checking)
- [x] Logging to stderr, no secrets exposed
- [x] Unit tests per interface implementation pass (§3.17)
- [x] End-to-end test passes: ingest fixture → ask → answer + citations verified (§3.17)
- [x] Exact (non-ANN) vector search confirmed working; no IVFFlat index in place
- [x] Retrieval is language-filtered — English default; English queries return English chunks and English citations only (fix 2026-09-06)

**Live verification evidence (2026-09-06, full corpus):**
- Corpus ingested completely: 16,535 chunks / 1,657 files — disk ↔ DB set-diff shows 0 missing, 0 extra.
- Idempotency probe: deleting + re-ingesting `en/docs/tutorial/first-steps.md` through the real pipeline stays flat at 25 chunks across runs; no duplication, no leaked fixture rows (0 `%integration%` rows).
- `language` column backfilled from `source_file` (13 language tags; no nulls; `en` = 2,416 chunks).
- Post-fix English QA (live Groq + pgvector): "How do I install FastAPI?", "What is a query parameter in FastAPI?", "How do I run FastAPI with uvicorn?" — all retrieved chunks and all cited sources are English and grounded. `--lang any` restores unfiltered (multi-language) retrieval.
- Full test suite: 134 passed (131 hermetic + 3 live pgvector integration).

---

## 4. Phase 2 — Agentic Retrieval — COMPLETE 2026-09-06 (milestone `phase-2`)

Implements the conditional agentic loop behind the `Agent` interface (first
implementation; PLAN §9 interface list). Fast path (Phase 1 `pipeline_ask.ask`)
stays untouched — default `ask` behavior is identical to Phase 1.

### 4.1 Locked decisions (2026-09-05, user sign-off)

- **Engine: LangGraph.** Loop graph: gate → retrieve → judge → (reformulate |
  answer+cite | refuse).
- **Gate: heuristics only, zero LLM calls.** Word-count, connector words
  (and/or/how/combine/difference…), multiple distinct tech terms →
  `direct | agentic`. Deterministic and instant.
- **Judge: LLMSufficiencyJudge, one LLM call per retry.** Structured output
  `{verdict: sufficient|insufficient|ambiguous, reason, reformulated_query|null}`.
  `reformulated_query` is its own field (never an unstructured blob): reformulation
  is folded into the judge today, but a future real `Reformulator` can replace
  internals without changing the judge's call site.
- **Budget:** `AGENT_MAX_RETRIES` (default 2), hard-enforced in the graph;
  exhausted → refuse with the §3.9 "I don't know" wording.
- **Tracing:** our `LoopTraceStep` only. LangGraph/LangSmith observability is NOT
  used — it is vendor surface outside the interface table.
- **CLI:** `ask --strategy auto|direct|agentic` (default `auto`). No `--agentic`
  alias — one way to do one thing.
- **Latency discipline:** per-step latencies are recorded in the trace for
  inspectability (applies from day one, §10). Phase 2 computes no aggregates,
  draws no classic-vs-agentic comparison, and claims no performance win — all of
  that is Phase 4 evaluation work.

### 4.2 Components

| Component | Responsibility | Lives in |
|---|---|---|
| `QueryClassifier` (ABC) + `HeuristicQueryClassifier` | classify(question) → `direct` \| `agentic`; no LLM | `agent/gate.py` |
| `SufficiencyJudge` (ABC) + `LLMSufficiencyJudge` | judge(context, sources, question) → `Judgment` (one LLM call, structured) | `agent/judge.py` |
| `Agent` (ABC) + `AgentResult` | run(question, *, deps…, strategy) → AskResult + trace; `direct` strategy = classic ask() | `agent/interface.py` |
| `StateGraph` | routes gate/retrieve/judge/reformulate/answer/refuse; enforces budget | `agent/graph.py` |
| `agentic_ask` | orchestrator entry point | `agent/pipeline_agentic.py` |

Shared contract: `agent/types.py` (`GateDecision`, `Judgment`, `LoopTraceStep`,
loop state) — write-once, all agents treat as read-only. `core/models.py` is
unchanged (contract_version 1.0).

### 4.3 Config

| Key | Default | Purpose |
|---|---|---|
| `AGENT_MAX_RETRIES` | 2 | Max judge/reformulate iterations |
| `AGENT_DEFAULT_STRATEGY` | auto | CLI overridable via `--strategy` |
| `AGENT_GATE_LONG_THRESHOLD` | 18 | Word-count gate trigger |
| `AGENT_JUDGE_MODEL` | (empty → `GROQ_MODEL`) | Optional separate judge model |
| `AGENT_LOOP_TOP_K` | 8 | Loop retrieve breadth (fast path keeps `RETRIEVAL_TOP_K`=5; caller override wins) |

### 4.4 Trace

Every `LoopTraceStep`: turn # → `query_used`, `verdict`, `reason`,
`reformulated_query`, `retrieved_count` + top scores, step label, step `latency_ms`.
Logged at DEBUG on stderr; surfaced as additive `"trace"` in `--json`. Latency
fields are for inspectability only (§4.1).

### 4.5 Exit criteria (Phase 2) — live-verified 2026-09-06/07

- [x] Gate is correct on the seed question set (multi-hop → agentic, simple → direct).
- [x] Loop answers the multi-hop seed questions with correct `[N]` citations + footer.
- [x] Budget enforced (≤ `AGENT_MAX_RETRIES`), then refuses.
- [x] "I don't know" (exact §3.9 wording) when evidence stays insufficient.
- [x] Fast-path regression: simple questions behave identically to Phase 1.
- [x] Trace per query in `--debug` (stderr) and `--json` (`"trace"`).
- [x] All behind the `Agent` interface; LangGraph code has no vendor calls; our
      tracing only; CLI surface is exactly `--strategy auto|direct|agentic`.
- [x] Phase 2 produces no performance numbers or Phase-1-vs-2 comparisons.
- [x] Full test suite green; agent tests hermetic (stubbed Generator), integration
      variants marked and skip-if-unreachable.

**Live evidence (seed QA + edge QA, 2026-09-06/07):** 12/12 seed questions run
against the live stack (Groq `gpt-oss-20b`, BGE-small, pgvector, LangGraph
1.2.11). Gate: all 8 multi-hop seeds route agentic (seeds 3/7/8 via the
standalone `multi_concept` signal — ≥3 distinct concepts — a fix from the first
QA round where they mis-routed direct); simple seeds 9/10 stay direct. Loop:
6/8 multi-hop seeds answered with correct `[N]` citations + footer, judged
`sufficient` on attempt 1; seeds 4/6 refused honestly after the 2-round budget
(judge: chunks lacked explicit evidence for the "same model reused for body +
response_model" / "Annotated vs legacy mixability" claims). Refuse seeds 11/12
emit the verbatim §3.9 sentence. Fast-path regression byte-identical
(`diff` empty); `--json` trace `gate→search→judge→answer` with int latencies and
additive keys; judge calls ≤ `AGENT_MAX_RETRIES` on every loop run; degenerate
inputs exit 0 with no tracebacks. Suite 193 passed (190 hermetic + 3 live).

**Guardrail (unchanged):** Simple questions route straight through classic RAG
(fast, cheap). Agentic looping is conditional — only for multi-hop or ambiguous
queries. Never mandatory for every query.

**Measurable-improvement note:** the formal classic-vs-agentic comparison and its
metrics are Phase 4 work. Phase 2's exit criteria above are functional and
inspectable; no improvement claim is made here.

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
