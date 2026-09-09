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
- Calls out to GitHub — a plain REST tool behind the `Tool` interface, no MCP (Phase 3+) — when static docs can't answer.
- Returns "I don't know" rather than hallucinating.
- Every retrieval and tool decision is loggable and inspectable.

**Purpose:** DocPilot is the learning/validation project for a future reusable RAG framework. DocPilot comes first; the framework is extracted from it later.

---

## 2. Phased Build Order

```
Phase 1: Classic RAG → Phase 2: Agentic retrieval → Phase 3: GitHub tooling →
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

**Phase 3 update (2026-09-07):** `Tool` is implemented — `ToolRequest`/`ToolResult` + `Tool(ABC)` with one implementation, `GitHubTool` over GitHub REST (no MCP — §5). The agentic loop gained the judge's structural `needs_tool`/`tool_request` signal and a `tool_call` graph node; tool evidence appears as a `LIVE GITHUB EVIDENCE:` context section with continuing footer sources. CLI surface is unchanged (no new flags). `Reranker` and `Evaluator` remain Phase 4. The table above is the Phase 1 snapshot.

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

## 5. Phase 3 — GitHub Tooling — implemented 2026-09-07 (live QA pending)

Adds one meaningful external-data tool — GitHub — reached only when static docs are demonstrably insufficient (live issue state, repo state, recent commits). Originally labelled "MCP / GitHub Tooling"; the locked decision (user sign-off 2026-09-07) is a **plain GitHub REST tool behind the `Tool` interface — no MCP protocol or SDK**.

**Example:**
> "Does this repo currently have an issue related to OAuth token expiration?"
> Doc search → insufficient → judge `needs_tool` → GitHub tool → inspect issues → answer with evidence.

**Guardrail:** the tool solves a real problem (docs can't answer this), not a checkbox. If no natural example exists where it's needed, it isn't wired in.

### 5.1 Locked decisions (2026-09-07, user sign-off)

- **No MCP.** Plain GitHub REST via `requests` (`Accept: application/vnd.github+json`, `Authorization: Bearer <GITHUB_PAT>`). The `Tool` interface keeps vendor action names namespaced (`github.search_issues`), so a second tool slots in without touching the agent.
- **Tool interface:** `Tool(ABC)` + `ToolRequest(name, params)` + `ToolResult(ok, summary, source_label, items, error)`; implementations never raise — errors are non-`ok` results. One implementation: `GitHubTool` (`github.search_issues`, `github.list_issues`, `github.get_commits`).
- **Tool decisions are free:** `needs_tool` + `tool_request` ride the judge's existing structured output — no separate LLM call decides tooling.
- **Trigger contract:** `needs_tool=true` only with verdict `"insufficient"` and only for live-state gaps a GitHub call could answer (open issues, repo state, current commits). Doc-content gaps retry the retrievers (`reformulated_query`) instead; when `needs_tool=true`, `reformulated_query` must be null.
- **Graph wiring:** one `tool_call` node per question, only within budget (`attempts < AGENT_MAX_RETRIES` — budget trumps the tool on the last round); the tool NEVER loops back to the judge. Success → answer with tool evidence; failure → verbatim §3.9 refusal.
- **Availability gating:** with no tool (no `GITHUB_PAT`) the judge is told `TOOLS AVAILABLE: no` and the loop is byte-identical Phase 2; a defensive `needs_tool` degrades to plain insufficient — a tool that isn't wired is never called.
- **Evidence & citations:** tool items become continuing `SourceRef`s (`github:{owner}/{repo}#{n}` / `@{sha}`) surfaced as a separated `LIVE GITHUB EVIDENCE:` context section numbered `[k+1]…` after the doc chunks; the citation footer lines up with the `[n]` markers.

### 5.2 Components

| Component | Responsibility | File |
|---|---|---|
| `Tool` (ABC), `ToolRequest`, `ToolResult` | vendor-free call-site interface; never raises | `tools/base.py` |
| `GitHubTool` | `search_issues` / `list_issues` / `get_commits` over REST; repo-scoped `q`; per-item `source_label` | `tools/github.py` |
| `Judgment.needs_tool` / `tool_request` | structural tool signal from the judge (tolerant parse) | `agent/types.py`, `agent/judge.py` |
| `tool_call` graph node + routers | judge → tool_call (within budget) → answer \| refuse | `agent/graph.py` |
| `agentic_ask` / `AgenticAgent` `tool` | default `GitHubTool()` when `config.GITHUB_PAT` is set | `agent/pipeline_agentic.py` |

### 5.3 Config

| Key | Default | Purpose |
|---|---|---|
| `GITHUB_PAT` | "" | GitHub token; empty → tool disabled, loop Phase 2-identical |
| `GITHUB_API_BASE` | `https://api.github.com` | API base (also the hermetic-test seam) |
| `GITHUB_OWNER` / `GITHUB_REPO` | "" | default repo when a tool request doesn't name one |

### 5.4 Trace

The judge step records `decision=insufficient/needs-tool` with the `tool_request` in `detail`; the `tool_call` step records decision `tool_call` or `tool_error`, the action, item count / error message and `latency_ms` (step label `"tool_call"` flows through `--json` like any other step). No secrets (never the PAT) in any trace or log output.

### 5.5 Exit criteria (Phase 3)

- [x] `Tool` interface + `GitHubTool` over GitHub REST (no MCP), gated by `GITHUB_PAT`.
- [x] Judge emits structured `needs_tool` + `tool_request` (no extra LLM call); tolerant parsing with safe defaults.
- [x] `tool_call` graph node: within budget, single call, never loops back; success → answer with evidence, failure → verbatim §3.9 refusal.
- [x] No tool / no PAT → Phase 2-identical loop (judge told tools unavailable).
- [x] Tool evidence cited: `LIVE GITHUB EVIDENCE` context section + continuing footer sources.
- [x] Full suite 230 passed (227 hermetic + 3 live pgvector integration).
- [ ] Live QA on the real GitHub API against `fastapi/fastapi`: tool-fires (issues + commits), tempt-the-tool guardrail (no `tool_call` on doc-answerable questions), refusal regression. **Blocked only on `GITHUB_PAT` in `.env`** — until it runs, live tool behavior (judge `needs_tool` accuracy, trigger necessity) is unmeasured and claimed nowhere.

---

## 6. Phase 4 — Evaluation (scope locked 2026-09-07)

Build a benchmark dataset and track metrics before polishing anything. Nothing
is claimed about "agentic RAG being better" — or the Phase 3 tool being worth
its cost — until this phase's measurements exist (§2 hard rules).

### 6.1 Baseline metrics + required comparison

- Retrieval quality (precision/recall of top-k)
- Answer correctness
- Citation correctness
- Groundedness / hallucination rate
- "I don't know" accuracy
- Latency
- Number of retrieval/tool calls per query

**Required comparison:** Classic RAG (Phase 1) vs. agentic RAG (Phase 2) on the
same benchmark. Claims of improvement must be backed by this data, not asserted.

**Scoring contract (amended 2026-09-08):** for docs-answerable rows, gold key
facts are short paraphrase-robust fragments — an identifier or concept token
every correct answer must contain — rather than verbatim corpus sentences.
Verbatim-sentence containment measures paraphrase-avoidance, not correctness;
on the first live run it scored 0.00 across all doc questions while the sidecar
evidence showed every retrieval and generation was correct. Gold source files
are the stored corpus-relative paths exactly as the vector store returns them
(not filesystem paths), so retrieval recall and citation-gold accuracy compare
against strings that actually occur in `source_files`.

### 6.2 Client-required evaluation scope (locked 2026-09-07, client review)

- **False-refusal rate + 3-way decomposition.** Every refusal is classified as a
  retrieval failure, a judge failure, or a routing failure (gate misroute).
  Supersedes the Phase 2 "seed refusals are honest" verdict as a claim —
  refusals count as honest only when the decomposition shows the evidence
  genuinely was insufficient.
- **Judge-specific calibration.** Labeled verdict triples (question, context,
  gold verdict) + adversarial paraphrases + parse-failure rate
  (empty / unparseable / bad-verdict judge outputs per question). Judge-only —
  independent of retrieval and routing quality.
- **Tool-necessity evaluation.** Gold-labeled tri-class questions:
  docs-answerable / live-state-answerable / neither. Reports tool false-positive
  rate (tool fired when docs alone could answer) and false-negative rate
  (live-state question where the tool never triggered). This is Phase 3's
  necessity measurement — SPEC §5.5 explicitly claims nothing about live tool
  behavior until this runs.
- **Judge two-prompt A/B (first evaluation slice).** The current single judge
  prompt is pressure-tested against a second prompt variant on the same labeled
  verdict triples; prompt changes are adopted on calibration data alone, never
  on judgment. Also serves as the two-implementation pressure test for the
  `SufficiencyJudge` interface.

### 6.3 Known open questions feeding the eval set

- Phase 2 seeds 4/6 "honest refusals" may be **false refusals** — the FastAPI
  corpus likely contains retrieved evidence for `response_model=Item` with
  `item: Item` (mixin-style body), meaning retrieval may have failed instead of
  the judge judging honestly. Resolved by the 3-way decomposition eval, not by
  re-running one question ad hoc.
- The judge parse-fallback default stays `sufficient` (the generator's §3.9
  honesty gate remains the final safety layer) **resolved by the flip-condition
  check (2026-09-09): KEEP**. Flip conditions, per Phase 4 data: a non-trivial
  live parse-fallback rate (made measurable by the judge fallback counter), or
  the generator under-refusing on weak evidence. The check measured 0 live
  parse-fallbacks and a 0.0 calibration parse-failure rate, and observed the
  generator self-refusing with the verbatim §3.9 sentence on weak evidence
  (benchmark bn03) — neither condition met. Evidence: `parse_fallback_flip_
  20260909_091251.json` (SPEC §6.4).

### 6.4 Exit criteria (Phase 4)

- [x] Classic-RAG vs. agentic-RAG comparison on all §6.1 metrics, one table.
      **Run 2026-09-08/09, rescored 2026-09-09** — `src/docpilot/eval/reports/
      benchmark_20260908_190539.json` (combined + sidecars; see PLAN §5.3 for
      the table and reading).
- [x] False-refusal decomposition (retrieval / judge / routing) on the benchmark.
      **Run 2026-09-09** — `src/docpilot/eval/reports/
      false_refusals_20260909_091251.json`: 0 false refusals agentic,
      2/15 classic (bl01/blob03 — capability-routing: live-state questions hit
      the no-tool fast path), 0 judge-caused; all HR/neither refusals honest.
- [x] Judge calibration report: verdict accuracy, adversarial robustness,
      parse-failure rate, and the two-prompt A/B result.
- [x] Tool-necessity report: false-positive / false-negative rates on the
      tri-class set.
- [x] Parse-fallback flip-condition check run, with the keep-vs-flip decision
      recorded. **Run 2026-09-09** — decision **KEEP** the defensive
      `sufficient` default: 0 live judge parse-fallbacks (16-question agentic
      half) and 0.0 calibration parse-failure rate (24 triples, both prompts),
      and the generator's §3.9 honesty gate demonstrably refused on weak
      evidence (bn03). Evidence: `src/docpilot/eval/reports/
      parse_fallback_flip_20260909_091251.json`.

### 6.5 Live-run protocol & quota resilience (amended 2026-09-08, user sign-off)

Phase 4 live runs draw judge, generator, and grounding-checker calls from one
model-level daily token budget (`openai/gpt-oss-20b` free tier: 200k TPD /
8k TPM), and a full benchmark pass is token-heavy — every question embeds full
retrieved contexts and agentic questions make 3–5 LLM calls each. Live-run
protocol, locked:

- **Per-half checkpointing.** Each pipeline's scored run is written to a
  sidecar file the moment it completes; the combined report + §6.1 comparison
  is written only when both halves exist. A quota crash never loses a finished
  half.
- **Resume-at-stamp.** The missing half is rerun under the crashed run's
  original stamp (`--pipeline {classic,agentic} --resume STAMP`); the finished
  sidecar is adopted and the same-stamp combined report produced.
  `--merge DIR STAMP` merges two existing sidecars. Granularity is per
  pipeline — a crashed half reruns wholesale; per-question resume is not in
  scope.
- **Probe before launch.** Live runs are gated on a `--probe` functional check:
  auth + model reachable + one real completion served + per-minute token
  headroom for a call right now. **TPD is not probe-able** — Groq exposes only
  per-minute buckets as response headers and does not gate admission on
  `max_tokens` (verified live 2026-09-08: a 16k-`max_tokens` probe was admitted
  and served ~18k output tokens on an org whose daily bucket was ~500 tokens
  from the wall; a tiny call can likewise return OK inside a nearly exhausted
  daily bucket). The probe therefore claims nothing about daily headroom:
  before a live run the operator must assert the key's org has a fresh daily
  bucket, and a mid-run TPD wall is absorbed by per-half checkpointing +
  resume-at-stamp + retry-after fail-fast rather than by the gate.
- **429 self-healing.** The generator honors the server's `retry-after` on
  rate-limit errors (bounded), so per-minute bursts recover; a daily-cap wall
  fails fast instead of hanging retries.
- **Provenance.** A resumed half is flagged in the report note; the §6.1 table
  states its run provenance (single continuous run vs. resumed windows) next
  to the numbers. Same-model temp-0 runs still show run-to-run variance in the
  NLI-style groundedness audit, which is documented as a proxy.
- **Token accounting (open).** The generator discards Groq's per-call `usage`;
  token consumption is not yet inspectable from committed reports. Recording
  token usage is a follow-up harness improvement so quota economics stay
  auditable.

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
