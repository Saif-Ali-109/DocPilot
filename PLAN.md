# DocPilot — Execution Plan

## 1. meta

- pitch: >
    DocPilot ingests real technical documentation (Markdown/MDX, code blocks,
    nested headings) and answers questions using retrieved evidence — but instead
    of naive retrieve-and-answer, it evaluates whether its own evidence is
    sufficient, retries searches when it isn't, reaches for live GitHub data via
    a plain REST tool (no MCP — SPEC §5) when static docs can't answer, and says
    "I don't know" rather than hallucinating. Citations and retrieval/tool
    decisions are exposed throughout for debugging and demonstration.
- governing_rule: >
    Reliable RAG → Agentic retrieval → GitHub tooling → Evaluation → API/UI →
    optional code validation → extract reusable framework components.
    Do not skip ahead. Do not let later phases' ambitions leak into earlier
    phases' scope.
- current_phase: 3
- working_repo: https://github.com/Saif-Ali-109/DocPilot.git
- working_dir: /home/ain/Desktop/RAG
- spec_source_of_truth: >
    SPEC.md (v0.1.0, approved) is the source of truth for all locked decisions.
    PLAN.md holds the HOW: task breakdown and parallel build orchestration.

---

## 2. phase_1_active: "Excellent Classic RAG (the baseline)" — COMPLETE 2026-09-06

- phase_1_status: COMPLETE (milestone `phase-1-complete` cut 2026-09-06)
- pipeline: >
    Docs → parsing → code-aware chunking → local embeddings → pgvector →
    retrieval → Groq LLM → cited answer
- locked_decisions: |
    Corpus      : FastAPI docs .md/.mdx, pinned commit, recorded in docs/CORPUS.md
    Environment : Python >=3.13,<3.14; uv + pyproject.toml + uv.lock (no requirements.txt)
    Config      : .env only; secrets never committed
    Embeddings  : BAAI/bge-small-en-v1.5 (384-dim, local)
    Vector DB   : pgvector, localhost:5432, db=docpilot
    Schema      : single `chunks` table (see SPEC.md §3.4)
    ANN index   : NONE in Phase 1 (exact cosine search; corpus ~1-3k rows)
    LLM         : Groq openai/gpt-oss-20b (via .env GROQ_MODEL; was llama-3.1-8b-instant, retired on Groq), temp=0, retry+backoff
    Chunking    : Markdown-aware semantic; ~500-800 tokens; ~50-100 overlap;
                  code blocks + tables intact; heading-hierarchy metadata
    Citations   : inline [1] markers + source footer
    CLI         : python -m docpilot ingest [--debug]; ask "..." [--debug] [--json]
    Output      : plain text default; --json for structured
    Async       : none
    Retrieval   : top_k=5 (configurable)
    Reranker    : none

## 2.1 shared_contracts

- contract_version: 1.0
- rule: >
    Defined once for ALL parallel agents. Single-owner write-once; all other
    agents treat them as read-only imports. Any addition to a contract goes
    through its owner and bumps contract_version.
- db_schema:
    owner: AGENT C
    file: src/docpilot/db/schema.sql
    note: >
      Single source of truth for the `chunks` table DDL. Mirrors SPEC.md §3.4
      exactly (vector + pgcrypto extensions, NO ANN index). No other agent
      writes this file or creates the table.
- data_models:
    owner: AGENT B
    file: src/docpilot/core/models.py
    types:
      - Document   [file_path, content, frontmatter]
      - Chunk      [id, content, heading_path, source_file, chunk_index, metadata]
      - RetrieverResult [chunk, score]
      - SourceRef  [ref, file, heading]
    note: >
      Single source of truth for shared data types, imported (never modified)
      by all other agents. No agent redefines or duplicates these types.

## 2.2 rules

- rule_global: >
    STRICT single-owner per file. No agent edits a file owned by another agent.
    Shared contracts (core/models.py, db/schema.sql, pyproject.toml, uv.lock,
    .env) are write-once by their owner; all others treat them as read-only.
- rule_secrets: Never log or commit secrets/API keys. .env is git-ignored.
- rule_testing: >
    Each owner writes tests for its own module under tests/ next to its work;
    no two agents write the same test file.
- deps_installer: >
    Only AGENT A runs uv sync / installs dependencies (owns pyproject.toml,
    uv.lock). Other agents list needed deps for A in deps.md; they do NOT
    modify pyproject.toml / uv.lock themselves.
- rule_git: >
    Every agent commits and pushes its OWN task's output (see §11 git_workflow).
    Commit when the task is complete and its tests pass. Never commit another
    agent's files. This keeps git history synced with the working tree as work
    progresses.

## 2.3 agents

### AGENT B — Core models + Ingestion
- files_owned:
  - src/docpilot/core/__init__.py
  - src/docpilot/core/models.py          # shared contracts (owner)
  - src/docpilot/ingestion/__init__.py
  - src/docpilot/ingestion/loader.py      # DocumentLoader (documented interface)
  - src/docpilot/ingestion/parser.py      # Parser (documented interface)
  - src/docpilot/ingestion/chunker.py     # Chunker (interface + Markdown impl)
  - src/docpilot/ingestion/fastapi_loader.py  # fetch docs from pinned commit
  - tests/test_core_models.py
  - tests/test_ingestion_parser.py
  - tests/test_ingestion_chunker.py
- deps_needed: [markdown parsing pkg]
- depends_on: [AGENT A STARTED (scaffold + pyproject + env)]
- deliverable: >
    Chunker produces list[Chunk] per Document exactly matching core/models.py
    types: heading_path, source_file (relative under docs/), chunk_index,
    metadata {code_block, table, heading_level}.
- guards:
  - Code blocks and tables never split mid-block.
  - Chunk size ~500-800 tokens; overlap ~50-100 tokens.
  - Unit tests cover code-block integrity, heading metadata, table atomicity.

### AGENT C — Embeddings + VectorStore + Retriever + DB
- files_owned:
  - src/docpilot/db/__init__.py
  - src/docpilot/db/schema.sql            # chunks DDL (contract owner)
  - src/docpilot/db/connection.py         # pg connection from .env
  - src/docpilot/embeddings/__init__.py
  - src/docpilot/embeddings/provider.py   # EmbeddingProvider interface + BGE impl
  - src/docpilot/retrieval/__init__.py
  - src/docpilot/retrieval/vector_store.py # VectorStore interface + pgvector impl
  - src/docpilot/retrieval/retriever.py   # Retriever interface + top-k impl
  - tests/test_db_schema.py
  - tests/test_embeddings_provider.py
  - tests/test_retrieval_retriever.py
- deps_needed: [psycopg[binary], pgvector, sentence-transformers]
- depends_on: [AGENT A STARTED, AGENT B STARTED (imports core/models.py)]
- deliverable: >
    Store chunks with 384-dim embeddings; exact cosine top_k search returning
    list[RetrieverResult]; retriever returns exactly top_k (default 5) ordered
    by score; idempotent ingest delete-by-source_file.
- guards:
  - All behind EmbeddingProvider / VectorStore / Retriever interfaces.
  - No pgvector query outside vector_store impl.
  - Idempotency: delete source_file rows before insert.
  - No ANN index.

### AGENT D — Generation + Citations
- files_owned:
  - src/docpilot/generation/__init__.py
  - src/docpilot/generation/generator.py   # Generator interface + Groq impl
  - src/docpilot/generation/prompts.py     # Phase 1 system prompt (SPEC §3.9)
  - src/docpilot/citations/__init__.py
  - src/docpilot/citations/engine.py        # CitationEngine + inline [n] + footer
  - tests/test_generation_prompts.py
  - tests/test_citations_engine.py
- deps_needed: [groq]
- depends_on: [AGENT A STARTED]
- deliverable: >
    Generator takes context + sources + question, calls Groq per SPEC §3.8
    (temp=0, retry+backoff, .env GROQ_MODEL), returns answer with [n] markers.
    CitationEngine maps markers to SourceRef and produces footer.
- guards:
  - Unit tests use a fake generator; never call live Groq API in tests.
  - Prompt template matches SPEC.md §3.9 exactly as baseline.
  - Never invent citations; engine refuses out-of-range indices.

### AGENT E — CLI + Orchestration + E2E testing
- files_owned:
  - src/docpilot/cli.py                     # argparse: ingest / ask commands
  - src/docpilot/__main__.py                # entrypoint (python -m docpilot)
  - src/docpilot/pipeline_ingest.py         # wires loader→parser→chunker→store
  - src/docpilot/pipeline_ask.py            # wires retrieve→generate→cite
  - tests/test_cli.py
  - tests/test_pipeline_e2e.py
- deps_needed: [python-dotenv]
- depends_on: [AGENT B, AGENT C, AGENT D COMPLETE]  # LAST agent
- deliverable: >
    Working CLI: `python -m docpilot ingest [--debug]` and
    `python -m docpilot ask "..." [--debug] [--json]`. Orchestrates all interface
    implementations. E2E test uses a fake Generator (no live API).
- guards:
  - --debug shows retrieval details, scores, prompt/context, raw response (ask)
    and chunk stats (ingest); never exposes secrets.
  - logging to stderr only; stdout has answers only.
  - --json returns structured source list.

### AGENT A — Scaffold + corpus + env
- files_owned:
  - pyproject.toml, uv.lock, .gitignore
  - src/docpilot/__init__.py
  - src/docpilot/config.py                  # .env loader + settings
  - docs/ (fetched FastAPI corpus + docs/CORPUS.md)
  - README.md (stub; honest scope, Phase 1 only)
- deps_needed: [uv]
- depends_on: []  # FIRST agent
- deliverable: >
    Project scaffold, uv venv + lock, config loader, pinned FastAPI docs corpus
    under docs/ with CORPUS.md recording repo + commit hash.
- guards:
  - pyproject/uv.lock write-only by A; others must not touch.
  - .env keys per SPEC §3.16; template ships as `.env.example`.
  - config.py reads all .env keys; survives missing optional keys with clear errors.

### deps.md
- purpose: >
    Single shared notes file (owned by coordinating process) where AGENT B, C,
    D, E each declare their deps_needed. AGENT A reads deps.md once before
    uv sync. Only the coordinator writes deps.md; agents append via it.

## 2.4 parallel_execution

- start_order: >
    Start AGENT A first (unblocks scaffold + deps + corpus). Then run
    AGENT B, C, D in parallel (C depends on A STARTED + B's core/models.py
    contract). AGENT E runs last, after B, C, D complete.
- sequence: "AGENT A → {AGENT B, AGENT C, AGENT D} parallel → AGENT E"
- no_conflict_guarantee: >
    File ownership is disjoint across agents. Shared contracts
    (core/models.py, db/schema.sql, pyproject.toml, uv.lock, .env) are
    single-owner write-once; others read-only. Deps requests flow through the
    single deps.md file. Test files are owned per-agent, so no two agents write
    the same file. This prevents any file overwriting or contradiction.

## 2.5 exit_criteria
- ingestion: Ingestion pipeline runs E2E (fetch → parse → chunk → embed → store)
- corpus_pinned: Commit hash recorded in docs/CORPUS.md
- idempotent: Re-running ingest does not duplicate chunks
- ask_plain: python -m docpilot ask "..." returns cited answer, correct sources
- ask_json: python -m docpilot ask --json "..." returns structured output
- debug: --debug works on both ingest and ask
- dont_know: "I don't know" returned when context insufficient
- interfaces: All components behind interfaces; no direct vendor calls
- groq: temp=0, retry+backoff, GROQ_MODEL configurable
- retrievable: Retrieval quality measurable (manual top-k review)
- answerable: Answer quality measurable (manual citation check)
- logging: stderr logs, no secrets exposed
- unit_tests: Per-interface unit tests pass (see AGENT guards)
- e2e_test: E2E test passes with fake Generator
- exact_search: Exact (non-ANN) cosine search verified; no IVFFlat index

### 2.6 phase_1_verification (2026-09-06, live corpus)

- corpus_complete: 16,535 chunks / 1,657 files — disk ↔ DB set-diff: 0 missing, 0 extra
- idempotent_probe: re-ingesting `en/docs/tutorial/first-steps.md` twice through the real pipeline → 25 chunks flat (delete-by-source then re-insert, no duplication)
- fixture_hygiene: 0 `%integration%` leaked rows
- language_filter: `chunks.language` column backfilled from source_file (13 tags, no nulls, en=2,416); default retrieval `RETRIEVAL_LANGUAGE=en`; `ask --lang <tag>` / `--lang any`
- english_qa: post-fix English queries retrieve English chunks only; cited sources English and grounded
- suite: 134 passed (131 hermetic + 3 live pgvector integration)

---

## 3. phase_2: "Agentic Retrieval" — COMPLETE 2026-09-07 (milestone `phase-2`)

- status: COMPLETE (milestone `phase-2` cut 2026-09-07; §3.7 exit criteria live-verified, §3.11)
- summary: >
    LangGraph conditional loop behind the `Agent` interface (PLAN §9 interfaces,
    first implementation). The loop analyzes → searches → judges evidence
    sufficiency → reformulates/retries → answers or refuses — while simple
    questions keep routing through the Phase 1 classic-RAG fast path unchanged.
- locked_decisions: |
    Engine      : LangGraph (SPEC §4). Loop = gate → retrieve → judge →
                  reformulate|answer+cite|refuse StateGraph.
    Gate        : Heuristic QueryClassifier ONLY (zero LLM calls, deterministic):
                  word-count, connector words (and/or/how/combine/difference…),
                  multi tech-term detection → direct | agentic.
    Judge       : LLMSufficiencyJudge, ONE LLM call per retry, structured output
                  {verdict: sufficient|insufficient|ambiguous, reason,
                  reformulated_query|null}. reformulated_query is its own field
                  (never part of an unstructured blob) — folded into the judge
                  today, splittable into a real Reformulator later by swapping
                  internals only, never the judge's call site.
    Fast path   : pipeline_ask.ask() untouched; default behavior identical to Phase 1.
    CLI         : ask --strategy auto|direct|agentic (default auto). NO --agentic
                  alias/shorthand — one way to do one thing (§10.4).
    Retries     : AGENT_MAX_RETRIES (default 2); budget hard-enforced in the graph;
                  exhausted → refuse with the SPEC §3.9 "I don't know" wording.
    Tracing     : OUR LoopTraceStep only. LangGraph/LangSmith tracing explicitly
                  NOT used (vendor surface not in the interface table).
    Latency     : Per-step ms collected in the trace for inspectability ONLY
                  (§10.2). NO aggregates, NO Phase1-vs-2 comparison, NO
                  "X% faster" claims in Phase 2 docs — that is Phase 4 work.
    Approach    : draft the seed question set from the corpus; formal benchmark
                  dataset + metrics belong to Phase 4 and are NOT built here.

## 3.1 graph

```
(question) ──► [gate] ── direct ──► classic ask() fast path (exact Phase 1)
                 │  agentic
                 ▼
            [retrieve]   (reuses SimpleRetriever)
                 ▼
            [judge]      (LLM: verdict + reason + reformulated_query)
                 │
    ┌────────────┼──────────────────────────┐
    │ SUFFICIENT │ INSUFFICIENT/AMBIGUOUS   │ EXHAUSTED BUDGET
    ▼            ▼                          ▼
 [answer+cite] [reformulate]──count ≤ max?─► [answer+cite]
    (GEN reuse)              │ > max
                             └─────────────► [refuse]  ("I don't know")
```

## 3.2 files & ownership (mirrors §2.3)

- NOTE: `src/docpilot/agent/types.py` is the Phase 2 shared contract (write-once,
  owned by AGENT F like §2.1; AGENT G treats it read-only). No contract_version
  bump in core/models.py — Phase 2 types live in agent/types.py.

### AGENT F — Agent core components ("the brain")
- files_owned:
  - src/docpilot/agent/__init__.py
  - src/docpilot/agent/interface.py    # Agent (ABC) + AgentResult dataclass
  - src/docpilot/agent/types.py        # shared contract: GateDecision, Judgment,
                                       # LoopTraceStep, loop state types (write-once)
  - src/docpilot/agent/gate.py         # QueryClassifier ABC + HeuristicQueryClassifier
  - src/docpilot/agent/judge.py        # SufficiencyJudge ABC + LLMSufficiencyJudge (folded reformulation)
  - src/docpilot/agent/prompts.py      # judge + reformulation prompts (SPEC §3.9 style honesty rules)
  - src/docpilot/agent/questions.py    # shared seed question set (§3.8)
  - tests/test_agent_gate.py
  - tests/test_agent_judge.py          # verdict parsing, reformulated_query field
  - tests/test_agent_types.py
- guards:
  - Gate makes ZERO LLM calls; classify is pure/deterministic.
  - Unit tests hermetic — no live API/DB; judge tested with a stubbed Generator.
  - Verdict parsing tolerant of prompt-drift (JSON with regex/YAML fallback); loop never crashes.
  - Never invents citations or revisions outside the structured fields.

### AGENT G — LangGraph wiring + pipeline + CLI (depends on F COMPLETE)
- files_owned:
  - src/docpilot/agent/graph.py           # builds the StateGraph from injected components
  - src/docpilot/agent/pipeline_agentic.py# agentic_ask(...) orchestrator → AgentResult
  - src/docpilot/cli.py                   # ask --strategy auto|direct|agentic; --json "trace"
  - tests/test_agent_graph.py             # routing + budget with injected fakes
  - tests/test_agent_pipeline.py          # E2E via CLI with injected fakes
- guards:
  - LangGraph code contains NO vendor calls; all LLM/DB work behind injected interfaces.
  - Budget hard-enforced at graph level (max AGENT_MAX_RETRIES judge calls).
  - Trace always produced; --json output stays backward-compatible (additive "trace" key).

### COORDINATOR — shared, cross-cutting
- deps.md: declare `langgraph` (AGENT A runs uv sync; pyproject/uv.lock single-owner).
- conftest.py shared agent fixtures; seed set; PLAN §3 + SPEC §4 final text.
- Manual QA on the seed set; milestone commit/tag when exit criteria are met.

## 3.3 interfaces (concise)

- QueryClassifier(ABC).classify(question) -> GateDecision  # direct | agentic
- SufficiencyJudge(ABC).judge(context, sources, question) -> Judgment
  # Judgment{verdict, reason, reformulated_query|None}  — structured, single LLM call
- Agent(ABC).run(question, *, retriever, generator, judge, citation_engine,
  strategy="auto", top_k=None) -> AgentResult
  # AgentResult = AskResult + trace; "direct" strategy == classic ask()
- AgentResult embeds the existing AskResult; "auto" = gate decides.

## 3.4 trace & inspectability

Every LoopTraceStep: turn # → query_used, verdict, reason, reformulated_query,
retrieved_count + top scores, step label, step latency ms.
- stderr DEBUG logging (same stream discipline as Phase 1); --json "trace" key.
- Latency fields exist for inspectability only — no aggregates/comparisons here (§10.2).

## 3.5 config (.env, new keys)

| Key | Default | Purpose |
|-----|---------|---------|
| AGENT_MAX_RETRIES | 2 | Max judge/reformulate iterations |
| AGENT_DEFAULT_STRATEGY | auto | CLI overridable via --strategy |
| AGENT_GATE_LONG_THRESHOLD | 18 | Word-count gate trigger |
| AGENT_JUDGE_MODEL | (empty → GROQ_MODEL) | Optional separate judge model |

## 3.6 test strategy

- Hermetic: gate (pure), judge (stubbed Generator), graph routing, budget, CLI
  (injected fakes). Integration variants marked `integration`, skip-if-unreachable.
- Regression: fast path — simple questions produce IDENTICAL output to Phase 1
  (existing E2E + manual spot check).

## 3.7 exit_criteria

- Gate correct on the seed set (multi-hop → agentic; simple → direct).
- Loop answers multi-hop seed questions with correct [N] citations + footer.
- Budget enforced (≤ AGENT_MAX_RETRIES); refuse after budget.
- "I don't know" (SPEC §3.9) when evidence stays insufficient.
- Fast path regression: simple questions identical to Phase 1.
- Trace per query in --debug (stderr) and --json ("trace").
- All behind Agent interface; LangGraph code no vendor calls; our tracing only.
- CLI surface exactly ask --strategy auto|direct|agentic.
- Phase 2 claims NO performance numbers / no Phase1-vs-2 comparison.
- Full test suite green; agent tests hermetic.

## 3.8 seed_question_set (dev + Phase 2 manual QA; NOT the Phase 4 benchmark)

- multi_hop: >
    1. Combine path, query, and body parameters in one endpoint — validation rules per kind?
    2. Do dependencies interact with path operations — one dependency validating params and
       helping produce the response, shared state passing?
    3. Exception raised inside a dependency — interaction with exception handlers/middleware?
    4. Same Pydantic model for request-body validation AND response_model — differences?
    5. Background tasks vs yield-dependencies — when does cleanup really run?
    6. Annotated[...] = Depends(...) vs legacy = Depends(...) — difference, mixable?
    7. WebSocket endpoint + HTTP route sharing one auth dependency — wiring?
    8. OAuth2 security scopes + custom dependency to restrict routes?
- simple (must stay on fast path): 9. "How do I install FastAPI?" 10. "What is a query parameter in FastAPI?"
- refuse: 11. Auto-cache DB queries without extra code (uncovered → gate-classified
  simple, no signal fires → direct fast-path refuse; loop → budget → refuse is
  live-verified separately by seeds 4/6 and the forced-agentic edge probes — §3.11).
  12. Live OAuth token-expiration bug status (live state → Phase 3 territory → refuse in Phase 2).

## 3.9 execution_order

1. Cut phase-1-complete (ingest + idempotency + English QA + §2.5/§3.18 statuses).
2. deps.md ← langgraph; AGENT A uv sync.
3. AGENT F (contract + gate/judge/types/prompts/questions + tests).
4. AGENT G (graph + pipeline + CLI --strategy + trace + tests).
5. Coordinator: fold SPEC §4 final text (done with this update), manual seed QA,
   flip §3 status to COMPLETE + tag phase-2 when §3.7 met.
6. Measurable improvement claims deferred to Phase 4 — never asserted here.

## 3.10 resolved_decisions (user sign-off 2026-09-05)

1. Reformulation folded into judge; reformulated_query stays its own structured field
   (splittable later without redesigning the call site).
2. Per-step latency in trace only; no aggregates/comparisons/claims in Phase 2.
3. Our LoopTraceStep only; no LangGraph/LangSmith tracing.
4. ask --strategy auto|direct|agentic only; no --agentic alias.

## 3.11 verification (live seed QA — milestone gate for the `phase-2` tag)

Method: `.venv/bin/python -m docpilot ask "<seed>" --debug` against the live stack
(real Groq `gpt-oss-20b`, local BGE-small, pgvector, LangGraph 1.2.11), default
`auto` strategy, `RETRIEVAL_LANGUAGE=en`, `AGENT_MAX_RETRIES=2`,
`AGENT_LOOP_TOP_K=8`. Full stdout/stderr artifacts retained
(`/tmp/opencode/reqa/`).

Seed results (12/12 live):
- multi_hop seeds 1–8 → **all route agentic** (`Gate decision: agentic → agentic
  loop engaged`). Seeds 3/7/8 now fire the standalone `multi_concept` signal
  (≥3 distinct concepts — `exception,dependency,middleware` /
  `websocket,endpoint,auth,dependency` / `oauth2,security,scopes`), fixing their
  mis-routing to direct in the first QA round (§3.7 criterion 1).
- Loop answers with correct `[N]` citations + footer: seeds 1, 2, 3, 5, 7, 8 (6/8),
  judge `sufficient` on attempt 1; every cited Source under `en/`.
- Seeds 4, 6: loop engaged, judge `insufficient` on both permitted rounds →
  budget exhausted → verbatim §3.9 refusal. Judge reasons (quoted): 4 — "the
  retrieved chunks do not contain explicit evidence that the same Pydantic model
  can be used for both the request body and the response_model"; 6 — "explain the
  difference between Annotated and legacy Depends syntax but do not provide
  evidence about whether they can be mixed". These are spec-correct honest
  refusals (SPEC §3.9); answer-coverage quality is Phase 4 (Evaluation) territory.
- simple seeds 9, 10 → direct fast path; guardrail held (cross-A: byte-identical
  stdout vs `--strategy direct`).
- refuse seeds 11, 12 → verbatim §3.9 refusal. Loop→budget→refuse is live-proven
  by seeds 4/6 and the forced-agentic probes; seed 11's fast-path refusal is the
  gate correctly classifying a structurally simple uncovered question.

Cross-checks:
- A. `diff` seed-9 default stdout vs `--strategy direct` → **empty** (byte-identical).
- B. `--json` (seed 5): Phase-1 keys intact + additive `strategy|direct|refused|
  trace`; trace `['gate','search','judge','answer']`, trace[0]
  `{'step':'gate','decision':'agentic'}`, `latency_ms` ints, valid JSON, no
  LangGraph/LangSmith objects leaked.
- C. judge calls ≤ 2 on every loop run; `Agent refuse node: budget exhausted
  (attempts=2)` after the two `insufficient` rounds.
- D. `--strategy agentic` on simple seed 10 → `forced-agentic` + answered;
  `--strategy direct` on multi-hop seed 7 → `forced-direct`, fast path only.

Edge QA (2026-09-06, disjoint probes): per-turn `Agent retrieve: turn=N top_k=8
language=en` DEBUG lines in the loop; degenerate inputs (empty / whitespace /
uppercase / code-block-mention) exit 0 with no tracebacks; refuse-path `--json`
is plain-serialisable.

Suite: **193 passed (190 hermetic + 3 live pgvector integration)**.

## 4. phase_3: "GitHub Tooling" — implemented 2026-09-07, live QA passed 2026-09-08
- status: COMPLETE — implementation + hermetic suite (236 passed) + live QA
  (fastapi/fastapi, 2026-09-08) all green; tagged `phase-3` (SPEC §5.5).
- name_note: >
    Originally "MCP / GitHub Tooling". Locked decision (2026-09-07): plain
    GitHub REST behind the `Tool` interface — NO MCP protocol/SDK (SPEC §5.1).
- summary: >
    One meaningful external-data tool — GitHub — reached only when static docs
    are demonstrably insufficient (live issue state, repo state, recent
    commits). The judge's existing structured output carries the tool decision
    (`needs_tool` + `tool_request`) — no extra LLM call.
- guardrail: >
    The tool solves a real problem (docs can't answer this), not a checkbox.
    A `needs_tool` request reaches GitHub only within budget; without a wired
    tool the loop stays byte-identical Phase 2.

### 4.1 locked_decisions (user sign-off 2026-09-07)
1. Plain GitHub REST behind `Tool` — NO MCP protocol/SDK.
2. Tool decision rides the judge's structured output; never a separate LLM call.
3. `needs_tool` only with verdict insufficient + only for live-state gaps; doc
   gaps retry retrieval (`reformulated_query`) instead; `needs_tool=true` ⇒
   `reformulated_query=null`.
4. One `tool_call` per question, within budget (budget trumps the tool on the
   last round); tool never loops back to the judge; success → answer, failure →
   verbatim §3.9 refusal.
5. No PAT ⇒ judge told `TOOLS AVAILABLE: no`; loop Phase 2-identical; a
   defensive `needs_tool` degrades to plain insufficient.
6. Tool evidence cited: `LIVE GITHUB EVIDENCE` `[k+1…]` context section +
   continuing `SourceRef`s (`github:{owner}/{repo}#{n}` / `@{sha}`) in the footer.

### 4.2 files_owned
- AGENT T1 (tools, commit 2c16c7f): `src/docpilot/tools/{__init__,base,github}.py`,
  `tests/test_tools_github.py`, `config.py` GITHUB_* keys.
- AGENT T2 (wiring, commit 1149087): `agent/{types,judge,prompts,graph,
  pipeline_agentic}.py`, `cli.py` injection seam + "tool", agent tests.
- Config: `GITHUB_PAT` / `GITHUB_API_BASE` / `GITHUB_OWNER` / `GITHUB_REPO`
  (SPEC §5.3). `.env.example` + SPEC §3.16 / README / PLAN env refs (commit 85c0fb9).

### 4.3 test_strategy
- Hermetic: tools (injected `request_fn`, zero network), graph routing (StubTool;
  budget-vs-tool ordering; tool failure → refuse; no-tool degradation), judge
  `needs_tool`/`tool_request` tolerant parsing, pipeline tool injection,
  guardrail assertions (no `tool_call` step on doc-answerable questions).
- Live GitHub QA passed 2026-09-08 (fastapi/fastapi, GITHUB_PAT via `~/.bashrc`,
  owner/repo injected at runtime) — checklist in §4.5.

### 4.4 exit_criteria (SPEC §5.5)
- [x] `Tool` + `GitHubTool` (no MCP), PAT-gated
- [x] Judge `needs_tool` signal, no extra LLM call
- [x] `tool_call` node: within budget, never loops back; error → verbatim refuse
- [x] No-PAT → Phase 2-identical loop
- [x] Tool evidence cited (context + footer)
- [x] Suite 236 passed (233 hermetic + 3 live)
- [x] Live QA (fastapi/fastapi 2026-09-08): tool-fires (issues + commits),
      tempt-the-tool guardrail, refusal regression — all 5 probes pass
- [x] `phase-3` tag + native `GitHubTool` demo (tag cut after live QA passed)

### 4.5 verification (live QA passed 2026-09-08 — fastapi/fastapi)
- 236 tests green (233 hermetic + 3 live pgvector); Phase 1/2 regression
  protected (direct fast path untouched, `tool=None` loop byte-identical).
- Live probes, all PASS:
  (a) open-issue OAuth2 → `github.search_issues` fires (`is:issue is:open`),
      cited answer `github:fastapi/fastapi#10`;
  (b) latest commit → `github.get_commits` (no `ref` → default branch),
      cited `github:fastapi/fastapi@50113da` incl. author;
  (c) tempt-the-tool guardrail — doc-answerable question answers with 0
      `tool_call` (gate → direct fast path);
  (d) out-of-domain → verbatim §3.9 refusal, 0 `tool_call`;
  (e) judge parse-fallback counter — 0 fallback lines across all QA logs.
- Live QA surfaced 3 fixes (fix(agent) 0f13951 + follow-ups): judge search
  syntax (`in:issue`→`is:issue`), branch assumption (`ref=main` → omit-ref
  default), commit-author evidence completeness. Judge parse-fallback counter
  wired (SPEC §6.3) — `_record_parse_fallback` INFO lines, hermetic tests.

## 5. phase_4: "Evaluation"
- status: IN PROGRESS (2026-09-08) — slices 1–2 complete (judge two-prompt A/B,
  tool-necessity tri-class). Slice 3 (classic-vs-agentic §6.1 comparison) in
  progress: classic half complete + committed (stamp 190539, post-fix scoring),
  agentic half blocked on a fresh-quota Groq key (daily TPD exhausted on the
  current org); resume + L2 quota insurance agreed (SPEC §6.5). Then:
  false-refusal decomposition, parse-fallback flip-condition check.
- summary: >
    Build a benchmark dataset and track: retrieval quality, answer correctness,
    citation correctness, groundedness/hallucination rate, "I don't know"
    accuracy, latency, retrieval/tool-call count.
- required_comparison: >
    Classic RAG (Phase 1) vs. agentic RAG (Phase 2) on the same benchmark.
    Claims of improvement must be backed by this data, not asserted.
- client_required_scope: >
    Locked 2026-09-07 (SPEC §6.2): false-refusal rate + 3-way decomposition
    (retrieval / judge / routing); judge-specific calibration (labeled triples +
    adversarial paraphrases + parse-failure rate); tool-necessity eval
    (docs-answerable / live-state-answerable / neither → false pos/neg rates);
    judge two-prompt A/B as the first slice.

### 5.1 slice_1 judge two-prompt A/B (2026-09-08 — complete)
- dataset: `src/docpilot/eval/dataset/judge_triples.json` — 24 labeled verdict
  triples (real FastAPI-corpus contexts; 15 sufficient / 9 insufficient;
  categories sufficient-direct / sufficient-partial / insufficient-missing /
  adversarial ×6 each). Root `data/` is git-ignored, so the SPEC-mandated
  committed dataset ships in-package.
- harness: `src/docpilot/eval/judge_ab.py` + `python -m docpilot.eval`; judge
  gains optional `system_prompt` (prompt-B pressure test of the
  `SufficiencyJudge` interface). Prompt B = decision-procedure variant,
  contract-compatible (same JSON keys, lean-sufficient policy, live-validated
  Phase 3 tool rules). 20 new hermetic tests; suite 256 passed.
- result (live Groq, temp 0, symmetric one-pass): **A and B both 1.000
  verdict accuracy, 1.000 adversarial accuracy, 0.000 parse-failure rate —
  tie on every metric, zero parse fallbacks** (report
  `src/docpilot/eval/reports/judge_ab_20260908_174922.json`).
- decision: production judge keeps **Prompt A** (the live-QA-validated
  prompt). Nothing to adopt from B on this seed; the tie is a ceiling effect
  on a 24-triple set — enrichment with harder adversarial + the tool tri-class
  set is the next pressure test, not a prompt swap.
- known_open_questions: >
    Phase 2 seeds 4/6 refusals may be false refusals (retrieval failed, judge
    honest) — resolved via the decomposition eval, not ad hoc retries.
    Parse-fallback default (sufficient) stays pending Phase 4 flip-condition
    data (SPEC §6.3); slice 1 measured 0 parse fallbacks on the labeled set.

### 5.2 slice_2 tool-necessity tri-class (2026-09-08 — complete)
- dataset: `src/docpilot/eval/dataset/tool_necessity.json` — 15 gold tri-class
  questions (5 docs-answerable / 5 live-state-answerable / 5 neither), each with
  gold verdict, gold needs_tool and (for live-state) the expected tool action;
  real FastAPI-corpus contexts; strict schema loader (`tool_necessity.py`).
- harness: `src/docpilot/eval/tool_necessity.py` +
  `python -m docpilot.eval tool-necessity`; judge run with
  `tools_available=True` (decision-time measurement only — the actual GitHub
  call is exercised by the classic-vs-agentic comparison). Metric block:
  false-positive rate (tool fired on docs-answerable), false-negative rate (no
  tool on live-state-answerable), neither_fire_rate, verdict accuracy,
  tool-request validity, expected-action match; per-row parse-fallback
  attribution. 28 hermetic tests; suite 284 passed.
- result (live Groq, temp 0, prompt A, 15 questions; report
  `src/docpilot/eval/reports/tool_necessity_20260908_181226.json`):
  | metric | value |
  | --- | --- |
  | verdict accuracy | 1.000 |
  | false-positive rate (docs-answerable → tool) | 0.000 |
  | false-negative rate (live-state → no tool) | 0.000 |
  | neither_fire_rate | 0.200 |
  | tool_request_valid_rate | 1.000 |
  | tool_action_match_rate | 0.667 |
- reading: on this seed the judge's *necessity* signal is calibrated — it never
  fired on docs-answerable (0 FP) and never missed a live-state trigger (0 FN).
  Two flagged rows:
  - tl03 ("how many open issues?") — fired correctly (`needs_tool` true) but
    chose `search_issues` over gold `list_issues`; both actions can answer a
    count → benign action-choice alternate, not a necessity failure; gold kept
    strict (tool_action_match 4/6 on the error side of that strictness).
  - tn03 ("maintainer hiring plans?") — genuine neither-class false fire
    (`search_issues` on a personal/org question the repo-scoped tool can't
    answer): needs_tool accuracy 14/15. Seed-level signal the judge reaches for
    the tool on open-ended org questions; re-check in the benchmark +
    decomposition slices.
- exit_criteria_progress: SPEC §6.4 "tool-necessity report: FP/FN rates on the
  tri-class set" → **CLOSED**; "benchmark dataset committed (labeled verdict
  triples + adversarial + tri-class gold)" → committed across slices 1–2;
  "judge calibration report" → CLOSED (slice 1). SPEC §6.4 updated to match
  (2026-09-08). Remaining: §6.1 classic-vs-agentic comparison, false-refusal
  decomposition, parse-fallback flip-condition check.

### 5.3 slice_3 classic-vs-agentic §6.1 comparison (2026-09-08 — in progress)
- dataset: `src/docpilot/eval/dataset/benchmark.json` — 15 questions (8
  docs-answerable / 4 live-state-answerable / 3 neither). Gold key facts are
  short paraphrase-robust fragments; gold sources are stored corpus-relative
  paths (post-fix v2, commit 08086dd).
- harness: `src/docpilot/eval/benchmark.py` — `run_pipeline`, injectable
  `GroundingChecker` (NLI-style proxy over `Generator.generate`, callable
  form), per-half sidecar checkpointing, `--pipeline {classic,agentic}` /
  `--resume STAMP` / `--merge DIR STAMP`, one-table comparison builder.
  334 hermetic tests.
- scoring-fix root cause: the first live run scored 0.00 across ALL doc rows;
  the sidecar audit showed retrieval hit the correct files and every answer
  was correct. Two gold-data defects, not system failures: (1) gold paths
  carried a `docs/` filesystem prefix the vector store never emits
  (`en/docs/...`), zeroing recall + citation-gold; (2) verbatim corpus
  sentences as gold facts — the generator paraphrases by design, so sentence
  containment penalized correct answers. Fixed to stored paths + fragment
  facts; offline re-scoring of the completed run validated the fix (recall@k
  1.0000 for both pipelines) before any rerun.
- live runs: stamp `20260908_184441` completed BOTH pipelines pre-fix
  (superseded scoring); stamp `20260908_190539` reran post-fix — classic
  complete + committed (`benchmark_20260908_190539_classic.json`), agentic
  crashed mid-run on daily quota (TPD limit 200000, Used 199443).
- classic half (stamp 190539, post-fix scoring):
  | metric | value |
  | --- | --- |
  | answer_correctness | 0.8000 |
  | retrieval recall@k (docs) | 1.0000 |
  | citation validity | 1.0000 |
  | citation gold accuracy (docs) | 0.5833 |
  | refusal accuracy (I-don't-know) | 1.0000 |
  | groundedness (NLI proxy) | 0.7000 |
  | avg latency | 9046 ms |
  | retrieval calls / tool calls | 1.0000 / 0 |
- blocked on: fresh-quota Groq key (decision, user): swap org → resume agentic
  half (`--pipeline agentic --resume 20260908_190539`) → auto-merge under the
  same stamp → §6.1 one-table comparison. Fallback if quota never materializes
  (not preferred): labeled two-run table using the completed pre-fix agentic
  sidecar, documented caveat.
- agreed insurance (user sign-off → SPEC §6.5): generator honors server
  `retry-after` on 429 (self-heals TPM bursts) + `--probe` headroom gate
  before launch. L3 per-question checkpointing considered and declined for now.
- follow-up: token accounting — generator discards Groq `usage`; record
  per-call tokens in reports so quota economics are inspectable.
- exit_criteria_progress: §6.4 "classic-vs-agentic comparison" → in progress
  (classic half + table machinery done; agentic half pending quota). After
  this slice: false-refusal decomposition, parse-fallback flip-condition
  check.

## 6. phase_5: "API + UI"
- status: PLANNED
- summary: >
    After core is working and evaluated. FastAPI backend (async introduced
    here), streaming, citations + retrieved-context debug panel in UI,
    SQLite session/chat history. Streamlit first; React/Next.js only if slack.
- guardrail: Frontend polish never delays or distorts the RAG/agent core.

## 7. phase_6: "Code Generation / Validation"
- status: PLANNED
- summary: >
    Documentation retrieval → generate code → validate against retrieved
    API/schema/examples → return code + sources. Only after core is reliable.

## 8. framework_extraction
- status: PLANNED (post-Phase 6)
- summary: >
    Extract reusable components (interfaces, orchestration, eval harness) into
    a standalone SDK. Only after DocPilot works end-to-end and is evaluated.

## 9. architectural_discipline
- interfaces:
  - DocumentLoader
  - Parser
  - Chunker
  - EmbeddingProvider
  - VectorStore
  - Retriever
  - Reranker
  - Tool
  - Agent
  - Generator
  - CitationEngine
  - Evaluator
- rule: >
    No hard-coded vendor/library calls in business logic (no bare groq.chat /
    pgvector query outside its interface impl). Keep interfaces clean so
    BGE/Qdrant/Groq/LangGraph can be swapped later.
- long_term: >
    DocPilot is the learning/validation project for a future framework, not the
    framework itself. "Easy by default, customizable when needed." Don't build
    the framework first.

## 10. conflict_resolution
- later_phase_feature_early: >
    Implement behind an interface if truly needed now, but state plainly that
    this pulls forward work from Phase N; ask whether intended.
- run_agent_on_every_query: >
    Push back — violates Phase 2 guardrail and the cost/latency argument.
- skip_evaluation_for_polish: >
    Implement what's asked, but note Phase 4 is still owed and nothing about
    "agentic RAG being better" is proven yet.
- unsure_of_phase: Ask before restructuring existing code to fit it.

## 11. git_workflow
- repo: https://github.com/Saif-Ali-109/DocPilot.git
- branch: main
- rule: >
    Keep the repo in sync with the work. Commit every task and every phase as
    it completes. The git history must mirror the build progression so each
    change is attributable and reviewable.
- commit_units:
  - per_task: >
      Commit at the end of each sub-agent task (one task may span multiple
      small commits if it is large). Each commit must be self-contained and
      leave the working tree in a passing/runnable state where practical.
  - per_phase: >
      When a phase's exit criteria are met, cut a phase milestone commit (tag
      the phase, e.g. phase-1-complete) before moving to the next phase.
- message_format: |
    Conventional Commits: <type>(<scope>): <subject>

    <body>

    types: feat, fix, refactor, docs, test, chore, build
- scope_examples: [ingestion, retrieval, generation, citations, cli, db, config, corpus, tests]
- commit_hygiene:
  - Never commit secrets or .env (git-ignored).
  - No raw vendor keys or connection passwords in the tree at any point.
  - Push to origin/main after each meaningful commit batch.
- status_file: >
    PLAN.md §2.5 (exit criteria) drives phase completion. A phase-milestone
    commit is only created when its checklist is satisfied.
