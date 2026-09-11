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
- current_phase: 5 (IN PROGRESS — API + UI; SPEC §7 + amendment 2026-09-09; §6 task breakdown)
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

## 5. phase_4: "Evaluation" — COMPLETE 2026-09-09 (milestone `phase-4`)
- status: COMPLETE (milestone `phase-4` cut 2026-09-09). Slices: (1) judge
  two-prompt A/B, (2) tool-necessity tri-class, (3) classic-vs-agentic §6.1
  comparison (rescored 2026-09-09 — see §5.3), (4) false-refusal 3-way
  decomposition, (5) judge calibration + parse-fallback flip-condition check
  (KEEP, recorded). All §6.4 exit criteria `[x]` in SPEC.
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

### 5.3 slice_3 classic-vs-agentic §6.1 comparison (2026-09-08/09 — DONE ✓)
- dataset: `src/docpilot/eval/dataset/benchmark.json` — 15 questions (8
  docs-answerable / 4 live-state-answerable / 3 neither). Gold key facts are
  short paraphrase-robust fragments (v3 — 2026-09-09: bd02/bd06 facts were
  still corpus-strict, e.g. `type annotation` penalizing the correct
  paraphrase `type hint`; replaced with invariants both live pipelines
  satisfy); gold sources are stored corpus-relative paths (commit 08086dd).
- harness: `src/docpilot/eval/benchmark.py` — `run_pipeline`, injectable
  `GroundingChecker` (NLI-style proxy over `Generator.generate`, callable
  form), per-half sidecar checkpointing, `--pipeline {classic,agentic}` /
  `--resume STAMP` / `--merge DIR STAMP`, one-table comparison builder.
  364 hermetic tests.
- scoring-fix root cause: the first live run scored 0.00 across ALL doc rows;
  the sidecar audit showed retrieval hit the correct files and every answer
  was correct. Two gold-data defects, not system failures: (1) gold paths
  carried a `docs/` filesystem prefix the vector store never emits
  (`en/docs/...`); (2) verbatim corpus sentences as gold facts — the generator
  paraphrases by design. Fixed to stored paths + fragment facts (v2), then
  v3 (see above).
- live runs: stamp `20260908_184441` completed BOTH pipelines pre-fix
  (superseded); stamp `20260908_190539` reran post-fix — classic complete on
  2026-09-08, agentic resumed in later windows (2026-09-09) and completed.
- **Final §6.1 one-table comparison** (`benchmark_20260908_190539.json`,
  rescored 2026-09-09 under gold-v3 + §3.9 refusal-text fallback + bd05
  regeneration — see notes below):

  | metric | classic | agentic | winner |
  | --- | --- | --- | --- |
  | answer_correctness | 0.8000 | **0.9667** | agentic |
  | retrieval recall@k (docs) | 1.0000 | 1.0000 | tie |
  | citation validity | 1.0000 | 1.0000 | tie |
  | citation gold accuracy (docs) | **0.5833** | 0.5312 | classic |
  | refusal accuracy (I-don't-know) | 1.0000 | 1.0000 | tie |
  | groundedness (NLI proxy) | **0.7000** | 0.5000 | classic |
  | avg latency | **9046 ms** | 31862 ms | classic |
  | avg retrieval calls | **1.0000** | 1.1333 | classic |
  | avg tool calls | **0.0000** | 0.2667 | classic |

  Tally **5 classic / 1 agentic / 3 ties** — neither pipeline dominates.
- **Reading (data-backed, the phase's honest verdict):** the agent layer is
  NOT net-positive on this dataset, and its one clear win is exactly what it
  was built for — live-state coverage. Agentic fires the GitHub tool on 3/4
  live questions and scores those 1.0 (classic has no tool and structurally
  cannot; it refuses and gets 0.0/0.5), pulling agentic correctness to 0.9667
  vs 0.8000. But agentic pays for it: 3.5× latency (31.9s vs 9.0s avg), a
  weaker groundedness audit (0.50 vs 0.70 — its live-evidence answers audit
  as ungrounded at the NLI proxy), slightly worse citation-gold precision,
  and needless tool engagement on one HR question (bn03, see observations).
  Simple doc questions cost the loop 1.5–2× the latency with no quality
  margin — the strong argument for keeping the fast/cheap path default for
  doc-answerable queries.
- **Artifacts corrected before scoring (all measurement bugs, not system
  behavior — full provenance in the report notes):**
  1. gold v3 (bd02/bd06) applied to BOTH pipelines; classic control
     bit-identical apart from the note.
  2. §3.9 refusal-text fallback in `run_pipeline`: bn03 emitted the verbatim
     refusal sentence (generator self-refused on the answer path after a
     needless tool call) but the flag was False — counted as refused now.
  3. bd05's completion was empty (model-side glitch); regenerated live under
     the empty-completion retry fix (new answer 1.0 facts, grounded False).
- live-run log (2026-09-09, multi-window resume of the agentic half):
  - probe Sep-09: functional gate exit 0 (per-minute 5878/8000) — overnight
    24h-window slide had freed the org's daily bucket (rolling windows, no
    midnight reset).
  - resume #2 (evening 09-08) had died on the same TPD wall (Used 199916–
    199964/200000, ~36–84 tokens headroom); retry-after cap proved: server
    waits "19m6s / 7m52s / 9m7s" → all slept 10.0s, fail-fast works.
  - resume #3 (09-09): crashed on a NEW non-quota failure — Groq 400
    `tool_use_failed`: the hosted model emitted `repo_browser.open_file`
    despite no tools declared. GroundingChecker's plain NLI prompt was the
    victim. Fixed (fe2c238): retry exactly once with a plain-prose guard
    (applies to judge/generator/grounding alike).
  - resume #4: **exit 0** — agentic half complete under stamp 190539.
  - bd05 regen + offline rescore (no quota beyond 1 question + 1 NLI audit).
- **Slices completed (2026-09-09, both report JSONs committed):**
  - False-refusal 3-way decomposition (`false_refusals_20260909_091251.json`):
    0 false refusals agentic; classic 2/15 (bl01, bl03 — capability routing:
    live-state questions reach the no-tool fast path; retrieval genuinely
    insufficient for docs-only); 0 judge-caused; all HR/neither refusals
    honest. Observation: bn03's needless tool engagement cross-validates the
    tool-necessity neither_fire_rate 0.2.
  - Parse-fallback flip-condition check (`parse_fallback_flip_20260909_
    091251.json`): **KEEP** the defensive `sufficient` default — 0 live
    parse-fallbacks (agentic half), 0.0 calibration parse-failure rate (24
    triples, both prompts), no generator under-refusal observed (bn03
    self-refused with the verbatim §3.9 sentence).
- **Phase 5 hardening AFTER-run:** the §6.1 locked gate against before =
  `20260908_190539`. Same 15 questions, both pipelines, post-hardening core
  (streaming, source-kind + SYSTEM_PROMPT rule 9, AGENT_LOOP_TOP_K 8→5,
  judge-skip default disabled).
  - First attempt (2026-09-09, stamp `20260909_162036`): classic sidecar ran
    on the then-live dirty corpus (16,535 rows, 1,216 twin rows); the agentic
    half was TPD-walled (Used 198698–199940/200000) and the corpus was later
    deduped (16,535→15,319) *between* the halves → asymmetric, kept as
    historical evidence only, NOT used for the gate table.
  - **Fresh symmetric after-run (2026-09-10, stamp `20260910_201739`):** both
    halves re-run on the deduped corpus + hardened search (top_k×2 dedupe
    window) — apples-to-apples classic-vs-agentic. Classic 20:20, agentic
    resumed under the same stamp 20:27 (exit 0); combined report + comparison
    auto-written (`benchmark_20260910_201739.json`). This stamp is the gate
    source. **Before-table cit-gold columns corrected below** (they were
    transposed in the original draft: before classic 0.5833 / before agentic
    0.5312).
  - **before/after (stamp 20260910_201739):**

    | metric | before classic | after classic | before agentic | after agentic |
    | --- | --- | --- | --- | --- |
    | answer_correctness | 0.8000 | 0.7333 | 0.9667 | 0.9333 |
    | retrieval recall@k (docs) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
    | citation validity | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
    | citation gold (docs) | 0.5833 | **0.6333** | 0.5312 | **0.6429** |
    | refusal accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
    | groundedness | 0.7000 | **0.7778** | 0.5000 | **0.7273** |
    | avg latency | 9046 ms | 9797 ms | 31862 ms | **22439 ms** |
    | avg retrieval calls | 1.0000 | 1.0000 | 1.1333 | 1.1333 |
    | avg tool calls | 0.0000 | 0.0000 | 0.2667 | 0.3333 |

  - reading (fresh symmetric run):
    - classic correctness 0.8→0.7333: single-row delta bd02 (1.0→0.5, now
      ungrounded) — the documented source-kind miss (§6.1 #6): path-params.md
      retrieved rank-1 but the generator cited body/query-params markers; the
      before-run was luckier. bl02 improved (hallucinated "PR #13871" →
      verbatim §3.9 honest refusal), bd08 grounded False→True, bl01/bl03
      honest refusals unchanged.
    - agentic correctness 0.9667→0.9333: single-row delta bl03 (live-state) —
      tool fired with issue evidence (`fastapi#10370`) but the generator
      self-refused (verbatim §3.9) instead of answering "3 open issues"; the
      before-run answered that from a single issue citation. Safe-direction
      single-run noise. Groundedness improved 0.50→0.7273 (bd05, bd08, bl04
      flip to grounded), latency 31.9s→22.4s, cit-gold 0.5312→0.6429.
    - comparison (9 decided metrics): agentic wins answer_correctness (+0.2
      over classic after) and citation-gold; classic wins groundedness
      (+0.05), latency (2.3× faster), retrieval/tool counts; ties on recall /
      citation validity / refusal accuracy. The agentic correctness premium
      over classic holds on the clean corpus.
    - **judge-skip: stays DISABLED (`AGENT_JUDGE_SKIP_MIN_SCORE` = 0.0)** —
      the judge's tool-necessity + sufficiency decisions drove every
      live-state row (bl01/bl02/bl04 tools, bn03 engagement, bl03 honest
      refusal); nothing in the after-run data supports bypassing the fast-path
      judge call.
- follow-up (open, Phase 5 scope): token accounting (record per-call `usage`);
  latency/citation-gold hardening per SPEC §7 amendment 2026-09-09 — top_k
  8→4–5, fast-path judge skip for clearly-doc questions, source-kind citation
  preference (SourceRef metadata + generator prompt), model endpoint/tier
  knob; each change re-validated on the 15-question benchmark (before/after
  gate) except pure-UI streaming.
- exit_criteria_progress: §6.4 — all four eval criteria now `[x]` in SPEC
  (comparison, false-refusal decomposition, parse-fallback flip check, plus
  the earlier judge-calibration + tool-necessity reports). Phase-4 milestone
  tagged `phase-4`.

## 6. phase_5: "API + UI" — COMPLETE (started 2026-09-09, gate closed 2026-09-10)

- status: COMPLETE — implementation on the Phase 4-closed core; the §7
  latency/citation hardening (SPEC amendment 2026-09-09) shipped with a locked
  before/after benchmark gate (before = stamp 20260908_190539, committed).
  Gate closed on the fresh symmetric after-run (stamp 20260910_201739, both
  halves on the deduped corpus); all §6.3 exit criteria `[x]`; milestone
  tag `phase-5`.
- summary: >
    Async FastAPI backend (first async code in the project), SSE token
    streaming, source citations surfaced in the UI, a retrieved-context debug
    panel (chunks + scores + judge/tool trace steps + latency), SQLite
    session/chat history, Chainlit UI. UI/API never reshape the RAG/agent core
    (guardrail §7).
- locked_decisions:
  - **Async boundary.** Core stays synchronous (Groq client, psycopg, BGE).
    FastAPI endpoints are async; every blocking core call runs in a worker
    thread (`asyncio.to_thread` / `run_in_threadpool`). Progressive events are
    bridged thread → SSE via an `asyncio.Queue`
    (`loop.call_soon_threadsafe`), consumed by a `StreamingResponse` async
    generator. Streaming is UX-only w.r.t. the §7 latency metric.
  - **Streaming.** `Generator` gains `generate_stream(prompt) -> Iterator[str]`
    and `generate_answer_stream(...)`; `GroqGenerator` implements them
    (`stream=True`, same transient-retry frame, buffered — no delta is
    yielded until the request is past the retry window). Direct/fast path
    streams tokens through the API service; the agentic loop's answer node
    streams via the new `emit` hook when a caller passes one (default
    `None` → byte-identical Phase 2/4 behaviour, CLI + eval untouched).
  - **Progress hook.** `agentic_ask(..., emit=None)` / `build_graph(..., emit=)`
    take an optional event callback (dict events: `step` mirrors each
    LoopTraceStep live; `token` streams answer deltas). Default `None` =
    zero behaviour change; the CLI passes nothing.
  - **SSE protocol** (type-tagged JSON `data:` lines): `gate` / `search`
    (chunks + scores — debug panel) / `judge` / `tool_call` / `token` /
    `answer` (text + footer + sources incl. `kind`) / `error` / `done`
    (trace + latency + `usage`).
  - **Sessions.** SQLite via stdlib `sqlite3` (no ORM). Tables `sessions` +
    `messages` (question/answer/sources/trace/latency/refused timestamps).
    HTTP API owns this record; Chainlit keeps its built-in persistence for UI
    chat history (SPEC §7 amendment). DB file under `data/` (git-ignored).
  - **Hardening** (SPEC §7 amendment, each behind the after-benchmark gate):
    1. `AGENT_LOOP_TOP_K` default 8 → **5** (judge + answer prompt shrink);
       `RETRIEVAL_TOP_K` stays 5.
    2. Fast-path judge skip: `AGENT_JUDGE_SKIP_MIN_SCORE` (float, default 0.0
       = **disabled** until the after-run justifies enabling). When > 0, the
       retrieve node marks `skip_judge` when the top retrieval score clears
       the threshold and routes straight to answer (judge LLM call saved).
    3. Source-kind citations: `SourceRef.kind` (contract addition) derived
       from `source_file` — `tutorial` / `advanced` / `how-to` / `reference`
       / `index` / `other` (github tool refs → `live`); `format_sources`
       shows it (prompt preference) + SYSTEM_PROMPT rule "prefer the most
       specific/authoritative file when several cover the same claim".
    4. Model knob: chat API accepts `model` per request (default `GROQ_MODEL`).
    5. Token accounting: `GroqGenerator` records `last_usage`
       (prompt/completion/total tokens) per call; surfaced in the `done`
       event. Eval-report integration stays a §5.3 follow-up.
    6. bd02 targeted look (both pipelines 0.0 citg): classic cited body.md /
       query-params.md, agentic python-types.md — the gold path-params.md was
       offered as [1] in both; source-kind preference is the fix lever.
- guardrail: Frontend polish never delays or distorts the RAG/agent core;
  the CLI and all Phase 1–4 behaviour stay byte-identical when no emit hook
  is passed; Phase 6 (code gen) stays locked out.

### 6.1 files & ownership

- **AGENT U — API + UI** (this phase's implementer):
  - `src/docpilot/api/__init__.py`, `api/app.py` (FastAPI), `api/service.py`
    (event orchestration), `api/sse.py` (protocol/format helpers),
    `api/store.py` (SQLite sessions/messages)
  - `src/docpilot/ui/__init__.py`, `ui/chainlit_app.py`, `ui/chainlit.md`
  - `tests/test_api_store.py`, `tests/test_api_chat.py`
- **Coordinator — additive core touchpoints** (contract bumps, none breaking):
  - `src/docpilot/core/models.py` — `SourceRef.kind` + `derive_source_kind()`
  - `src/docpilot/generation/generator.py` — `generate_stream`,
    `generate_answer_stream`, `last_usage`
  - `src/docpilot/generation/prompts.py` — kind in `format_sources` + rule 9
  - `src/docpilot/agent/types.py` — `AgentLoopState.skip_judge` field
  - `src/docpilot/agent/graph.py` — emit hook (answer node), judge-skip
    routing, `source_to_dict`/`dict_to_source` carry `kind`
  - `src/docpilot/agent/pipeline_agentic.py` — `agentic_ask(..., emit=None)`,
    pass judge-skip config through
  - `src/docpilot/pipeline_ask.py` — sources built with `derive_source_kind`
  - `src/docpilot/config.py` — `AGENT_LOOP_TOP_K` 5, `AGENT_JUDGE_SKIP_MIN_SCORE`,
    `DOCPILOT_DB_PATH`
  - `tests/test_generation_retry.py` (stream tests), `tests/test_agent_graph.py`
    (skip + emit)

### 6.2 deps

- `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `chainlit>=2.0` (+ uv.lock).
  Dev/test: httpx already present (TestClient).

### 6.3 exit_criteria (SPEC §7)

- [x] FastAPI async app runs (`uvicorn docpilot.api.app:app`); `/api/v1/health`
- [x] `POST /api/v1/chat` SSE-streams tokens + gate/search/judge/tool events;
      final `answer` event carries citations (sources + footer) and `done`
      carries trace + latency + usage
- [x] Debug-panel data per question: retrieved chunks + scores + trace steps
- [x] Sessions CRUD + messages in SQLite (`/api/v1/sessions*`)
- [x] Chainlit UI answers E2E with streaming + citation elements + steps
- [x] CLI + Phase 1–4 suite still green (byte-identical without emit hook)
- [x] Hardening after-run recorded (15 questions × both pipelines) with a
      before/after table in the report note; judge-skip default enabled only
      if the data supports it — verdict: **stays disabled (0.0)**; symmetric
      gate stamp `20260910_201739`, table in §6, report committed
- [x] README honest — Phase 5 scope only, no Phase 6 claims

### 6.4 verification (live)

- [x] functional probe (`bash -ic`, exit 0 + per-minute headroom) — 2026-09-10
- [x] uvicorn + curl SSE smoke (direct + agentic + refusal rows) — direct +
      refusal streamed 2026-09-10; refusal `answer` event carries the verbatim
      §3.9 sentence and `refused: true` (direct-path flag fixed 2026-09-10)
- [x] Chainlit interactive QA (fast/agentic/refuse + debug panel) — user
      validated in-browser 2026-09-09 (chat + debug panel screenshots)
- [x] after-benchmark run (classic quick; agentic checkpointed/resumed if TPD
      walls) → merged stamp `20260910_201739` + before/after §6.1 table in the
      report note (fresh symmetric run on the deduped corpus)
- [x] milestone: cut `phase-5` when §6.3 all `[x]`

## 6.5 pre_phase6_hardening (external-review findings, decided 2026-09-10)

- status: IMPLEMENTED (levers default OFF until benchmark-justified); benchmark
  record below.
- origin: external reviewer raised 7 findings on the phase-5 codebase; all 7
  verified by the coordinator (2026-09-10). Items 4 (single VectorStore) and 7
  (no API auth/rate-limit, SQLite-only sessions) are deferred *by design* —
  aspirational store-swap per AGENTS.md §interfaces; demo-scale API
  intentionality. Items 1, 2, 3, 5, 6 implemented as this §H batch.
- decision: the user chose to solve these *before* Phase 6 (not bank-as-debt).

### 6.5.1 changes

| # | Finding | Change | Default |
|---|---------|--------|---------|
| 1 | no `Reranker` interface | `src/docpilot/reranking/` — `Reranker` ABC + `BCEReranker` (BAAI/bge-reranker-base, CPU, lazy-load); `SimpleRetriever` fetches a `RERANK_CANDIDATES` (20) window and re-scores to top_k | `RERANK_ENABLED=0` |
| 2 | no hybrid/Bm25 | `retrieval/lexical.py` — `LexicalSearcher` ABC + `PostgresFTSSearcher` (GIN `chunks_content_fts` on `to_tsvector('english', content)`, OR-semantics `to_tsquery` + `ts_rank`, stopword-filtered); `retrieval/hybrid.py` — `HybridRetriever` fusing both halves via weighted RRF (k=60); wired in `_build_default_retriever` | `HYBRID_ENABLED=0` |
| 3 | dead gate knob | `HeuristicQueryClassifier(long_word_limit=None)` reads `config.AGENT_GATE_LONG_THRESHOLD`; config comment no longer says "NOT YET WIRED" | 18 (unchanged) |
| 5 | 15-question eval set | `eval/dataset/benchmark.json` expanded 15 → 30 (20 docs / 4 live / 6 neither): +12 docs-answerable exact-identifier/multi-hop rows (bd09–bd20), +3 adversarial refusals (bn04–bn06) | n/a |
| 6 | judge = 1 LLM call, no cross-check | `ScoreFloorBackstopJudge` wrapper in `agent/judge.py` — forces `insufficient` when top retrieval score < floor (or results empty) while preserving `needs_tool`/`tool_request`; wired in `_build_default_judge` | `AGENT_JUDGE_SCORE_FLOOR=0.0` |

Icon: levers ship default-OFF so the committed baseline (stamp `20260910_201739`,
levers-off) stays byte-identical behaviour; evidence gate below decides whether
to flip them on.

### 6.5.2 retrieval-level evidence (zero-LLM recall probe, language="en" to mirror `RETRIEVAL_LANGUAGE`)

Probe = direct retriever calls (no LLM, no quota); recall@5 / MRR@5 over the 20
docs-answerable rows, gold sources as offered-source recall. Baseline = Phase 1
cosine top-5 as committed.

| Config | recall@5 | MRR@5 |
|--------|---------|-------|
| **baseline** (levers off) | **0.900** (18/20) | **0.717** |
| both levers ON (rerank@20 + hybrid 1:1) | 0.850 (17/20) | 0.597 |

**Attribution + tuning pass** (one embed/search/rerank pass shared across all
configs, no LLM; bge "query:"/"passage:" prefix variant included):

| Config | recall@5 | MRR@5 |
|--------|---------|-------|
| rerank (bge, plain) | 0.850 | 0.632 |
| rerank (bge, `query:`/`passage:` prefixes) | 0.850 | 0.679 |
| hybrid 1:1 | 0.900 | 0.767 |
| **hybrid 2:1 (vector-heavy)** | **0.900** | **0.783** |
| both (rerank + hybrid) 1:1 | 0.850 | 0.597 |
| both 2:1 / both-fts3 / pfx-both-w2 | 0.850 | 0.603 / 0.629 / 0.704 |

**Verdict (evidence-backed):**
- **Reranker stays OFF.** Every rerank variant lowers recall (0.900→0.850) and
  MRR — even with the bge-required prefixes — and costs ~85 s/predict on this
  CPU (≈3 min per query end-to-end). The cross-encoder re-orders toward
  *related* topics, not gold sources (bd17 debug: behind-a-proxy.md,
  custom-response.md instead of header-params.md). Not a corpus match.
- **Hybrid (vector + FTS) is a strict retrieval winner:** 2:1 weights keep
  recall at 0.900 and lift MRR 0.717→0.783 (1:1 → 0.767) at near-zero latency
  cost (FTS + RRF only). `HYBRID_WEIGHT_VECTOR=2.0` / `HYBRID_WEIGHT_LEXICAL=1.0`
  are now the sanctioned defaults (inert while disabled).
- **"Both" levers compound harm** (worse than either alone) — do not combine
  them on this corpus.
- bd17/bd18 (Header / Depends paraphrase cold-start) unreachable by *any*
  config — no query term reaches the gold source; only agentic reformulation
  (Phase-2 loop) can change the query. Not a retrieval-lever fix.

**Standing decisions:**
- `HYBRID_ENABLED` stays `0` (default-off) until the answer-level gate on the
  hybrid-2:1 config passes — retrieval-level gains are necessary but not
  sufficient (AGENTS.md evidence-first precedent, judge-skip).
- `RERANK_ENABLED` stays `0` on the retrieval evidence above; no answer-level
  spend is warranted for a config that loses at retrieval.

### 6.5.3 answer-level evidence

Plan pivoted mid-gate: the levers-off **expanded-baseline run** (30 rows, both
pipelines; stamp `20260910_215240`) is now the committed reference that
supersedes the 15-row `20260910_201739` for post-hardening comparisons. The
classic half **completed** day 1; the agentic half was **stalled** by the Groq
daily-TPD wall twice (199,155/200,000 used; 0 rows persisted; resume lossless)
before completing 2026-09-11 after switching to a **fresh Groq organization**
(`org_01m288jaqwey6agkpkk8ch80vv`, new-org key set in `.bashrc` and picked up by
launching under `bash -ic` — env var overrides the `.env` fallback). No *any*s
lever-on answer-level spend happened — the 6.5.2 table shows why (levers
strictly worse at retrieval), so the retrieval gate already rules the levers
off; if tuning finds a winner it needs a fresh answer-level day before flipping
defaults.

**Expanded baseline — classic pipeline, levers OFF, stamp `20260910_215240`:**

| Metric | value |
|--------|-------|
| answer_correctness | 0.8167 (24.5/30) |
| retrieval_recall@k | 0.900 (18/20 — matches the zero-LLM probe exactly) |
| citation_validity | 1.000 (n=14) |
| citation_gold_accuracy | 0.628 (n=13) |
| refusal_accuracy | 1.000 (6/6 — bn01–bn06 all correct) |
| groundedness_rate | 0.762 (16/21) |
| avg_latency_ms | 11,130 |
| retrieval/tool calls | 1.0 / 0.0 |

Groundedness with the expanded set is the clearest weak spot (bd03/bd05/bd14/
bd20 ungrounded; bd09/bd18 partial facts; bd17/bd18 recall-miss) — a Phase 6
input but **not** fixed by either lever.

**Expanded baseline — agentic pipeline, levers OFF, stamp `20260910_215240`
(completed 2026-09-11, 30 rows, exit 0):**

| Metric | value |
|--------|-------|
| answer_correctness | 0.850 (25.5/30) |
| retrieval_recall@k | 0.900 (18/20 — same as classic) |
| citation_validity | 1.000 |
| citation_gold_accuracy | 0.694 (n=18) |
| refusal_accuracy | 1.000 (6/6) |
| groundedness_rate | 0.714 (15/21) |
| avg_latency_ms | 24,772 |
| avg retrieval calls | 1.233 |
| avg tool calls | 0.167 (GitHub tool reached on live rows) |

**Verdict (30 rows, both pipelines, installed in the report
`benchmark_20260910_215240.json` → comparison):** agentic wins
answer_correctness (+3.3pp) and citation_gold (+6.6pp); ties recall@k,
citation_validity, refusal_accuracy (all 1.000/0.900 saturation); classic wins
groundedness (−4.8pp for agentic), latency (2.2× cheaper), and both call-count
metrics (agentic is modest: 1.23 retrievals, 0.17 tools — not runaway), **4
classic / 2 agentic / 3 ties**. Agentic is justified *only* for hard/anomalous
questions, never as the default path — it costs 2.2× latency for +3.3pp
correctness with slightly worse grounding.

**Reproducibility (15-row overlap vs the committed `20260910_201739`):** classic
0.800 → 0.800 (14/15 same; sole flip bd02 0.5→1.0) and agentic 0.933 → 0.933
(15/15 same) — the expanded baseline reproduces the committed numbers exactly.
Note: Groq's daily TPD is a **rolling 24-hour window**, not a calendar-day reset
(2026-09-11 evidence: same-org bucket ~99% full midday); a fresh bucket = new
**organization**, never a same-org re-key.

**Pending:** the hybrid answer-level gate — classic pipeline with
`HYBRID_ENABLED=1` (2:1 weights), 30 rows, fresh stamp — the last answer-level
test before `HYBRID_ENABLED` may flip on (retrieval gate: MRR 0.717→0.783,
recall parity). Runs on the new org's bucket (8k TPM throttling makes runs
slower but they fit within 200k TPD).

## 7. phase_6: "Code Generation / Validation"
- status: PLANNED — gated; **no implementation before the pending hybrid
  answer-level gate passes and the `HYBRID_ENABLED` decision is made** (see
  §6.5 "Pending" + §10). This section is the working HOW for that phase, not a
  license to start it.
- summary (SPEC §8, verbatim scope): >
    Documentation retrieval → generate code → validate against retrieved
    API/schema/examples → return code + sources. Only after the core is
    reliable; never pulled into the pitch/demo until actually built.
- build_order placement: after Phase 5 (closed) and after the §6.5 hardening
  gate closes. Phase 6 output never enters the fast/cheap core loop — it is an
  explicit, opt-in route (SPEC §8 guardrail + AGENTS.md rule 3).

### 7.1 gate (do before writing any Phase 6 code)
- [ ] run the hybrid answer-level gate (classic, `HYBRID_ENABLED=1`, 2:1
      weights, 30 rows, fresh stamp) via
      `bash -ic 'bash /tmp/opencode/run_hybrid_gate.sh'` on a fresh TPD bucket
- [ ] record the before/after answer-level table (correctness, groundedness,
      citation_gold, latency) in §6.5 and DECIDE `HYBRID_ENABLED` on the data
      (recall gate already: MRR 0.717→0.783, recall parity — answer level is
      the last evidence)
- [ ] user sign-off to enter Phase 6 (per AGENTS.md rule 3 — never self-enter)
- [ ] only then: flip PLAN `current_phase` → 6 and cut the `phase-6-start`
      branch/milestone

### 7.2 task breakdown & file ownership (single dev; ownership = area of change)

| # | Task | Files owned | Depends on |
|---|------|-------------|------------|
| T1 | `CodeValidator` interface + `RetrieveThenValidate` impl: given a code output + retrieved API/schema/examples, return verdict (pass/fail + reasons) — static/structural checks first (symbols, signatures, required imports), LLM judge only for semantic fits | `src/docpilot/validation/` (new: `validator.py`, `verdict.py`) | gate §7.1 |
| T2 | `CodeGenerator`: retrieves docs (existing `Retriever`), builds a code-request prompt with the retrieved API/schema/examples inline, calls the existing `Generator` interface, attaches `CitationEngine` sources to every code block | `src/docpilot/codegen/` (new: `pipeline_ask_code.py`) | T1 |
| T3 | Code route in the fast path only as an **explicit opt-in** (query intent or API surface; never the default answer path) — reuse the Phase 2 gate/strategy interfaces; a non-code question must fall back to the normal ask | `src/docpilot/agent/` (strategy dispatch), `config.py` | T2 |
| T4 | Validation loop: unvalidated code is never returned — on `fail` the generator reformulates (reuse judge/max-turns pattern from Phase 2, capped) and re-validates; persistent fail → honest "couldn't validate — not returning code" with sources | `src/docpilot/validation/`, `src/docpilot/agent/` | T1–T3 |
| T5 | Evaluation: extend Phase 4 `Evaluator`/dataset with code rows — gold: compiles/imports cleanly, symbol usage matches retrieved API, citations resolve to the right doc pages; reuses the harness (`run_pipeline` duck-typed runner, row checkpoints) | `src/docpilot/eval/` (dataset + metrics), `tests/test_eval_benchmark.py` | T2 |
| T6 | API/UI surface (Phase 5 patterns): `POST /api/v1/code` streaming route + Chainlit "ask for code" control; debug panel already carries trace + sources — validation verdict is a new trace event | `src/docpilot/api/`, `src/docpilot/chainlit_app.py` | T4, T5 |
| T7 | Validation-fixture corpus: small checked-in `.md`/`.py` pairings (retrieved API spec ↔ expected valid/invalid sample code) so T1/T4 are hermetic-testable without live LLM | `tests/fixtures/codegen/` | T1 |

### 7.3 execution order (parallelizable steps collapse into the row)

1. T7 fixtures + T1 interface + structural validator (hermetic, no LLM) with
   tests — this is buildable first and de-risks everything else
2. T2 code path (retrieve → prompt → generate → cite) behind the existing
   `Generator`/`CitationEngine` interfaces
3. T1→T2 wiring in a code-gated route (T3) — still no validation loop
4. T4 validation loop + reformulation cap
5. T5 eval rows + a small code-focused benchmark run (hermetic + live)
6. T6 API/UI surface, reusing Phase 5 patterns untouched
7. sweep: README/SPEC-consistent claims, CHANGELOG-able milestone, `phase-6`
   tag when §7.5 is all `[x]`

### 7.4 interfaces (reuse, don't rebuild)

- reuse: `Retriever`, `Generator`, `Agent`/strategy dispatch, `CitationEngine`,
  `Evaluator`, the benchmark harness (checkpointing + duck-typed runners from
  the §6.5 hardening)
- new: `CodeValidator` (see AGENTS.md interfaces list) — retrieval, generation,
  and validation stay behind interfaces so the framework-extraction goal
  (§8) is not prejudiced

### 7.5 exit_criteria (checked when the phase closes — mirror §6.3 style)

- [ ] `CodeValidator` returns pass/fail + reasons with no hallucinated
      "fine" — structural checks are deterministic; LLM judge only for
      semantic fit
- [ ] code route is explicit and opt-in; non-code queries never produce a
      code answer; the fast/cheap path is byte-identical for non-code asks
- [ ] unvalidated code is never returned — persistent validation failure →
      refusal-style "couldn't validate" with sources, never fabricated code
- [ ] generated code carries citations resolving to real doc pages (existing
      `CitationEngine` output, no new marker syntax)
- [ ] Phase 4–5 suite + new T1/T2/T4/T5 tests green (the current 472 baseline
      + additions)
- [ ] a code-focused eval run exists in `src/docpilot/eval/reports/` with
      validation-pass rate + citation accuracy; compared against the
      non-code baseline where overlap exists
- [ ] README/pitch honest: Phase 6 described only to the extent implemented
      (SPEC §8: not a differentiator until it's built)

### 7.6 non-scope / deferred (decide later, not now)

- generalized multi-language codegen (start with the docs' own language —
  Python/fastapi-flavored docs corpus); framework extraction (§8) stays
  post-Phase-6
- no change to how the §6.5 levers (`RERANK_ENABLED`, `AGENT_JUDGE_*`) behave;
  Phase 6 does not paper over retrieval weaknesses — fix retrieval first
  (AGENTS.md build-order rule 1)

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
