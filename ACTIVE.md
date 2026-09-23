---
title: DocPilot — ACTIVE current view
role: derived current view — phase, gates, constants, load index
authority: lowest — fix ACTIVE if it contradicts SPEC/PLAN
load: read whole — the only file meant to be read whole
last_updated: 2026-09-23
---

# ACTIVE — DocPilot current view (derived, never authoritative)

> Every session: read this file in full first, then load ONLY the SPEC/PLAN
> line ranges listed in §4 — never the whole `SPEC.md` / `PLAN.md`.
> Authority: **SPEC.md > PLAN.md > ACTIVE.md**. If ACTIVE contradicts them,
> fix ACTIVE. Refresh it in the same commit as any SPEC/PLAN/lever/phase change
> (§6).

## 1. Status

- current_phase: **7 — Framework Extraction (ragkit), ACTIVE** (entered
  2026-09-17). **Stage 1 (core chain) COMPLETE 2026-09-18 (`v0.1.0`). Stage 2
  (agentic) COMPLETE 2026-09-18 (`v0.2.0`). Stage 3 (eval) COMPLETE
  2026-09-19 (`v0.3.0`, DocPilot pinned `@v0.3.0`). Stage 4 — codegen,
  ACTIVE 2026-09-23 — the **final** extraction stage.**
- last_updated: 2026-09-23
- servers: uvicorn :8000 (`/api/v1/health` — store ok, ~15.3k chunks; serves
  `POST /api/v1/code`), chainlit :8050 — keep both healthy.
- test baseline: **combined 552** (ragkit 410 + DocPilot 142; unit tests
  relocated zero loss; DocPilot can't import `docpilot.codegen`/`docpilot.validation`
  until Stage 4 completes).

### Gate status (resolved 2026-09-12)

- **Hybrid answer-level gate COMPLETE** — stamp `20260912_080743`, exit 0, one
  8-min run on the refreshed bucket: correctness +1.7pp (25/30), groundedness
  −9.5pp (16→14/21), citation_gold −1.8pp (n 13→17), latency +8%; bd17 flipped
  1.0→0.0 (hybrid re-ranking displaced its gold chunk). Full table + verdict →
  PLAN §6.5.
- **Decision: `HYBRID_ENABLED` stays OFF** — no net answer-level win; same
  verdict shape as RERANK (retrieval gain ≠ answer-level gain).
- **WI-1 Judge-skip gate COMPLETE** — stamp `20260912_083701` (0.5), `20260912_102852`
  (0.6), `20260912_103621` (0.7), all on gpt-oss-120b; 0.0 baseline
  `20260912_082131` on gpt-oss-20b.  Correctness regressed at every threshold:
  0.900 → 0.833 (−6.7pp) / 0.783 (−11.7pp) / 0.733 (−16.7pp).  Live-state
  category (bl01–bl04) systematically broken: tool_calls 0.20→0.00,
  correct→refused/partial on all 4 rows.  **Decision: `AGENT_JUDGE_SKIP_MIN_SCORE`
  locked at 0.0 (disabled).**  Full table + flips in `wi1_analyze2.py` output.
- Phase 6 ACTIVE (sign-off 2026-09-12): building per PLAN §7.3 — **T7 + T1
  DONE** (hermetic `CodeValidator` + fixture corpus + 16 tests); **T2 DONE**
  (`ask_code` codegen pipeline, 6 tests); **T3 DONE** (opt-in code dispatch
  `agent/code_route.py` + T1→T2 verdict wiring, 14 tests); **T4 DONE**
  (validation loop + cap, `CODE_VALIDATE_MAX_TURNS`, refusal-with-sources on
  persistent failure, 5 tests); **T5 DONE** (code eval —
  `eval/code_benchmark.py` + committed dataset, validation-pass/citation
  metrics, 17 tests); **T6 DONE** (API/UI — `POST /api/v1/code` streaming
  route + `service.code_events` with the new `code` validation-verdict
  event, Chainlit *Generate validated code* settings toggle, 11 tests);
  §7.5 exit-criteria sweep COMPLETE (2026-09-12);
  `phase-6` tag cut.

## 2. Contract constants

- **Refusal sentence** (SPEC §3.9, mandatory "I don't know" wording):
  `"I don't know — the available documentation does not cover this question."`
  Effective refusal flag = pipeline flag **OR** verbatim-text match on the
  answer body.
- **Levers** (all OFF pending gate data):
  - `RERANK_ENABLED=OFF` — dead on this corpus (recall 0.850 across all
    variants, ~85 s/predict CPU). Permanent verdict; revisit only with a new
    corpus. Do not re-enable to "fix" retrieval.
  - `HYBRID_ENABLED=OFF` — until the hybrid gate passes (§1).
  - `AGENT_JUDGE_SKIP_MIN_SCORE=0.0` — judge-skip **disabled by evidence gate**
    (WI-1: every threshold 0.5/0.6/0.7 regressed correctness −6.7/−11.7/−16.7pp
    and broke live-state tool calls 0.20→0.00).
  - `AGENT_JUDGE_SCORE_FLOOR=0.0` — score-floor backstop disabled (no gate yet).
  - Gate-only weights: hybrid **2.0 / 1.0** — never default; `HybridRetriever`
    ctor stays 1.0/1.0 for tests.
- **Quota facts (Groq)**:
  - TPD is a **rolling 24-hour window**, not a calendar-day reset. A fresh
    bucket requires a **new organization** — same-org re-keys never help.
  - Working org: 200k TPD, 8k TPM — throttling slows runs but does not block
    them (a 30-row run fits within TPD).
- **Key/env rules**:
  - Never read or modify `.env` or shell profiles (`.bashrc`).
  - Keys arrive via `bash -ic` (env overrides `.env`); run keyed commands that way.
  - Inject `GITHUB_OWNER=fastapi GITHUB_REPO=fastapi` per command.
  - Never commit `opencode.jsonc` or the `benchmark_20260908_184441*` reports
    (intentionally untracked).

## 3. Current work state

- Eval baseline COMPLETE + committed (`120887f`, stamp `20260910_215240`):
  agentic correctness 0.850 vs classic 0.817 (4 classic / 2 agentic / 3 ties);
  groundedness weak spot bd03/bd05/bd14/bd20, partial bd09/bd18, recall-miss
  bd17/bd18; reproducibility confirmed on the 15-row overlap. Truth: PLAN §6.5.3
  (range in §4 load index) + its committed benchmark JSON.
- Harness hardening (TPD clean-exit + row-level checkpoints in the eval
  harness) DONE — exit code 3 on daily-quota wall, resume skips checkpointed
  rows, full-set report preserved.
- Hybrid gate DONE 2026-09-12 — verdict `HYBRID_ENABLED` stays OFF (evidence
  in §1 + PLAN §6.5); reports committed (stamp `20260912_080743`).
- **WI-1 Judge-skip gate DONE 2026-09-12** — stamps `20260912_082131` (0.0),
  `20260912_083701` (0.5), `20260912_102852` (0.6), `20260912_103621` (0.7).
  Evidence: correctness 0.900→0.833/0.783/0.733; tool_calls 0.20→0.00 (live-
  state broken); latency 26.8s→14.0/11.8/11.0s. **Locked at 0.0 (disabled).**
- Code eval `20260912_codefix` COMPLETE — report
  `src/docpilot/eval/reports/code_benchmark_20260912_codefix.json`
  (6 rows, committed as evidence): validation-pass 0.75, compiles 1.0,
  citation-gold 0.333, refusal accuracy 0.833, avg 1.7 validation turns,
  ~11.4 s/row (live-run budget TPD-rolled-off; every row retried until
  scored).
- Phase 6 COMPLETE — §7.5 exit-criteria sweep passed and `phase-6` tag
  cut; T1–T6 done (see §1 gate status). Levers stay OFF (gate verdict).
- Phase 7 ACTIVE 2026-09-17 — framework extraction (ragkit): PLAN §8 signed
  off (separate repo https://github.com/Saif-Ali-109/ragkit.git, dogfood,
  staged core-first, no CI/PyPI this phase). S1-T1 DONE (repo scaffolded +
  pushed + editable-installed into the DocPilot venv); S1-T1b DONE (ragkit
  self-docs — SPEC/PLAN/ACTIVE trio + README pointer). S1-T2 DONE (moved
  the §8.2 core-chain modules + unit tests into ragkit, mechanical prefix
  swap; moved suite green in the DocPilot venv — 182 passed, 1 skipped).
  S1-T3 DONE (DocPilot dogfood rewire: src + tests import `ragkit.*`; moved
  modules + 11 unit-test files deleted; pyproject pins ragkit `@ed0f908`;
  DocPilot 369 + ragkit 183 = combined 552). S1-T4 DONE (connection
  semantics ownership: `ragkit/config.py` owns the framework keys —
  env-read, safe defaults, no hard-fail; core-chain modules read
  `ragkit.config`, reverse dep gone; DocPilot re-exports non-secret keys and
  loads `.env` before ragkit.config; `db/maintenance.py` moved to
  `ragkit.db` — dedupe CLI now uses it; DocPilot 361 + ragkit 191 = combined
  552; ragkit live-PG tests now hermetic-skip under a bare environment).
  S1-T5/T6/T7 DONE (standalone hermetic suite green 181+10-skips; combined
  552 re-verified; retrieval parity evidence — 30 benchmark queries top-k
  IDENTICAL pre/post, live same-process run at `docpilot@69f91dc` vs
  `docpilot@ebbdefa`+`ragkit@34686e2`; CLI smoke exit 0). S1-T8 DONE (exit
  sweep: READMEs honest, clean-venv install from git verified, `v0.1.0`
  tagged + pushed on ragkit, DocPilot pin → `@v0.1.0`, §8.5 all `[x]`,
  `phase-7` tag cut). **Stage 1 COMPLETE — paused for review before Stage 2
  (agentic).**
- 2026-09-18 post-close: independent audit of Stage 1 — merge-ready, no
  blockers; stale §8.3 S1-T5/T6/T7 checkboxes ticked (`391ebc8`); ragkit
  conftest gained optional repo-root `.env` loading (9 live-PG tests run on
  dev machines: ragkit 190+1 there, 181+10 hermetic — totals identical).
- 2026-09-18 S2-T1: Stage 2 (agentic) kicked off — task list + exit criteria
  written (§8.7/§8.7b); ragkit PLAN §3 mirrors it; ACTIVE refreshed.
- 2026-09-18 S2-T2: mechanical move — `docpilot.agent` → `ragkit.agent`,
  `docpilot.tools` → `ragkit.tools` (prefix swap); `ragkit.config` extended
  with 10 keys; langgraph dep added; 6 hermetic agent/tool test files moved.
  `3ded3fb` — ragkit 299 passed/1 skipped, 290/10 hermetic.
- 2026-09-18 S2-T3: DocPilot dogfood — `src/docpilot/` + `tests/` imports
  rewired to `ragkit.agent.*` / `ragkit.tools.*`; moved modules + 6 test
  files deleted; DocPilot config re-exports the 10 keys; pin → `@3ded3fb`.
  DocPilot 252 + ragkit 300 = 552.
- 2026-09-18 S2-T4: agentic parity evidence — hermetic determinism harness
  (fakes, fixed queries: routing, judge-retry, fake-tool live path): full
  trace identical pre (`docpilot.agent` @ Stage-2 start `391ebc8`) vs post
  (`ragkit.agent`). 6/6 scenarios IDENTICAL + one live CLI smoke (`docpilot
  ask --strategy agentic`, exit 0). Evidence: `ragkit/parity/s2_*`.
- 2026-09-18 S2-T5: exit sweep — READMEs honest, clean-venv install from git
  `@v0.2.0` (deps incl. langgraph 1.2.11 resolved; imports OK; `docpilot`
  not importable), tag `v0.2.0` (`500717d`), DocPilot pin → `@v0.2.0`,
  §3.3 / §8.7b all `[x]`, both repos pushed. **Stage 2 COMPLETE.**
- 2026-09-18 S3-T1: Stage 3 (eval) kicked off — task list + exit criteria
  written (§8.8/§8.8b); ragkit PLAN §4 mirrors it; Stage 4 (§8.9) scope
  pre-planned in the same commit; ACTIVE refreshed both repos (`6122c8c` /
  ragkit `b8f35b5`).
- 2026-09-19 S3-T2: mechanical move — `docpilot.eval` → `ragkit.eval`
  (benchmark/triples/judge_ab/tool_necessity + `__main__`/`__init__` + 3
  dataset JSONs); 3 hermetic eval test files moved; `code_benchmark.py` +
  `test_eval_code_benchmark.py` stay DocPilot-side (Stage 4) but rewire to
  `ragkit.eval.benchmark`. `309037b` — ragkit 410.
- 2026-09-19 S3-T3: DocPilot dogfood — moved eval modules + 3 tests deleted;
  `python -m docpilot.eval` → `python -m ragkit.eval` (code-benchmark
  dispatcher lazy-guarded "moves in Stage 4" unless run as
  `python -m docpilot.eval.code_benchmark`); pin → `@309037b`; DocPilot 142
  + ragkit 410 = 552.
- 2026-09-19 S3-T4: eval parity evidence — hermetic report-JSON harness:
  pre `docpilot.eval`@`6122c8c` (worktree) vs post (`ragkit.eval`) — report
  JSON **IDENTICAL field-for-field** (judge_ab 24, tool_necessity 15,
  benchmark 30); + one live TPD-aware smoke (`python -m ragkit.eval
  judge-ab`, 48 live Groq calls). Evidence: `ragkit/parity/s3_eval_*`.
- 2026-09-19 S3-T5: exit sweep — READMEs honest, clean-venv install from git
  `@v0.3.0` (deps unchanged vs `v0.2.0`), tag `v0.3.0` (`f03c2e7`), DocPilot
  pin → `@v0.3.0`, §8.8b all `[x]`, both repos pushed. **Stage 3 COMPLETE.**
- 2026-09-23 S4-T1: Stage 4 (codegen) kicked off — §8.9 scope+tasks
  pre-written (`6122c8c`); ACTIVE refreshed (Stage 3 COMPLETE, Stage 4
  ACTIVE); Stage-4-start pre-side worktree created
  (`/tmp/opencode/s4_parity_pre` @ `34dea0d`). Next: S4-T2 — mechanical
  move of `codegen/*`, `validation/*`, `agent/code_route.py`,
  `eval/code_benchmark.py` + dataset into ragkit.
- Stage 4 (final, §8.9) moves `codegen/*`, `validation/*`,
  `agent/code_route.py`, `eval/code_benchmark.py` + `code_benchmark.json` →
  `ragkit.codegen` / `ragkit.validation` / `ragkit.agent.code_route` /
  `ragkit.eval` (activates the code-benchmark `__main__` lazy-guard); 3
  CODE_* config keys; 5 hermetic test files move; `test_api_code.py` stays.
  Gate: full parity harness + 1 live codegen smoke. After S4-T5: extraction
  complete.

## 4. Load index — read ONLY these SPEC/PLAN lines

Core (always) + current-phase bundle; all other rows on demand.

| File | Section | Lines | Load when |
|---|---|---|---|
| SPEC.md | §1 Project Overview | 17–31 | core — always |
| SPEC.md | §2 Phased Build Order | 32–47 | core — always |
| SPEC.md | §10 Architectural Rules | 755–763 | core — always |
| SPEC.md | §11 Conflict Resolution | 764–769 | core — always |
| SPEC.md | §6.1 Baseline metrics | 579–601 | gate/eval work |
| SPEC.md | §6.5 Live-run protocol & quota | 665–707 | gate/eval work |
| SPEC.md | §8 Phase 6 outline | 736–748 | Phase 6 planning |
| PLAN.md | §1 meta | 11–34 | core — always |
| PLAN.md | §6.5 pre-Phase-6 hardening evidence | 915–1087 | on-demand: Phase-6 evidence |
| PLAN.md | §7 Phase 6 plan (gated) incl. §7.5 exit criteria | 1088–1229 | on-demand: completed-phase history |
| PLAN.md | §8 framework extraction — Stages 1–4 (Stage 4 ACTIVE) | 1231–1536 | always — source-of-truth extraction plan |
| PLAN.md | §8.8 stage-3 tasks (eval, done) | 1440–1495 | completed-phase history |
| PLAN.md | §8.9 stage-4 tasks (codegen, ACTIVE) | 1496–1536 | on-demand: stage-4 work |
| PLAN.md | §9 architectural discipline | 1537–1560 | core — always |
| PLAN.md | §10 conflict resolution | 1561–1571 | core — always |
| PLAN.md | §11 git_workflow | 1572–1604 | core — always |
| SPEC.md | §3–§5 locked phase details | 48–572 | on-demand: task touches that phase's code (classic RAG / agent / tooling) |
| PLAN.md | §2–§5 completed-phase history | 35–913 | on-demand: same rule (incl. §5.3 eval slices, §6 API/UI) |

On-demand rule: load a completed-phase section ONLY if the task changes that
phase's code. If a cited range looks stale, `grep -n '^## '` the file for the
section heading and read that section's lines instead — then refresh this table.

## 5. Interfaces (unchanged from AGENTS.md)

`DocumentLoader`, `Parser`, `Chunker`, `EmbeddingProvider`, `VectorStore`,
`Retriever`, `Reranker`, `Tool`, `Agent`, `Generator`, `CitationEngine`,
`Evaluator`, `CodeValidator` — never hard-code vendor/library calls behind
them (no sprinkling `groq.chat(...)` / `pgvector.search(...)` through
business logic). Phase 7 (PLAN §8): these move into the `ragkit` package;
DocPilot imports ragkit (dogfood).

## 6. Refresh rules (maintain this file)

- After any SPEC.md / PLAN.md edit → update §4 line ranges.
- After any lever flip / phase change / new gate result / milestone → update
  §1–§3 (`last_updated` too).
- After a completed-phase section's code is touched → revisit §4 "on-demand"
  entries only if the task needs them.
- Authority: SPEC > PLAN > ACTIVE. ACTIVE contradicts → fix ACTIVE, never the
  other way.