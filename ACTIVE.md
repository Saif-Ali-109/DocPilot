---
title: DocPilot — ACTIVE current view
role: derived current view — phase, gates, constants, load index
authority: lowest — fix ACTIVE if it contradicts SPEC/PLAN
load: read whole — the only file meant to be read whole
last_updated: 2026-09-11
---

# ACTIVE — DocPilot current view (derived, never authoritative)

> Every session: read this file in full first, then load ONLY the SPEC/PLAN
> line ranges listed in §4 — never the whole `SPEC.md` / `PLAN.md`.
> Authority: **SPEC.md > PLAN.md > ACTIVE.md**. If ACTIVE contradicts them,
> fix ACTIVE. Refresh it in the same commit as any SPEC/PLAN/lever/phase change
> (§6).

## 1. Status

- current_phase: **pre-Phase-6 hardening gate** (Phase 5 COMPLETE). Phase 6
  (code gen/validation) is PLANNED and gated — implementation waits for the
  hybrid gate below (PLAN §7).
- last_updated: 2026-09-11
- servers: uvicorn :8000 (`/api/v1/health` — store ok, ~15.3k chunks),
  chainlit :8050 — keep both healthy.
- test baseline: 472 passing (Phase 1–5 suites).

### Active gate (the only thing blocking Phase 6)

- **Hybrid answer-level gate** — classic pipeline, `HYBRID_ENABLED=1`,
  weights 2.0 vector / 1.0 lexical, 30 rows, fresh stamp. The last
  answer-level test before `HYBRID_ENABLED` may flip on.
  Run: `bash -ic 'bash /tmp/opencode/run_hybrid_gate.sh'` on a fresh 200k TPD
  bucket. **Blocked**: current org's bucket is nearly full (TPD = rolling
  24 h window — see §2).
- After it passes: decide `HYBRID_ENABLED` on the data (retrieval gate
  already passed: MRR 0.717→0.783, recall parity; verdict + tables → PLAN
  §6.5.3). Then user sign-off to enter Phase 6.

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
  - `AGENT_JUDGE_SCORE_FLOOR=OFF` — judge-skip stays disabled.
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
- Allowed while gated: retrieval/test/docs work + planning only — no Phase 6
  implementation, no lever flips.

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
| PLAN.md | §6.5 pre-Phase-6 hardening evidence | 915–1055 | current-phase |
| PLAN.md | §7 Phase 6 plan (gated) | 1056–1142 | current-phase |
| PLAN.md | §9 architectural discipline | 1149–1171 | core — always |
| PLAN.md | §10 conflict resolution | 1172–1182 | core — always |
| PLAN.md | §11 git_workflow | 1183–1211 | core — always |
| SPEC.md | §3–§5 locked phase details | 48–572 | on-demand: task touches that phase's code (classic RAG / agent / tooling) |
| PLAN.md | §2–§5 completed-phase history | 35–913 | on-demand: same rule (incl. §5.3 eval slices, §6 API/UI) |

On-demand rule: load a completed-phase section ONLY if the task changes that
phase's code. If a cited range looks stale, `grep -n '^## '` the file for the
section heading and read that section's lines instead — then refresh this table.

## 5. Interfaces (unchanged from AGENTS.md)

`DocumentLoader`, `Parser`, `Chunker`, `EmbeddingProvider`, `VectorStore`,
`Retriever`, `Reranker`, `Tool`, `Agent`, `Generator`, `CitationEngine`,
`Evaluator` — never hard-code vendor/library calls behind them (no sprinkling
`groq.chat(...)` / `pgvector.search(...)` through business logic).

## 6. Refresh rules (maintain this file)

- After any SPEC.md / PLAN.md edit → update §4 line ranges.
- After any lever flip / phase change / new gate result / milestone → update
  §1–§3 (`last_updated` too).
- After a completed-phase section's code is touched → revisit §4 "on-demand"
  entries only if the task needs them.
- Authority: SPEC > PLAN > ACTIVE. ACTIVE contradicts → fix ACTIVE, never the
  other way.