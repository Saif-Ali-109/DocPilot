"""DocPilot configuration — loads all settings from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(
            f"Required environment variable {key} is not set. "
            f"Add it to your .env file (see SPEC.md §3.16 for all keys)."
        )
    return val


# --- Secrets (required) ---
GROQ_API_KEY: str = _require("GROQ_API_KEY")
POSTGRES_USER: str = _require("POSTGRES_USER")
POSTGRES_PASSWORD: str = _require("POSTGRES_PASSWORD")

# --- Secrets (optional, sensible defaults) ---
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "3"))

# --- Database ---
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "docpilot")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking ---
CHUNK_SIZE_TARGET: int = int(os.getenv("CHUNK_SIZE_TARGET", "650"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "75"))

# --- Retrieval ---
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_LANGUAGE: str = os.getenv("RETRIEVAL_LANGUAGE", "en")

# --- Reranking (PLAN §H finding 1) ---
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "0") == "1"
"""Master switch for the cross-encoder reranking pass.

``0`` (default) keeps retrieval byte-identical to Phase 1–5 (pure cosine
top-k).  ``1`` makes ``SimpleRetriever`` fetch ``RERANK_CANDIDATES`` from the
vector store and re-score them with the ``RERANKER_MODEL`` cross-encoder
before cutting to ``top_k``.  A cross-encoder pass over the top-N window:
*re-orders* by query-specific relevance — it cannot resurrect a chunk the
dense index did not retrieve at all.  Reranking alone is a candidate-in
re-ordering lever; the exact-match recall gap is the hybrid retriever's job
(PLAN §H finding 2).

Gate evidence (PLAN §6.5.2, 2026-09-10): on the 30-row eval suite the
cross-encoder re-orders *worse* than plain cosine — recall drops 0.900→0.850
in every rerank variant (prefixes/weights included) — and costs ~85 s/predict
on this CPU.  It stays default-off by evidence, not by omission."""

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
"""Cross-encoder model for ``BCEReranker`` (downloaded lazily, CPU)."""

RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "20"))
"""Width of the candidate window fetched before reranking down to top_k."""

# --- Hybrid retrieval (PLAN §H finding 2) ---
HYBRID_ENABLED: bool = os.getenv("HYBRID_ENABLED", "0") == "1"
"""Master switch for hybrid (vector + lexical) retrieval.

``0`` (default) keeps retrieval byte-identical to Phase 1–5 (pure cosine
top-k).  ``1`` makes the default retriever a ``HybridRetriever`` that fuses
the dense vector path (optionally reranked) with Postgres full-text search
(``PostgresFTSSearcher`` over ``db/schema.sql``'s ``chunks_content_fts`` GIN
index) via weighted reciprocal rank fusion.  The lexical half rescues exact
API identifiers / error codes the dense index under-weights; the vector half
rescues paraphrase queries keyword matching cannot express.  Both levers
(re-rank + hybrid) can be on together — they act on different failure modes.

Gate evidence (PLAN §6.5.2, 2026-09-10): on the 30-row eval suite the hybrid
half *alone* improves retrieval (recall parity 0.900, MRR 0.717→0.783 at the
2:1 weights below) for negligible latency cost; the cross-encoder reranker
*hurts* retrieval on this corpus and stays off.  ``1`` here is the answer-
level-gated follow-up, not yet shipped.
"""

HYBRID_TOP_K_EACH: int = int(os.getenv("HYBRID_TOP_K_EACH", "5"))
"""Candidate-list width pulled from each half before RRF fusion."""

HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
"""RRF smoothing constant (standard value 60); higher flattens rank advantage."""

HYBRID_WEIGHT_VECTOR: float = float(os.getenv("HYBRID_WEIGHT_VECTOR", "2.0"))
"""Relative weight of the dense-vector half in RRF fusion.

Gate-winning setting (PLAN §6.5.2): 2:1 vector over lexical lifted MRR
0.717→0.783 at recall parity; 1:1 ranked lower (0.767).
"""

HYBRID_WEIGHT_LEXICAL: float = float(os.getenv("HYBRID_WEIGHT_LEXICAL", "1.0"))
"""Relative weight of the lexical half in RRF fusion (see vector weight)."""

# --- Phase 2: Agentic retrieval (SPEC §4.3, PLAN §3.5) ---
AGENT_MAX_RETRIES: int = int(os.getenv("AGENT_MAX_RETRIES", "2"))
"""Maximum judge/reformulate iterations before the agent refuses (SPEC §4.1)."""

AGENT_LOOP_TOP_K: int = int(os.getenv("AGENT_LOOP_TOP_K", "5"))
"""Retrieval count used inside the agentic loop (the agentic retrieve calls)
when the caller did not override ``top_k``.  The direct/fast path keeps
``RETRIEVAL_TOP_K`` (5).  Phase 5 hardening (SPEC §7 amendment 2026-09-09):
the loop's context was broadened to 8 in Phase 2; Phase 5 shrinks it back to 5
so the judge + answer prompts carry less context (latency lever), validated by
the before/after §6.1 benchmark run."""

AGENT_DEFAULT_STRATEGY: str = os.getenv("AGENT_DEFAULT_STRATEGY", "auto")
"""Default ``ask --strategy`` when the flag is not given (``auto|direct|agentic``)."""

AGENT_JUDGE_SKIP_MIN_SCORE: float = float(os.getenv("AGENT_JUDGE_SKIP_MIN_SCORE", "0.0"))
"""Fast-path judge skip (SPEC §7 amendment 2026-09-09).

When ``0.0`` (default) the agentic loop always runs the judge LLM call — Phase
2/4 behaviour unchanged.  When ``> 0``, the retrieve node marks
``skip_judge`` when the top retrieval score clears the threshold and the graph
routes straight to the answer node (one LLM call saved per question).  The
threshold is deliberately disabled until the before/after §6.1 benchmark run
justifies a value — the locked evaluation gate decides, not an ad hoc guess."""

AGENT_JUDGE_MODEL: str = os.getenv("AGENT_JUDGE_MODEL", "")
"""Optional separate Groq model for the judge; empty string → ``GROQ_MODEL``."""

AGENT_JUDGE_SCORE_FLOOR: float = float(os.getenv("AGENT_JUDGE_SCORE_FLOOR", "0.0"))
"""Score-floor sanity backstop for the LLM sufficiency judge.

``0.0`` (default) disables it — the loop is identical to Phase 2/4/5
behaviour.  When ``> 0``, :class:`ScoreFloorBackstopJudge` (agent/judge.py)
forces an ``"insufficient"`` verdict whenever the top retrieval score is
below the floor (or retrieval is empty), so the loop can never route
straight to ``answer`` on thin evidence — a heuristic cross-check on the
LLM-as-judge.  Tool and retry signals are preserved, so live-state questions
still reach the GitHub tool.  Pre-Phase-6 hardening (PLAN §H finding 6);
enabled only if the expanded benchmark run justifies a value."""

AGENT_GATE_LONG_THRESHOLD: int = int(os.getenv("AGENT_GATE_LONG_THRESHOLD", "18"))
"""Word-count gate trigger.

Wired into ``HeuristicQueryClassifier`` (agent/gate.py) — the classifier's
constructor reads this key when no explicit threshold is passed (and env
``AGENT_GATE_LONG_THRESHOLD`` overrides the code default).  The threshold is a
heuristic (word count > this fires the ``long_question`` complexity signal);
routing calibration against a larger eval set is a pre-Phase-6 hardening
follow-up (PLAN §H)."""

# --- Phase 3: GitHub tool (SPEC §5, PLAN §4) ---
GITHUB_PAT: str = os.getenv("GITHUB_PAT", "")
"""GitHub Personal Access Token (optional — empty by default).

An empty PAT disables the GitHub tool: it returns a non-``ok`` ToolResult and
never makes an HTTP call (locked decision, PLAN §4).  Never log this value.
"""

GITHUB_API_BASE: str = os.getenv("GITHUB_API_BASE", "https://api.github.com")
"""GitHub REST API base URL — defaults to the public ``api.github.com``."""

GITHUB_OWNER: str = os.getenv("GITHUB_OWNER", "")
"""Default repository owner for the GitHub tool (e.g. ``fastapi``)."""

GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")
"""Default repository name for the GitHub tool (e.g. ``fastapi``)."""

# --- Phase 6: Code generation / validation (SPEC §8, PLAN §7) ---
CODE_ROUTE_ENABLED: bool = os.getenv("CODE_ROUTE_ENABLED", "0") == "1"
"""Arm the opt-in code route (PLAN §7.2 T3).

``0`` (default OFF): the code path is never reached through query routing —
only an explicit per-request opt-in (``explicit_code=True``) can take it.
``1`` arms the classifier-driven route: queries that look like code requests
(``CODE_INTENT_PHRASES``) may be answered with generated code.  Non-code
queries always fall back to the standard answer path, byte-identical to the
levers-off baseline (§7.5)."""

CODE_VALIDATE_MAX_TURNS: int = int(os.getenv("CODE_VALIDATE_MAX_TURNS", "2"))
"""Maximum validation/reformulation iterations on the code route (PLAN §7.2 T4).

The T4 loop generates code, validates it with the T1 ``CodeValidator`` and,
on a failed verdict, feeds the failure reasons back for one rewrite per turn.
After ``CODE_VALIDATE_MAX_TURNS`` rewrites (so 1 initial + N fixes) a still
failing output is **not returned** — the route answers with a "couldn't
validate" refusal plus the retrieved sources (§7.5: unvalidated code is never
returned)."""

CODE_INTENT_PHRASES: tuple[str, ...] = (
    "write code", "write a function", "write a class", "write an example",
    "generate code", "generate a function", "generate an example",
    "code snippet", "code example", "example code", "sample code",
    "show me the code", "show code", "give me the code",
    "how do i write", "how do i implement", "how can i write",
    "implement a function", "implement a class", "implementation for",
)
"""Deterministic code-intent seed phrases (PLAN §7.2 T3).

Matched case-insensitively against the normalised query (same normalisation
as the Phase 2 heuristic gate, ``agent/gate.py``).  A documented seed list —
like the gate's connector vocabulary it gets tuned against corpus-observed
phrasings, never extended ad hoc during a run."""

# --- Phase 5: API + UI (SPEC §7) ---
DOCPILOT_DB_PATH: str = os.getenv("DOCPILOT_DB_PATH", "data/docpilot.sqlite3")
"""SQLite file backing the API session/chat history (sessions + messages).
Defaults under ``data/`` which is git-ignored — the DB is never committed."""
