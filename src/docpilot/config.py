"""DocPilot configuration — loads all settings from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Framework settings are owned by ragkit.config (single source of truth —
# the core chain reads them from there). DocPilot re-exports the non-secret
# keys so ``from docpilot import config`` keeps working unchanged. The
# ``load_dotenv`` above MUST run first: ragkit.config reads os.environ at
# import time and caches it. The secrets below stay DocPilot-owned because
# they keep the fail-fast ``_require`` behaviour.
from ragkit.config import (
    AGENT_GATE_LONG_THRESHOLD,
    AGENT_JUDGE_MODEL,
    AGENT_JUDGE_SCORE_FLOOR,
    AGENT_JUDGE_SKIP_MIN_SCORE,
    AGENT_LOOP_TOP_K,
    AGENT_MAX_RETRIES,
    CODE_INTENT_PHRASES,
    CODE_ROUTE_ENABLED,
    CODE_VALIDATE_MAX_TURNS,
    EMBEDDING_MODEL,
    GITHUB_API_BASE,
    GITHUB_OWNER,
    GITHUB_PAT,
    GITHUB_REPO,
    GROQ_MAX_RETRIES,
    GROQ_MODEL,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PORT,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    RETRIEVAL_LANGUAGE,
    RETRIEVAL_TOP_K,
)


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
# GROQ_MODEL / GROQ_MAX_RETRIES are owned by ragkit.config (imported above).

# --- Database ---
# POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB are owned by ragkit.config
# (imported above).

# --- Chunking ---
CHUNK_SIZE_TARGET: int = int(os.getenv("CHUNK_SIZE_TARGET", "650"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "75"))

# --- Retrieval ---
# RETRIEVAL_TOP_K and RETRIEVAL_LANGUAGE are owned by ragkit.config
# (imported above).

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

# RERANKER_MODEL / RERANK_CANDIDATES are owned by ragkit.config (imported
# above).

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
# AGENT_MAX_RETRIES and AGENT_LOOP_TOP_K are owned by ragkit.config (imported
# above).  AGENT_LOOP_TOP_K = 5: Phase 5 shrunk the loop context back to 5
# (latency lever validated by the before/after §6.1 benchmark run).

AGENT_DEFAULT_STRATEGY: str = os.getenv("AGENT_DEFAULT_STRATEGY", "auto")
"""Default ``ask --strategy`` when the flag is not given (``auto|direct|agentic``)."""

# AGENT_JUDGE_SKIP_MIN_SCORE is owned by ragkit.config (imported above);
# WI-1 evidence gate (2026-09-12): threshold locked at 0.0 — every tested
# threshold regressed correctness (PLAN §6.5.2).

# AGENT_JUDGE_MODEL and AGENT_JUDGE_SCORE_FLOOR are owned by ragkit.config
# (imported above).  SCORE_FLOOR: score-floor sanity backstop, disabled at 0.0.

# AGENT_GATE_LONG_THRESHOLD is owned by ragkit.config (imported above): the
# word-count trigger wired into the heuristic query classifier when no
# explicit threshold is passed.

# --- Phase 3: GitHub tool (SPEC §5, PLAN §4) ---
# GITHUB_PAT / GITHUB_API_BASE / GITHUB_OWNER / GITHUB_REPO are owned by
# ragkit.config (imported above).  An empty PAT disables the tool (never
# logged); it is never a hard-required secret (no ``_require``).

# --- Phase 6: Code generation / validation (SPEC §8, PLAN §7) ---
# CODE_ROUTE_ENABLED / CODE_VALIDATE_MAX_TURNS / CODE_INTENT_PHRASES are owned
# by ragkit.config (imported above) — the same env keys and defaults DocPilot
# used to define here, now single-sourced in the framework.

# --- Phase 5: API + UI (SPEC §7) ---
DOCPILOT_DB_PATH: str = os.getenv("DOCPILOT_DB_PATH", "data/docpilot.sqlite3")
"""SQLite file backing the API session/chat history (sessions + messages).
Defaults under ``data/`` which is git-ignored — the DB is never committed."""
