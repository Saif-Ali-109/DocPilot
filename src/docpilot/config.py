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

AGENT_GATE_LONG_THRESHOLD: int = int(os.getenv("AGENT_GATE_LONG_THRESHOLD", "18"))
"""Word-count gate trigger.

NOT YET WIRED — wired in coordinator pass. ``HeuristicQueryClassifier``
(agent/gate.py) currently uses the fixed module constant ``SIMPLE_WORD_LIMIT``
and does not expose a constructor/parameter knob, so this key is defined here
for forward-compatibility but is not consumed anywhere yet. Do NOT edit gate.py.
"""

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

# --- Phase 5: API + UI (SPEC §7) ---
DOCPILOT_DB_PATH: str = os.getenv("DOCPILOT_DB_PATH", "data/docpilot.sqlite3")
"""SQLite file backing the API session/chat history (sessions + messages).
Defaults under ``data/`` which is git-ignored — the DB is never committed."""
