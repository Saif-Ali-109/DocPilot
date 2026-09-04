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
