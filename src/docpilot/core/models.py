from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Source "kind" tags (SPEC §7 amendment 2026-09-09 — citation-gold hardening).
# The generator prompt uses these to prefer the canonical file when several
# offered sources cover the same claim (FastAPI docs duplicate content across
# tutorial / advanced / how-to / index pages).  ``github`` marks live GitHub
# evidence from the Phase 3 tool.
SOURCE_KIND_TUTORIAL = "tutorial"
SOURCE_KIND_ADVANCED = "advanced"
SOURCE_KIND_HOWTO = "how-to"
SOURCE_KIND_REFERENCE = "reference"
SOURCE_KIND_INDEX = "index"
SOURCE_KIND_OTHER = "other"
SOURCE_KIND_GITHUB = "live"

# Path segments that distinguish doc "kinds" (FastAPI corpus layout:
# en/docs/tutorial/..., en/docs/advanced/..., en/docs/how-to/...,
# en/docs/reference/..., en/docs/index.md).
_KIND_BY_SEGMENT: dict[str, str] = {
    "tutorial": SOURCE_KIND_TUTORIAL,
    "advanced": SOURCE_KIND_ADVANCED,
    "how-to": SOURCE_KIND_HOWTO,
    "reference": SOURCE_KIND_REFERENCE,
}


def derive_source_kind(source_file: str) -> str:
    """Classify a ``source_file`` path into a citation ``kind``.

    ``github:{owner}/{repo}...`` refs (Phase 3 tool evidence) → ``"live"``;
    FastAPI doc paths map their first section segment to the matching kind
    (``en/docs/tutorial/...`` → ``tutorial``); an ``index`` file directly under
    a docs root → ``"index"``; anything else → ``"other"``.  Never raises —
    garbage paths fall back to ``"other"``.
    """
    if not source_file:
        return SOURCE_KIND_OTHER
    if source_file.startswith("github:"):
        return SOURCE_KIND_GITHUB
    parts = source_file.replace("\\", "/").split("/")
    for seg in parts:
        kind = _KIND_BY_SEGMENT.get(seg)
        if kind is not None:
            return kind
    if parts and parts[-1].lower() == "index.md":
        return SOURCE_KIND_INDEX
    return SOURCE_KIND_OTHER


def source_language(source_file: str) -> str:
    """Derive the document language from a ``source_file`` path.

    Corpus paths look like ``en/docs/...`` or ``zh-hant/docs/...`` where the
    first path segment is the language tag. Paths without a language segment
    (e.g. ``CORPUS.md`` at the corpus root) default to ``en``.
    """
    if "/" in source_file:
        return source_file.split("/", 1)[0]
    return "en"


@dataclass
class Document:
    """A raw documentation file (e.g. .md/.mdx) loaded from the corpus."""
    file_path: str
    content: str
    frontmatter: dict[str, Any] | None = None


@dataclass
class Chunk:
    """A semantic chunk extracted from a Document."""
    id: str
    content: str
    heading_path: str | None = None
    source_file: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverResult:
    """A single retrieval result with its cosine similarity score."""
    chunk: Chunk
    score: float


@dataclass
class SourceRef:
    """A citation reference (inline [n] + footer source line).

    ``kind`` (SPEC §7 amendment 2026-09-09) classifies the file for the
    generator's source-preference rule — e.g. ``tutorial`` vs ``index`` —
    derived via :func:`derive_source_kind` from the file path by callers
    (pipeline_ask, the agent graph).  ``None`` means "not classified" (footer
    and prompt rendering fall back to the pre-Phase-5 format).
    """
    ref: int
    file: str
    heading: str | None = None
    kind: str | None = None
