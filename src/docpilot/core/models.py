from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    """A citation reference (inline [n] + footer source line)."""
    ref: int
    file: str
    heading: str | None = None
