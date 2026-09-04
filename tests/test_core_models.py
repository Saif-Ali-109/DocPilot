"""Unit tests for docpilot.core.models shared dataclasses.

These dataclasses are the single-owner shared contract (AGENT B).
Every test uses plain inline values — no network, no DB, no corpus reads.
"""

from __future__ import annotations

import pytest
from docpilot.core.models import Chunk, Document, RetrieverResult, SourceRef


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_basic_construction(self) -> None:
        doc = Document(file_path="tutorial/intro.md", content="Hello")
        assert doc.file_path == "tutorial/intro.md"
        assert doc.content == "Hello"
        assert doc.frontmatter is None

    def test_with_frontmatter(self) -> None:
        fm = {"title": "Intro", "tags": ["quickstart"]}
        doc = Document(file_path="x.md", content="body", frontmatter=fm)
        assert doc.frontmatter == fm

    def test_empty_content(self) -> None:
        doc = Document(file_path="empty.md", content="")
        assert doc.content == ""


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_minimal_construction(self) -> None:
        c = Chunk(id="abc123", content="some text")
        assert c.id == "abc123"
        assert c.content == "some text"
        assert c.heading_path is None
        assert c.source_file == ""
        assert c.chunk_index == 0
        assert c.metadata == {}

    def test_metadata_default_is_shared_safe(self) -> None:
        """Default metadata dict is a fresh instance per Chunk (not shared)."""
        c1 = Chunk(id="a", content="x")
        c2 = Chunk(id="b", content="y")
        c1.metadata["key"] = "val"
        assert "key" not in c2.metadata

    def test_full_construction(self) -> None:
        c = Chunk(
            id="h1",
            content="chunk content",
            heading_path="/Intro/Setup",
            source_file="guide/setup.md",
            chunk_index=3,
            metadata={"code_block": True, "heading_level": 2},
        )
        assert c.heading_path == "/Intro/Setup"
        assert c.metadata["code_block"] is True
        assert c.chunk_index == 3


# ---------------------------------------------------------------------------
# RetrieverResult
# ---------------------------------------------------------------------------

class TestRetrieverResult:
    def test_construction(self) -> None:
        chunk = Chunk(id="r1", content="result text")
        rr = RetrieverResult(chunk=chunk, score=0.87)
        assert rr.chunk is chunk
        assert rr.score == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# SourceRef
# ---------------------------------------------------------------------------

class TestSourceRef:
    def test_minimal(self) -> None:
        sr = SourceRef(ref=1, file="docs/getting-started.md")
        assert sr.ref == 1
        assert sr.heading is None

    def test_with_heading(self) -> None:
        sr = SourceRef(ref=2, file="tutorial/first-steps.md", heading="First Steps")
        assert sr.heading == "First Steps"


# ---------------------------------------------------------------------------
# Type annotation checks
# ---------------------------------------------------------------------------

class TestTypeAnnotations:
    def test_document_fields_are_correct_types(self) -> None:
        doc = Document(file_path="a.md", content="b", frontmatter={"k": "v"})
        assert isinstance(doc.file_path, str)
        assert isinstance(doc.content, str)
        assert isinstance(doc.frontmatter, dict)

    def test_chunk_metadata_is_mutable_dict(self) -> None:
        c = Chunk(id="x", content="y")
        c.metadata["new"] = 42
        assert c.metadata["new"] == 42
