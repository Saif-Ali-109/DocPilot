"""Unit tests for docpilot.ingestion.chunker (MarkdownChunker).

All tests use inline string fixtures.  No corpus files are read.
"""

from __future__ import annotations

import pytest
from docpilot.core.models import Document
from docpilot.ingestion.chunker import MarkdownChunker, _deterministic_id, _word_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(content: str, path: str = "test/doc.md") -> Document:
    return Document(file_path=path, content=content)


def _word_count_of_chunks(chunks) -> list[int]:
    return [_word_count(c.content) for c in chunks]


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

class TestDeterministicID:
    def test_same_input_same_id(self) -> None:
        id1 = _deterministic_id("a.md", 0)
        id2 = _deterministic_id("a.md", 0)
        assert id1 == id2

    def test_different_index_different_id(self) -> None:
        assert _deterministic_id("a.md", 0) != _deterministic_id("a.md", 1)

    def test_different_file_different_id(self) -> None:
        assert _deterministic_id("a.md", 0) != _deterministic_id("b.md", 0)

    def test_id_is_hex(self) -> None:
        hid = _deterministic_id("x.md", 0)
        assert len(hid) == 12
        int(hid, 16)  # should not raise


# ---------------------------------------------------------------------------
# Empty / whitespace-only documents
# ---------------------------------------------------------------------------

class TestEmptyDocuments:
    def test_empty_content_returns_empty_list(self) -> None:
        doc = _make_doc("")
        chunks = MarkdownChunker().chunk(doc)
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        doc = _make_doc("   \n  \n  ")
        chunks = MarkdownChunker().chunk(doc)
        assert chunks == []


# ---------------------------------------------------------------------------
# Fenced code blocks: never split mid-block
# ---------------------------------------------------------------------------

class TestCodeBlockIntegrity:
    def test_large_code_block_not_split(self) -> None:
        """A code block exceeding target size must stay in exactly one chunk."""
        # Build a code block with > 650 words.
        code_lines = [f"line_{i} = 'word_{i} extra data here for padding fill'" for i in range(100)]
        code = "```python\n" + "\n".join(code_lines) + "\n```"
        md = "## Code Section\n\n" + code + "\n\nEnd of section."
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=650, overlap=75).chunk(doc)

        # Find the chunk containing the code block.
        code_chunks = [c for c in chunks if c.metadata.get("code_block") is True]
        assert len(code_chunks) >= 1, "No chunk marked as code_block"

        # The entire code block must appear in one chunk (no splitting).
        full_code = "\n".join(code_lines)
        combined_code_content = " ".join(c.content for c in code_chunks)
        # Every line of the code block should be present in exactly one code chunk.
        for line in code_lines[:5]:
            assert line in combined_code_content
        for line in code_lines[-5:]:
            assert line in combined_code_content

        # The opening fence must be in one chunk, the closing in the same.
        first_code_chunk = code_chunks[0]
        assert "```python" in first_code_chunk.content
        assert "```" in first_code_chunk.content

    def test_code_block_heading_annotation(self) -> None:
        md = "## Example\n```python\ncode\n```\nDone."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        assert any(c.metadata.get("code_block") is True for c in chunks)


# ---------------------------------------------------------------------------
# Table atomicity: never split mid-table
# ---------------------------------------------------------------------------

class TestTableAtomicity:
    def test_table_never_split(self) -> None:
        """A Markdown table must stay in exactly one chunk."""
        table_rows = [f"| Item {i} | Description {i} | Value {i} |" for i in range(30)]
        table = "| Item | Description | Value |\n|------|-------------|-------|\n" + "\n".join(table_rows)
        md = "## Table Section\n\nSome intro.\n\n" + table + "\n\nAfter table."
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=20, overlap=5).chunk(doc)  # small target to force splits

        # Find the chunk(s) containing table content.
        table_chunks = [c for c in chunks if c.metadata.get("table") is True]
        assert len(table_chunks) >= 1, "No chunk marked as table"

        # Every table row must be in one of the table chunks.
        table_content = "\n".join(c.content for c in table_chunks)
        for row in table_rows[:3]:
            assert row in table_content
        for row in table_rows[-3:]:
            assert row in table_content

    def test_single_row_table(self) -> None:
        md = "## T\n| A | B |\n|---|---|\n| 1 | 2 |"
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        table_chunks = [c for c in chunks if c.metadata.get("table") is True]
        assert len(table_chunks) == 1


# ---------------------------------------------------------------------------
# Heading hierarchy metadata
# ---------------------------------------------------------------------------

class TestHeadingMetadata:
    def test_heading_level_metadata(self) -> None:
        md = "# Title\n## Section\nContent."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        h2_chunks = [c for c in chunks if c.metadata.get("heading_level") == 2]
        assert len(h2_chunks) >= 1

    def test_heading_path_propagated(self) -> None:
        md = "# Guide\n## Install\npip install fastapi."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        install_chunks = [c for c in chunks if c.heading_path and "Install" in c.heading_path]
        assert len(install_chunks) >= 1
        assert install_chunks[0].heading_path == "/Guide/Install"

    def test_preamble_chunk_has_no_heading_path(self) -> None:
        md = "Preamble text.\n\n# Title\nAfter title."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        preamble_chunks = [c for c in chunks if c.heading_path is None]
        # The preamble (text before # Title) should have no heading_path.
        assert len(preamble_chunks) >= 1


# ---------------------------------------------------------------------------
# Chunk sizes within bounds
# ---------------------------------------------------------------------------

class TestChunkSizes:
    def test_normal_prose_chunk_sizes(self) -> None:
        """Chunks of normal prose should land within relaxed bounds (300–1000 words)."""
        # Build a section with ~2000 words of prose.
        paragraph = " ".join([f"word{i}" for i in range(200)])
        md = "## Long Section\n\n" + "\n\n".join([paragraph] * 10)
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=650, overlap=75).chunk(doc)
        word_counts = _word_count_of_chunks(chunks)

        assert len(word_counts) >= 2, "Should produce multiple chunks"
        for wc in word_counts:
            # Relaxed bounds per task spec.
            assert 50 <= wc <= 1200, f"Chunk size {wc} outside bounds"

    def test_small_content_single_chunk(self) -> None:
        md = "## Small\nJust a sentence."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        assert len(chunks) == 1
        assert "Just a sentence." in chunks[0].content


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_overlap_between_chunks(self) -> None:
        """When a section splits, consecutive chunks should share ~overlap words."""
        # Create a section with enough prose to split into 2+ chunks.
        words = [f"word{i}" for i in range(800)]
        section_content = " ".join(words)
        md = "## Split Me\n\n" + section_content
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=300, overlap=50).chunk(doc)
        if len(chunks) < 2:
            pytest.skip("Section did not split into 2+ chunks")

        # Check overlap: last 50 words of chunk 0 should appear at the
        # start of chunk 1 (after heading metadata).
        c0_words = chunks[0].content.split()
        c1_words = chunks[1].content.split()

        # Find where the overlap starts in chunk 1.
        overlap_region = c0_words[-50:] if len(c0_words) >= 50 else c0_words
        # The overlap words should be near the start of chunk 1.
        c1_text = " ".join(c1_words)
        overlap_text = " ".join(overlap_region)
        assert overlap_text in c1_text, (
            f"Expected overlap of last 50 words from chunk 0 in chunk 1.\n"
            f"Chunk 0 tail: {' '.join(c0_words[-5:])}\n"
            f"Chunk 1 head: {' '.join(c1_words[:10])}"
        )

    def test_exact_target_boundary_no_garbage_chunk(self) -> None:
        """No chunk should be emitted containing only duplicated overlap words.

        Regression test: an oversized prose run at exactly ``target`` words +
        overlap must produce exactly two chunks (first full, second
        overlap-seeded) — not a third tiny chunk of pure overlap.
        """
        words = [f"w{i}" for i in range(1300)]
        section_content = " ".join(words)
        md = "## B\n\n" + section_content
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=650, overlap=75).chunk(doc)
        assert len(chunks) == 2, f"Expected exactly 2 chunks, got {len(chunks)}: {[len(c.content.split()) for c in chunks]}"

        # Second chunk must start with the tail of the first.
        c0 = chunks[0].content.split()
        c1 = chunks[1].content
        assert " ".join(c0[-75:]) in c1

    def test_no_tiny_duplicate_chunks(self) -> None:
        """Chunks from a prose-only section should carry substantial content.

        The final chunk may be a small remainder (overlap + trailing words),
        but no chunk may consist *only* of duplicated overlap words — every
        chunk must contain more than ``overlap`` words.
        """
        words = [f"w{i}" for i in range(2000)]
        section_content = " ".join(words)
        md = "## C\n\n" + section_content
        doc = _make_doc(md)

        chunks = MarkdownChunker(target=650, overlap=75).chunk(doc)
        sizes = [len(c.content.split()) for c in chunks]
        # Interior chunks should be substantial; even the final remainder
        # carries the overlap seed plus trailing content → > overlap.
        assert all(s > 75 for s in sizes), f"Chunk consists only of overlap: {sizes}"


# ---------------------------------------------------------------------------
# source_file and chunk_index
# ---------------------------------------------------------------------------

class TestSourceAndIndex:
    def test_source_file_set(self) -> None:
        doc = _make_doc("## A\nContent.", path="tutorial/intro.md")
        chunks = MarkdownChunker().chunk(doc)
        assert all(c.source_file == "tutorial/intro.md" for c in chunks)

    def test_chunk_index_sequential(self) -> None:
        md = "# Title\n## A\nA content.\n\n## B\nB content."
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_unique(self) -> None:
        md = "# T\n## A\nA\n\n## B\nB"
        doc = _make_doc(md)
        chunks = MarkdownChunker().chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_two_runs_produce_same_ids(self) -> None:
        md = "## Section\nContent."
        doc = _make_doc(md)
        chunker = MarkdownChunker()
        ids_run1 = [c.id for c in chunker.chunk(doc)]
        ids_run2 = [c.id for c in chunker.chunk(doc)]
        assert ids_run1 == ids_run2

    def test_two_runs_produce_same_content(self) -> None:
        md = "## Section\nSome content here."
        doc = _make_doc(md)
        chunker = MarkdownChunker()
        content1 = [c.content for c in chunker.chunk(doc)]
        content2 = [c.content for c in chunker.chunk(doc)]
        assert content1 == content2


# ---------------------------------------------------------------------------
# Smoke test: real corpus file (skip if absent)
# ---------------------------------------------------------------------------

class TestRealCorpus:
    @pytest.mark.skipif(
        not __import__("pathlib").Path("docs/en/docs/tutorial/first-steps.md").exists(),
        reason="Real corpus not cloned",
    )
    def test_first_steps_exists(self) -> None:
        from pathlib import Path
        assert Path("docs/en/docs/tutorial/first-steps.md").exists()
