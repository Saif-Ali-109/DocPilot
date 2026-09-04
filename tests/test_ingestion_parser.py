"""Unit tests for docpilot.ingestion.parser (MarkdownParser).

All tests use inline string fixtures.  No corpus files are read.
"""

from __future__ import annotations

import pytest
from docpilot.core.models import Document
from docpilot.ingestion.parser import MarkdownParser, SectionBlock, _strip_frontmatter


# ---------------------------------------------------------------------------
# Front-matter stripping
# ---------------------------------------------------------------------------

class TestStripFrontmatter:
    def test_yaml_frontmatter_stripped(self) -> None:
        raw = "---\ntitle: Test\n---\n# Heading\nBody"
        result = _strip_frontmatter(raw)
        assert result == "# Heading\nBody"
        assert "title" not in result

    def test_no_frontmatter(self) -> None:
        raw = "# Heading\nBody"
        assert _strip_frontmatter(raw) == raw

    def test_only_opening_fence_not_stripped(self) -> None:
        """A line that is just '---' but not at the very start-of-file should not be treated."""
        raw = "# Intro\n\n---\n\n## Section"
        assert _strip_frontmatter(raw) == raw

    def test_frontmatter_with_subsequent_content(self) -> None:
        raw = "---\nkey: val\n---\nRest of the document"
        result = _strip_frontmatter(raw)
        assert result == "Rest of the document"


# ---------------------------------------------------------------------------
# Heading → heading_path mapping
# ---------------------------------------------------------------------------

class TestHeadingPath:
    def test_h1_produces_root_path(self) -> None:
        doc = Document(file_path="x.md", content="# Getting Started\nBody here.")
        parser = MarkdownParser()
        sections = parser.parse(doc)
        # First section is preamble (empty content before H1).
        # Second section is H1.
        h1_section = [s for s in sections if s.heading_level == 1][0]
        assert h1_section.heading_path == "/Getting Started"

    def test_nested_headings(self) -> None:
        md = "# Guide\n## Install\n### pip\nContent here."
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        pip_section = [s for s in sections if s.heading_text == "pip"][0]
        assert pip_section.heading_path == "/Guide/Install/pip"

    def test_siblings_replace(self) -> None:
        """Two H2s at the same level: /Guide/A then /Guide/B."""
        md = "# Guide\n## A\nA content\n## B\nB content"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        paths = [s.heading_path for s in sections if s.heading_level == 2]
        assert "/Guide/A" in paths
        assert "/Guide/B" in paths
        # Ensure /Guide/A is NOT a prefix of /Guide/B path.
        assert "/Guide/A" not in [p for p in paths if p != "/Guide/A"]

    def test_heading_path_has_no_trailing_slash(self) -> None:
        md = "# Root\n## Sub\nContent"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        for s in sections:
            assert not s.heading_path.endswith("/"), f"Trailing slash in: {s.heading_path}"

    def test_fastapi_heading_attributes_stripped(self) -> None:
        """FastAPI uses `# First Steps { #first-steps }` — attributes are stripped."""
        md = "# First Steps { #first-steps }\nBody"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        h1 = [s for s in sections if s.heading_level == 1][0]
        assert h1.heading_text == "First Steps"
        assert h1.heading_path == "/First Steps"

    def test_plain_prose_braces_kept(self) -> None:
        """A heading whose braces are prose (e.g. {like this}) is untouched."""
        md = "## Using { placeholders }\nBody"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        h2 = [s for s in sections if s.heading_level == 2][0]
        assert h2.heading_text == "Using { placeholders }"


# ---------------------------------------------------------------------------
# Fenced code blocks preserved (not treated as headings)
# ---------------------------------------------------------------------------

class TestCodeBlockPreservation:
    def test_heading_inside_fence_not_a_heading(self) -> None:
        md = "```python\n# This is a comment, not a heading\nprint('hi')\n```\nReal heading below\n## Real"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        # "Real" should be the only H2 heading.
        h2s = [s for s in sections if s.heading_level == 2]
        assert len(h2s) == 1
        assert h2s[0].heading_text == "Real"

    def test_fence_block_has_code_flag(self) -> None:
        md = "## Code\n```python\nx = 1\n```\nDone."
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        code_section = [s for s in sections if s.heading_text == "Code"][0]
        assert code_section.has_code is True

    def test_fastapi_annotate_fence(self) -> None:
        md = "## Example\n```python\n{ .python .annotate }\ndef foo():\n    pass\n```\nAfter."
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        example_section = [s for s in sections if s.heading_text == "Example"][0]
        assert example_section.has_code is True

    def test_tilde_fence(self) -> None:
        md = "## Note\n~~~python\ncode\n~~~\nEnd."
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        note_section = [s for s in sections if s.heading_text == "Note"][0]
        assert note_section.has_code is True


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

class TestTableDetection:
    def test_table_flag_set(self) -> None:
        md = "## Data\n| Col1 | Col2 |\n|------|------|\n| a    | b    |"
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        data_section = [s for s in sections if s.heading_text == "Data"][0]
        assert data_section.has_table is True

    def test_no_table_flag_without_table(self) -> None:
        md = "## Plain\nJust text."
        doc = Document(file_path="x.md", content=md)
        sections = MarkdownParser().parse(doc)
        plain = [s for s in sections if s.heading_text == "Plain"][0]
        assert plain.has_table is False


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_document(self) -> None:
        doc = Document(file_path="empty.md", content="")
        sections = MarkdownParser().parse(doc)
        assert len(sections) == 1
        assert sections[0].heading_level == 0
        assert sections[0].content == [""]

    def test_whitespace_only(self) -> None:
        doc = Document(file_path="ws.md", content="   \n  \n  ")
        sections = MarkdownParser().parse(doc)
        assert len(sections) == 1

    def test_no_headings(self) -> None:
        md = "Paragraph one.\n\nParagraph two."
        doc = Document(file_path="p.md", content=md)
        sections = MarkdownParser().parse(doc)
        assert len(sections) == 1
        assert sections[0].heading_level == 0
        assert sections[0].heading_path == ""

    def test_frontmatter_then_headings(self) -> None:
        md = "---\ntitle: X\n---\n# Intro\nBody"
        doc = Document(file_path="fm.md", content=md)
        sections = MarkdownParser().parse(doc)
        h1 = [s for s in sections if s.heading_level == 1]
        assert len(h1) == 1
        assert h1[0].heading_text == "Intro"


# ---------------------------------------------------------------------------
# SectionBlock type
# ---------------------------------------------------------------------------

class TestSectionBlock:
    def test_default_fields(self) -> None:
        sb = SectionBlock()
        assert sb.heading_text == ""
        assert sb.heading_level == 0
        assert sb.heading_path == ""
        assert sb.content == []
        assert sb.has_code is False
        assert sb.has_table is False
