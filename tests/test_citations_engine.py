"""Tests for citations/engine.py — StandardCitationEngine."""

from docpilot.citations.engine import CitationEngine, StandardCitationEngine
from docpilot.core.models import SourceRef


class TestStandardCitationEngine:
    """Validate marker validation, footer construction, and edge cases."""

    def setup_method(self) -> None:
        self.engine = StandardCitationEngine()
        self.sources = [
            SourceRef(ref=1, file="docs/getting-started.md", heading="Installation"),
            SourceRef(ref=2, file="docs/tutorial/first-steps.md", heading="First Steps"),
        ]

    def test_answer_with_valid_markers(self) -> None:
        raw = "Install with pip. [1] Then follow the tutorial. [2]"
        answer, footer = self.engine.format_answer(raw, self.sources)
        assert answer == raw
        assert "[1] docs/getting-started.md → Installation" in footer
        assert "[2] docs/tutorial/first-steps.md → First Steps" in footer
        assert footer.startswith("Sources:")

    def test_out_of_range_marker_dropped_from_footer(self) -> None:
        """[3] is referenced in the answer but only 2 sources exist — it
        must NOT appear in the footer, but [1] and [2] should still be there."""
        raw = "Good info. [1] But also [3] is weird."
        answer, footer = self.engine.format_answer(raw, self.sources)
        assert answer == raw  # body left as-is
        assert "[1] docs/getting-started.md → Installation" in footer
        assert "[2]" not in footer  # [2] not mentioned in answer → not in footer
        assert "[3]" not in footer  # [3] is out of range → dropped

    def test_empty_answer(self) -> None:
        answer, footer = self.engine.format_answer("", self.sources)
        assert answer == ""
        assert footer == ""

    def test_no_markers_in_answer(self) -> None:
        raw = "Just a plain answer with no citations."
        answer, footer = self.engine.format_answer(raw, self.sources)
        assert answer == raw
        assert footer == ""

    def test_only_valid_mentioned_sources_in_footer(self) -> None:
        """If answer mentions [2] but not [1], only [2] should appear in footer."""
        raw = "Follow the tutorial. [2]"
        answer, footer = self.engine.format_answer(raw, self.sources)
        assert answer == raw
        assert "[2] docs/tutorial/first-steps.md → First Steps" in footer
        assert "[1]" not in footer

    def test_source_without_heading(self) -> None:
        sources = [SourceRef(ref=1, file="docs/a.md", heading=None)]
        raw = "See docs. [1]"
        answer, footer = self.engine.format_answer(raw, sources)
        assert answer == raw
        assert "[1] docs/a.md" in footer
        assert "→" not in footer

    def test_footer_starts_with_sources_header(self) -> None:
        raw = "Info [1]"
        _, footer = self.engine.format_answer(raw, self.sources)
        assert footer.startswith("Sources:\n")


class TestFakeGenerator:
    """Verify a minimal fake Generator subclass works through the interface
    (no network, no real LLM)."""

    def test_fake_generator_through_interface(self) -> None:
        from docpilot.generation.generator import Generator

        class FakeGenerator(Generator):
            def generate(self, prompt: str) -> str:
                return f"fake answer for: {prompt[:20]}..."

        gen: Generator = FakeGenerator()
        result = gen.generate("What is FastAPI?")
        assert result.startswith("fake answer for:")

    def test_citation_engine_independently_usable(self) -> None:
        engine = StandardCitationEngine()
        answer, footer = engine.format_answer("hello", [])
        assert answer == "hello"
        assert footer == ""
