"""Tests for generation/prompts.py — prompt template and format_sources."""

from docpilot.core.models import SourceRef
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources


class TestSystemPrompt:
    """Verify the SYSTEM_PROMPT matches SPEC.md §3.9."""

    def test_contains_rules_header(self) -> None:
        assert "RULES:" in SYSTEM_PROMPT

    def test_rule_1_keyword(self) -> None:
        assert "ONLY the provided context" in SYSTEM_PROMPT

    def test_rule_2_keyword(self) -> None:
        assert "I don't know" in SYSTEM_PROMPT

    def test_rule_3_keyword(self) -> None:
        assert "Cite every claim" in SYSTEM_PROMPT

    def test_rule_4_keyword(self) -> None:
        assert "Never invent citations" in SYSTEM_PROMPT

    def test_rule_5_keyword(self) -> None:
        assert "technically accurate" in SYSTEM_PROMPT

    def test_rule_6_keyword(self) -> None:
        assert "Clearly distinguish" in SYSTEM_PROMPT

    def test_context_placeholder(self) -> None:
        assert "{context}" in SYSTEM_PROMPT

    def test_sources_placeholder(self) -> None:
        assert "{sources}" in SYSTEM_PROMPT

    def test_question_placeholder(self) -> None:
        assert "{question}" in SYSTEM_PROMPT

    def test_template_substitution(self) -> None:
        filled = SYSTEM_PROMPT.format(
            context="chunk content",
            sources="[1] docs/a.md",
            question="How do I install?",
        )
        assert "chunk content" in filled
        assert "[1] docs/a.md" in filled
        assert "How do I install?" in filled
        assert "{context}" not in filled
        assert "{sources}" not in filled
        assert "{question}" not in filled


class TestFormatSources:
    """Verify format_sources output format."""

    def test_single_source_with_heading(self) -> None:
        sources = [SourceRef(ref=1, file="docs/getting-started.md", heading="Installation")]
        result = format_sources(sources)
        assert result == "[1] docs/getting-started.md → Installation"

    def test_single_source_no_heading(self) -> None:
        sources = [SourceRef(ref=1, file="docs/a.md", heading=None)]
        result = format_sources(sources)
        assert result == "[1] docs/a.md"

    def test_multiple_sources(self) -> None:
        sources = [
            SourceRef(ref=1, file="docs/getting-started.md", heading="Installation"),
            SourceRef(ref=2, file="docs/tutorial/first-steps.md", heading="First Steps"),
        ]
        result = format_sources(sources)
        expected = (
            "[1] docs/getting-started.md → Installation\n"
            "[2] docs/tutorial/first-steps.md → First Steps"
        )
        assert result == expected

    def test_empty_sources_list(self) -> None:
        result = format_sources([])
        assert result == ""

    def test_arrow_character_is_unicode(self) -> None:
        """The arrow character used is the Unicode right-pointing arrow (→),
        consistent with SPEC.md §3.10 examples."""
        sources = [SourceRef(ref=1, file="docs/a.md", heading="X")]
        result = format_sources(sources)
        assert "→" in result
        assert "\u2192" in result  # same thing, explicit check
