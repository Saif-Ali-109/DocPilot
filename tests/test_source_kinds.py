"""Tests for the Phase 5 source-kind metadata (SPEC §7 amendment — citation-gold
hardening): derive_source_kind classification, SourceRef.kind default, the
prompt rendering of kinds in format_sources, and the new SYSTEM_PROMPT rule 9.
"""

from __future__ import annotations

from docpilot.core.models import (
    SOURCE_KIND_ADVANCED,
    SOURCE_KIND_GITHUB,
    SOURCE_KIND_HOWTO,
    SOURCE_KIND_INDEX,
    SOURCE_KIND_OTHER,
    SOURCE_KIND_REFERENCE,
    SOURCE_KIND_TUTORIAL,
    SourceRef,
    derive_source_kind,
)
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources


class TestDeriveSourceKind:
    def test_tutorial(self) -> None:
        assert (
            derive_source_kind("en/docs/tutorial/path-params.md")
            == SOURCE_KIND_TUTORIAL
        )

    def test_advanced(self) -> None:
        assert (
            derive_source_kind("en/docs/advanced/using-request-directly.md")
            == SOURCE_KIND_ADVANCED
        )

    def test_howto_anywhere_in_path(self) -> None:
        assert derive_source_kind("en/docs/how-to/contribute.md") == SOURCE_KIND_HOWTO

    def test_reference(self) -> None:
        assert (
            derive_source_kind("en/docs/reference/fastapi.md") == SOURCE_KIND_REFERENCE
        )

    def test_index_file(self) -> None:
        assert derive_source_kind("en/docs/index.md") == SOURCE_KIND_INDEX

    def test_other_plain_path(self) -> None:
        assert derive_source_kind("docs/CORPUS.md") == SOURCE_KIND_OTHER

    def test_github_live_ref(self) -> None:
        assert (
            derive_source_kind("github:fastapi/fastapi#12345") == SOURCE_KIND_GITHUB
        )

    def test_empty_and_garbage_fall_back_to_other(self) -> None:
        assert derive_source_kind("") == SOURCE_KIND_OTHER
        assert derive_source_kind("not a path>") == SOURCE_KIND_OTHER

    def test_windows_separators_normalised(self) -> None:
        assert (
            derive_source_kind("en\\docs\\tutorial\\first-steps.md")
            == SOURCE_KIND_TUTORIAL
        )


class TestSourceRefKind:
    def test_defaults_to_none(self) -> None:
        assert SourceRef(ref=1, file="a.md").kind is None

    def test_explicit_kind_round_trips(self) -> None:
        ref = SourceRef(ref=1, file="en/docs/index.md", kind=SOURCE_KIND_INDEX)
        assert ref.kind == SOURCE_KIND_INDEX


class TestFormatSourcesKind:
    def test_kind_rendered_in_parentheses(self) -> None:
        out = format_sources(
            [
                SourceRef(ref=1, file="en/docs/tutorial/a.md", kind=SOURCE_KIND_TUTORIAL),
                SourceRef(ref=2, file="en/docs/index.md", kind=SOURCE_KIND_INDEX),
            ]
        )
        assert out == (
            "[1] en/docs/tutorial/a.md (tutorial)\n"
            "[2] en/docs/index.md (index)"
        )

    def test_kind_with_heading(self) -> None:
        out = format_sources(
            [
                SourceRef(
                    ref=1,
                    file="en/docs/tutorial/a.md",
                    heading="Installation",
                    kind=SOURCE_KIND_TUTORIAL,
                )
            ]
        )
        assert out == "[1] en/docs/tutorial/a.md (tutorial) → Installation"

    def test_none_kind_keeps_phase1_format(self) -> None:
        # Unclassified sources render byte-identically to Phase 1–4.
        out = format_sources(
            [SourceRef(ref=1, file="docs/a.md", heading="Installation")]
        )
        assert out == "[1] docs/a.md → Installation"


class TestSystemPromptRule9:
    def test_source_preference_rule_present(self) -> None:
        assert "prefer tutorial / advanced / reference" in SYSTEM_PROMPT
        assert "most specific" in SYSTEM_PROMPT