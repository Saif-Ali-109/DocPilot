"""CitationEngine interface and standard implementation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from docpilot.core.models import SourceRef


class CitationEngine(ABC):
    """Interface for formatting answers with inline citation markers."""

    @abstractmethod
    def format_answer(self, raw_answer: str, sources: list[SourceRef]) -> tuple[str, str]:
        """Format an LLM answer with citation markers.

        Returns:
            (answer_with_markers, footer_string)
        """
        ...


class StandardCitationEngine(CitationEngine):
    """Formats LLM answers by validating [N] citation markers against
    provided sources and building a source footer.

    Design notes (lenient-drop):
        - Inline markers [N] in the LLM answer are preserved as-is — the
          engine never renumbers or fabricates them.
        - Any [N] whose N does not correspond to a valid source index is
          silently dropped from the footer. The answer text itself is left
          untouched (we do not strip invalid markers from the body either,
          since the LLM's phrasing around them would break).
        - If the LLM answer contains *no* citation markers at all, the
          footer is an empty string (we never fabricate citations).
    """

    # Matches [N] where N is a positive integer
    _MARKER_RE = re.compile(r"\[(\d+)\]")

    def format_answer(self, raw_answer: str, sources: list[SourceRef]) -> tuple[str, str]:
        """Validate citation markers and build a source footer.

        Args:
            raw_answer: The LLM-generated answer text (may contain [1], [2], ...).
            sources: The list of SourceRef objects available for citation.

        Returns:
            A tuple of (answer_with_markers, footer_string).
            footer_string is an empty string when no valid markers are found.
        """
        if not raw_answer:
            return (raw_answer, "")

        # Build a lookup of valid ref numbers
        valid_refs = {s.ref for s in sources}

        # Collect the set of ref numbers actually mentioned in the answer
        mentioned = set()
        for match in self._MARKER_RE.finditer(raw_answer):
            mentioned.add(int(match.group(1)))

        if not mentioned:
            # No citation markers at all — return as-is with empty footer
            return (raw_answer, "")

        # Only include sources whose ref was actually mentioned AND is valid
        source_map = {s.ref: s for s in sources}
        footer_lines: list[str] = []
        for s in sources:
            if s.ref in mentioned and s.ref in valid_refs:
                heading_part = f" → {s.heading}" if s.heading else ""
                footer_lines.append(f"[{s.ref}] {s.file}{heading_part}")

        footer = "Sources:\n" + "\n".join(footer_lines) if footer_lines else ""
        return (raw_answer, footer)
