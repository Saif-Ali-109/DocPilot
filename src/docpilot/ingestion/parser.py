"""Parser interface and Markdown/MDX parser for DocPilot ingestion."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from docpilot.core.models import Document


@dataclass
class SectionBlock:
    """Intermediate parsed section produced by a :class:`Parser`.

    Represents a heading-scoped section of a Markdown document along with
    metadata about the heading hierarchy and structural annotations (code
    blocks, tables) that the chunker will use for split decisions.

    Attributes:
        heading_text: The heading text without leading ``#`` markers.
        heading_level: Depth of the heading (1 = ``#``, 2 = ``##``, etc.).
        heading_path: Hierarchy path built from ancestors, e.g.
            ``"/Getting Started/Installation"`` (no trailing slash; ``/``
            separator). Empty string for the document root (H1 or preamble).
        content: Raw body lines of this section (excluding the heading line
            itself; blank lines, code fences, and table rows are preserved).
        has_code: ``True`` if the section body contains at least one fenced
            code block (opened by ```` ``` ```` or ``~~~``).
        has_table: ``True`` if the section body contains at least one
            Markdown table row (a line beginning with ``|``).
    """

    heading_text: str = ""
    heading_level: int = 0
    heading_path: str = ""
    content: list[str] = field(default_factory=list)
    has_code: bool = False
    has_table: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_frontmatter(text: str) -> str:
    """Remove YAML/TOML front matter delimited by ``---`` at the very top."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            rest = lines[i + 1:]
            return "\n".join(rest)
    return text


def _is_fence_opener(line: str) -> bool:
    """Return *True* if *line* is a fenced-code-block opening fence.

    Accepts ```` ``` ```` and ``~~~`` fences with optional attributes
    (e.g. ``{ .python .annotate }``).
    """
    stripped = line.strip()
    if stripped.startswith("```"):
        # Opening fence: must have at least ``` plus optional text.
        return len(stripped) > 3 or stripped == "```"
    if stripped.startswith("~~~"):
        return len(stripped) > 3 or stripped == "~~~"
    return False


def _is_fence_closer(line: str, fence_char: str = "`") -> bool:
    """Return *True* if *line* is a closing fence matching *fence_char*.

    A closing fence starts with the same character (`` `` ` ` `` or ``~~~``)
    and has **no** text after the marker (blank or whitespace only).
    """
    stripped = line.strip()
    if fence_char == "`":
        return stripped.startswith("```") and stripped[3:].strip() == ""
    else:
        return stripped.startswith("~~~") and stripped[3:].strip() == ""


def _is_heading(line: str) -> bool:
    """Return *True* if *line* is a Markdown heading (``# `` prefix)."""
    return bool(re.match(r"^#{1,6}\s", line))


def _parse_heading(line: str) -> tuple[int, str]:
    """Extract *(level, text)* from a heading line.

    FastAPI-docs attribute suffixes (e.g. ``{#first-steps}`` or
    ``{.python}``) are stripped from the heading text.

    >>> _parse_heading("## Getting Started")
    (2, 'Getting Started')
    >>> _parse_heading("## First Steps { #first-steps }")
    (2, 'First Steps')
    """
    match = re.match(r"^(#{1,6})\s+(.*)", line)
    if not match:
        return 0, ""
    return len(match.group(1)), _clean_heading_text(match.group(2).strip())


_HEADING_ATTR_SUFFIX = re.compile(r"\s*\{[^{}]*\}$")


def _clean_heading_text(text: str) -> str:
    """Strip trailing attribute-style spans such as ``{#anchor}`` or
    ``{.python}`` from a heading's text (e.g. ``First Steps { #first-steps }``).

    A trailing ``{...}`` span is removed only when its contents look like
    attribute tokens (``#anchor``, ``.class``, ``@directive`` or
    ``key="value"``) — plain prose braces are left untouched.
    """
    match = _HEADING_ATTR_SUFFIX.search(text)
    if not match:
        return text
    inner = text[match.start():].strip()[1:-1].strip()
    if not inner:
        return text
    tokens = inner.split()
    if not all(
        t.startswith(("#", ".", "@")) or ("=" in t) for t in tokens
    ):
        return text
    return text[:match.start()].strip()


# ---------------------------------------------------------------------------
# Parser ABC
# ---------------------------------------------------------------------------

class Parser(ABC):
    """Interface for converting a raw :class:`Document` into structured
    section blocks that a :class:`~docpilot.ingestion.chunker.Chunker` can
    consume."""

    @abstractmethod
    def parse(self, doc: Document) -> list[SectionBlock]:
        """Parse *doc* and return a list of :class:`SectionBlock` objects."""
        ...


# ---------------------------------------------------------------------------
# Markdown / MDX parser
# ---------------------------------------------------------------------------

class MarkdownParser(Parser):
    """Parse a Markdown/MDX :class:`Document` into heading-scoped sections.

    Parsing rules:
    * YAML/TOML front matter (``---`` delimited, file top) is stripped and
      stored in ``doc.frontmatter`` — it never becomes a heading.
    * Fenced code blocks (```` ``` ```` / ``~~~``, including FastAPI's
      ``{ .python .annotate }`` variants) are kept whole; heading-like
      lines inside fences are not treated as headings.
    * ``:``-prefixed admonition lines (``:::note``, ``:::tip``) are passed
      through without special handling; only ``#``-headings drive section
      boundaries.
    * ``heading_path`` is built by replacing ancestor entries at the same
      or deeper heading level — e.g. ``## A`` then ``## B`` produces
      ``"/B"`` (not ``"/A/B"``).
    """

    def parse(self, doc: Document) -> list[SectionBlock]:
        """Return heading-scoped :class:`SectionBlock` list for *doc*.

        Empty or whitespace-only content yields a single preamble block
        with ``heading_level == 0`` and empty content.
        """
        # ── strip front matter ────────────────────────────────────────────
        clean = _strip_frontmatter(doc.content)

        lines = clean.split("\n")
        sections: list[SectionBlock] = []
        pending: list[str] = []
        hierarchy: dict[int, str] = {}  # level → heading text

        in_fence = False
        fence_char = "`"  # ` or ~

        for line in lines:
            # ── inside a fenced code block ────────────────────────────────
            if in_fence:
                pending.append(line)
                if _is_fence_closer(line, fence_char):
                    in_fence = False
                continue

            # ── check for fence opener ────────────────────────────────────
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_char = "`" if stripped[0] == "`" else "~"
                in_fence = True
                pending.append(line)
                continue

            # ── check for heading ─────────────────────────────────────────
            if _is_heading(line):
                level, text = _parse_heading(line)
                sections.append(self._build_section(hierarchy, pending))
                hierarchy[level] = text
                for deeper in [k for k in hierarchy if k > level]:
                    del hierarchy[deeper]
                pending = []
                continue

            # ── ordinary line ─────────────────────────────────────────────
            pending.append(line)

        # ── flush final section ───────────────────────────────────────────
        # Always flush so a trailing heading (with no body) still produces a
        # section instead of silently vanishing.
        sections.append(self._build_section(hierarchy, pending))

        return sections

    # ------------------------------------------------------------------
    @staticmethod
    def _build_section(
        hierarchy: dict[int, str],
        lines: list[str],
    ) -> SectionBlock:
        """Build a :class:`SectionBlock` from the current hierarchy and lines."""
        ordered = [hierarchy[k] for k in sorted(hierarchy)]
        heading_path = "/" + "/".join(ordered) if ordered else ""
        top_level = max(hierarchy) if hierarchy else 0
        text = ordered[-1] if ordered else ""

        has_code = False
        has_table = False
        fence_open = False
        for line in lines:
            s = line.strip()
            if not fence_open:
                if s.startswith("```") or s.startswith("~~~"):
                    fence_open = True
                    has_code = True
                    continue
            else:
                if s.startswith("```") or s.startswith("~~~"):
                    fence_open = False
                continue
            if s.startswith("|"):
                has_table = True

        return SectionBlock(
            heading_text=text,
            heading_level=top_level,
            heading_path=heading_path,
            content=list(lines),
            has_code=has_code,
            has_table=has_table,
        )
