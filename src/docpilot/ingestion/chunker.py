"""Chunker interface and Markdown-aware semantic chunker for DocPilot."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from docpilot.core.models import Chunk, Document
from docpilot.ingestion.parser import SectionBlock

# ---------------------------------------------------------------------------
# Default chunking parameters (used when config is unavailable, e.g. in tests)
# ---------------------------------------------------------------------------
_DEFAULT_TARGET = 650   # words
_DEFAULT_OVERLAP = 75   # words


def _word_count(text: str) -> int:
    """Approximate token count via whitespace word splitting."""
    return len(text.split())


def _deterministic_id(source_file: str, chunk_index: int) -> str:
    """Return a deterministic, short hex identifier for a chunk.

    The ID is stable across runs: same *source_file* and *chunk_index*
    always produce the same hex string (first 12 hex chars of the
    SHA-256 of ``f"{source_file}#{chunk_index}"``).
    """
    key = f"{source_file}#{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Chunker ABC
# ---------------------------------------------------------------------------

class Chunker(ABC):
    """Interface for splitting a :class:`Document` into :class:`Chunk` objects.

    Implementations control how documents are segmented (semantic,
    fixed-size, recursive, etc.) while respecting structural invariants
    such as code-block and table integrity.
    """

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into a list of :class:`Chunk` objects."""
        ...


# ---------------------------------------------------------------------------
# Markdown-aware semantic chunker
# ---------------------------------------------------------------------------

class MarkdownChunker(Chunker):
    """Split a Markdown :class:`Document` into heading-scoped chunks.

    Splitting rules (per SPEC §3.5):
    * Fenced code blocks (```` ``` ```` / ``~~~``) are **never** split
      mid-block — a code block that exceeds the target size stays whole
      in its single chunk.
    * Markdown tables (consecutive ``|``-prefixed lines) are **never**
      split mid-table.
    * Target chunk size ≈ *target* words (default 650); overlap ≈
      *overlap* words (default 75) between consecutive chunks.
    * Metadata is set per chunk: ``code_block``, ``table``,
      ``heading_level``.
    * ``heading_path`` is inherited from the parent :class:`SectionBlock`.
    * ``source_file`` is the :class:`Document`'s ``file_path`` (relative
      to ``docs/``).
    * ``chunk_index`` is a 0-based counter within the document.
    * ``Chunk.id`` is deterministic (SHA-256 of ``source_file#chunk_index``).

    Empty or whitespace-only documents return an **empty list** — a
    deliberate choice: an empty document produces zero chunks, not a
    single empty chunk, so downstream ingestion never inserts
    content-less rows into the vector store.
    """

    def __init__(
        self,
        target: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.target = target if target is not None else _DEFAULT_TARGET
        self.overlap = overlap if overlap is not None else _DEFAULT_OVERLAP

    def chunk(self, doc: Document) -> list[Chunk]:
        """Return :class:`Chunk` list for *doc*.

        Uses :class:`~docpilot.ingestion.parser.MarkdownParser` internally
        to obtain heading-scoped sections, then further splits oversized
        sections while respecting code-block / table invariants.
        """
        from docpilot.ingestion.parser import MarkdownParser  # local to avoid cycles

        source_file = doc.file_path
        content = doc.content or ""
        if not content.strip():
            return []

        sections = MarkdownParser().parse(doc)

        all_chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections:
            section_text = "\n".join(section.content).strip()
            if not section_text:
                continue

            # Use a larger target when section is mostly code.
            wc = _word_count(section_text)
            effective_target = (
                self.target * 2
                if section.has_code
                else self.target
            )

            if wc <= effective_target:
                all_chunks.append(self._make_chunk(
                    content=section_text,
                    heading_path=section.heading_path,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    heading_level=section.heading_level,
                    has_code=section.has_code,
                    has_table=section.has_table,
                ))
                chunk_index += 1
            else:
                pieces = self._split_section(section_text, section)
                for piece in pieces:
                    piece_stripped = piece.strip()
                    if not piece_stripped:
                        continue
                    meta = self._detect_piece_meta(piece_stripped)
                    all_chunks.append(self._make_chunk(
                        content=piece_stripped,
                        heading_path=section.heading_path,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        heading_level=section.heading_level,
                        has_code=meta["code_block"],
                        has_table=meta["table"],
                    ))
                    chunk_index += 1

        return all_chunks

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        content: str,
        heading_path: str,
        source_file: str,
        chunk_index: int,
        heading_level: int,
        has_code: bool,
        has_table: bool,
    ) -> Chunk:
        return Chunk(
            id=_deterministic_id(source_file, chunk_index),
            content=content,
            heading_path=heading_path or None,
            source_file=source_file,
            chunk_index=chunk_index,
            metadata={
                "code_block": has_code,
                "table": has_table,
                "heading_level": heading_level or None,
            },
        )

    # ------------------------------------------------------------------
    # Splitting strategies
    # ------------------------------------------------------------------

    def _split_section(self, text: str, section: SectionBlock) -> list[str]:
        """Split *text* into chunks respecting the size target and overlap.

        If the section contains structural elements, the text is first
        decomposed into atomic units (whole code blocks, whole tables,
        prose runs) and units are packed into target-sized chunks.  Any
        single oversize **non-atomic** unit (a long prose run) is further
        split into target-sized windows via :meth:`_window_split`.  Atomic
        units (fenced code blocks and Markdown tables) are never split —
        an atomic unit larger than the target stays whole in its own chunk.
        """
        if section.has_code or section.has_table:
            units = self._extract_atomic_elements(text)
        else:
            units = [
                re.sub(r"\s+", " ", p).strip()
                for p in re.split(r"\n\s*\n", text)
                if p.strip()
            ] or [text]

        chunks: list[str] = []
        current: list[str] = []
        current_wc = 0
        pending_tail = ""  # word tail of last emitted chunk (overlap seed)

        def flush_current() -> None:
            """Emit the current chunk and remember its word tail for overlap."""
            nonlocal current, current_wc, pending_tail
            if not current:
                return
            content = "\n\n".join(current)
            chunks.append(content)
            words = content.split()
            pending_tail = (
                " ".join(words[-self.overlap:])
                if len(words) > self.overlap
                else content
            )
            current = []
            current_wc = 0

        def add(piece: str) -> None:
            """Add a piece to the current chunk; flush/seed with overlap as needed."""
            nonlocal current, current_wc, pending_tail
            piece_wc = len(piece.split())
            limit = self.target + self.overlap  # allow seed + content in one chunk

            # Start a fresh chunk with the overlap tail from the previous chunk.
            if not current and pending_tail:
                current.append(pending_tail)
                current_wc += len(pending_tail.split())
                pending_tail = ""

            if current_wc + piece_wc <= limit:
                current.append(piece)
                current_wc += piece_wc
                return

            flush_current()
            if pending_tail:
                current.append(pending_tail)
                current_wc += len(pending_tail.split())
                pending_tail = ""
            current.append(piece)
            current_wc += piece_wc

        for unit in units:
            unit_wc = len(unit.split())
            if unit_wc > self.target:
                if self._is_atomic_unit(unit):
                    # Code integrity > size: emit any carried chunk, then the
                    # whole atomic unit in its own chunk.
                    flush_current()
                    chunks.append(unit)
                    words = unit.split()
                    pending_tail = (
                        " ".join(words[-self.overlap:])
                        if len(words) > self.overlap
                        else unit
                    )
                    continue
                for window in self._window_split(unit):
                    add(window)
                continue
            add(unit)

        flush_current()
        return chunks or [text]

    def _window_split(self, text: str) -> list[str]:
        """Split an oversize prose run into clean target-sized windows.

        Windows are returned *without* overlap — the caller's ``add`` logic
        introduces the overlap between consecutive chunks.
        """
        words = text.split()
        if len(words) <= self.target:
            return [text]
        return [
            " ".join(words[i:i + self.target])
            for i in range(0, len(words), self.target)
        ]

    @staticmethod
    def _is_atomic_unit(text: str) -> bool:
        """Return *True* if *text* is a whole fenced code block or table.

        Used to decide whether an oversize unit must stay intact (atomic)
        or may be split into windows (prose).
        """
        stripped = text.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            return True
        for line in stripped.split("\n"):
            if not line.strip():
                continue
            return line.strip().startswith("|")
        return False

    def _extract_atomic_elements(self, text: str) -> list[str]:
        """Return *text* split into atomic pieces (code blocks, tables, prose).

        Each returned string is either a whole fenced code block, a whole
        Markdown table (consecutive ``|``-prefixed lines), or a prose run.
        Structural units are separated from neighbouring prose so they can be
        preserved intact during chunking.
        """
        lines = text.split("\n")
        pieces: list[str] = []
        current: list[str] = []
        in_fence = False
        fence_char = "`"

        def flush_current() -> None:
            nonlocal current
            if current:
                piece = "\n".join(current)
                if piece.strip():
                    pieces.append(piece)
                current = []

        for line in lines:
            stripped = line.strip()

            if in_fence:
                current.append(line)
                closer = "```" if fence_char == "`" else "~~~"
                if stripped.startswith(closer) and stripped[3:].strip() == "":
                    flush_current()
                    in_fence = False
                continue

            if stripped.startswith("```") or stripped.startswith("~~~"):
                flush_current()
                fence_char = "`" if stripped[0] == "`" else "~"
                in_fence = True
                current.append(line)
                continue

            if stripped.startswith("|"):
                # Table row: flush preceding prose so the table stays whole.
                if current and not current[-1].strip().startswith("|"):
                    flush_current()
                current.append(line)
                continue

            # Prose line: flush the table block once prose resumes.
            if current and current[-1].strip().startswith("|"):
                flush_current()
            current.append(line)

        flush_current()
        return pieces

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_piece_meta(text: str) -> dict[str, bool]:
        """Return ``{"code_block": bool, "table": bool}`` for a split piece."""
        has_code = False
        has_table = False
        fence_open = False
        fence_char = "`"
        for line in text.split("\n"):
            s = line.strip()
            if fence_open:
                closer = "```" if fence_char == "`" else "~~~"
                if s.startswith(closer) and s[3:].strip() == "":
                    fence_open = False
                continue
            if s.startswith("```") or s.startswith("~~~"):
                fence_char = "`" if s[0] == "`" else "~"
                fence_open = True
                has_code = True
                continue
            if s.startswith("|"):
                has_table = True
        return {"code_block": has_code, "table": has_table}
