"""Code-generation pipeline: retrieve → prompt → generate → cite (PLAN §7.2 T2).

T2 is the "CodeGenerator" capability: it retrieves documentation with the
existing ``Retriever``, builds a code-request prompt with the retrieved
API/schema/examples inline (:data:`CODE_PROMPT`), calls the existing
``Generator`` interface, and attaches ``CitationEngine`` sources to the
output.  It deliberately does **not** validate yet — T3 wires the T1
``CodeValidator`` in, T4 adds the reformulation loop; ``CodeRequest.verdict``
is the hook T4 fills.

No LLM and no database are hard-coded into the flow — every component is
injectable, so the hermetic tests drive fakes end-to-end (same pattern as
:func:`docpilot.pipeline_ask.ask`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from docpilot import config
from docpilot.citations.engine import StandardCitationEngine
from docpilot.codegen.prompts import CODE_PROMPT
from docpilot.core.direct import _NO_CONTEXT_NOTE
from docpilot.core.models import (
    RetrieverResult,
    SourceRef,
    derive_source_kind,
)
from docpilot.generation.prompts import format_sources
from docpilot.validation.validator import extract_code_blocks
from docpilot.validation.verdict import ValidationVerdict

logger = logging.getLogger(__name__)

# SPEC §3.9 mandatory refusal wording (ACTIVE.md §2 contract constant) — the
# code path refuses with the exact same sentence when retrieval cannot cover
# the request, and emits no code.
_REFUSAL_SENTENCE = (
    "I don't know — the available documentation does not cover this question."
)


@dataclass
class CodeRequest:
    """Everything the code pipeline produced, including observability fields.

    Attributes:
        question: The user's code request.
        raw_response: The generator's raw output before citation formatting
            (prose + fenced code blocks + [N] markers).
        code_blocks: Every fenced code block extracted from ``raw_response``
            (the candidate code T4 validates). Empty ⇒ the model emitted no
            code (refusal or a prose-only answer).
        answer: The citation-formatted answer string.
        footer: The source footer (``Sources:\\n...``) or ``""`` when the
            answer contains no valid citation markers.
        sources: The numbered :class:`SourceRef` list offered to the LLM
            (empty when retrieval returned nothing).
        results: The raw :class:`RetrieverResult` list from the store.
        raw_prompt: The full code prompt built from ``CODE_PROMPT`` with the
            context/sources/question substituted (for debugging).
        latency_ms: Total wall time of the code pipeline, in milliseconds.
        refused: ``True`` when the raw response contains the SPEC §3.9
            refusal sentence (uncovered request ⇒ no code).
        verdict: Validation verdict, filled by the T3/T4 wiring; ``None``
            while the T2-only pipeline is used.
    """

    question: str
    raw_response: str = ""
    code_blocks: list[str] = field(default_factory=list)
    answer: str = ""
    footer: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    results: list[RetrieverResult] = field(default_factory=list)
    raw_prompt: str = ""
    latency_ms: float = 0.0
    refused: bool = False
    verdict: ValidationVerdict | None = None

    @property
    def display(self) -> str:
        """The final user-facing string (answer + footer, SPEC.md §3.11)."""
        if self.footer:
            return f"{self.answer}\n\n{self.footer}"
        return self.answer

    @property
    def has_code(self) -> bool:
        """True when the pipeline produced at least one fenced code block."""
        return bool(self.code_blocks)


def ask_code(
    question: str,
    *,
    retriever=None,
    generator=None,
    citation_engine=None,
    top_k: int | None = None,
    language: str | None = None,
    prompt_template: str = CODE_PROMPT,
) -> CodeRequest:
    """Retrieve docs for *question*, generate code grounded in them, cite it.

    Args:
        question: The user's code request.
        retriever: An optional ``Retriever`` (default: ``SimpleRetriever`` +
            ``PgVectorStore``, same builder as the answer path).
        generator: An optional ``Generator`` (default: ``GroqGenerator``).
        citation_engine: An optional ``CitationEngine`` (default:
            ``StandardCitationEngine``).
        top_k: Number of chunks to retrieve; defaults to
            ``config.RETRIEVAL_TOP_K``.
        language: Retrieval language filter; defaults to
            ``config.RETRIEVAL_LANGUAGE``. The literal ``"any"`` disables
            filtering.
        prompt_template: Prompt template with ``{context}``, ``{sources}``
            and ``{question}`` placeholders (default ``CODE_PROMPT``).
            Injectable so tests and future prompts are trivial to swap.

    Returns:
        A :class:`CodeRequest` carrying the raw/cited output, extracted code
        blocks, sources and observability fields.
    """
    from docpilot.generation.generator import GroqGenerator
    from docpilot.pipeline_ask import _build_default_retriever

    conn = None
    if retriever is None:
        retriever, conn = _build_default_retriever()
    if generator is None:
        generator = GroqGenerator()
    if citation_engine is None:
        citation_engine = StandardCitationEngine()
    top_k = top_k if top_k is not None else config.RETRIEVAL_TOP_K

    resolved_language = language if language is not None else config.RETRIEVAL_LANGUAGE
    filter_language = None if resolved_language == "any" else resolved_language

    started = time.perf_counter()
    try:
        results = retriever.retrieve(question, top_k=top_k, language=filter_language)
        for i, r in enumerate(results):
            logger.debug(
                "Retrieved chunk %d: id=%s score=%.4f file=%s heading=%s",
                i + 1,
                r.chunk.id,
                r.score,
                r.chunk.source_file,
                r.chunk.heading_path or "None",
            )
        logger.info("Retrieved %d result(s) for code request", len(results))
        if not results:
            logger.warning(
                "No context retrieved for code request — the documentation "
                "may not cover this question."
            )

        # Context + sources — identical construction to the answer path
        # (core.direct), so citations resolve to the same real doc pages.
        if results:
            context_text = "\n\n".join(
                f"[{i + 1}] {r.chunk.content}" for i, r in enumerate(results)
            )
        else:
            context_text = _NO_CONTEXT_NOTE

        sources = [
            SourceRef(
                ref=i + 1,
                file=r.chunk.source_file,
                heading=r.chunk.heading_path or None,
                kind=derive_source_kind(r.chunk.source_file),
            )
            for i, r in enumerate(results)
        ]
        sources_text = format_sources(sources)

        # Generate code via the shared Generator interface (single system
        # prompt, same as generate_answer — no new call machinery).
        full_prompt = prompt_template.format(
            context=context_text,
            sources=sources_text,
            question=question,
        )
        raw_response = generator.generate(full_prompt)

        # Cite — existing CitationEngine output, no new marker syntax (§7.5).
        answer, footer = citation_engine.format_answer(raw_response, sources)

        code_blocks = extract_code_blocks(raw_response)
        refused = _REFUSAL_SENTENCE in raw_response
    finally:
        if conn is not None:
            conn.close()

    logger.debug("Full code prompt sent to LLM:\n%s", full_prompt)
    logger.debug("Raw code LLM response:\n%s", raw_response)
    logger.info(
        "Code request answered in %.1f ms total (%d code block(s), refused=%s).",
        (time.perf_counter() - started) * 1000.0,
        len(code_blocks),
        refused,
    )

    return CodeRequest(
        question=question,
        raw_response=raw_response,
        code_blocks=code_blocks,
        answer=answer,
        footer=footer,
        sources=sources,
        results=results,
        raw_prompt=full_prompt,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        refused=refused,
    )