"""End-to-end ask pipeline: retrieve → generate → cite.

Wires the retriever, the LLM-backed generator and the citation engine into a
single :func:`ask` callable (SPEC.md §3.7–§3.10). Components are injectable so
tests run hermetically without a live database or Groq API.

The final display string (SPEC.md §3.11) is ``answer + "\\n\\n" + footer``, or
just the bare answer when there is no footer. On empty retrieval the pipeline
still calls the generator with a placeholder context indicating that nothing
was found; the LLM is expected to refuse (the "I don't know" path) and the
pipeline must never crash.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from docpilot import config
from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.models import (
    RetrieverResult,
    SourceRef,
    derive_source_kind,
)
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources

logger = logging.getLogger(__name__)

# Placeholder context shown to the LLM when retrieval returned nothing.
_NO_CONTEXT_NOTE = "[no context retrieved — the documentation may not cover this question.]"


@dataclass
class AskResult:
    """Everything the ask pipeline produced, including observability fields.

    Attributes:
        question: The user's question.
        answer: The citation-formatted answer string.
        footer: The source footer (``Sources:\\n...``) or ``""`` when the
            answer contains no valid citation markers.
        sources: The numbered :class:`SourceRef` list offered to the LLM
            (empty when retrieval returned nothing).
        results: The raw :class:`RetrieverResult` list from the store.
        raw_prompt: The full prompt built from ``SYSTEM_PROMPT`` with the
            context/sources/question substituted (for debugging).
        raw_response: The generator's raw output before citation formatting.
        latency_ms: Total wall time of the ask pipeline, in milliseconds.
    """

    question: str
    answer: str
    footer: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    results: list[RetrieverResult] = field(default_factory=list)
    raw_prompt: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0

    @property
    def display(self) -> str:
        """The final user-facing string (answer + footer, SPEC.md §3.11)."""
        if self.footer:
            return f"{self.answer}\n\n{self.footer}"
        return self.answer


def _embedding_dimension(retriever) -> int | None:
    """Best-effort query-embedding dimension, or ``None`` when unknowable.

    The retriever interface does not expose its embedding provider, so this
    reads it defensively (``SimpleRetriever`` stores ``_embedding_provider``).
    Debug logging only — never raises.
    """
    provider = getattr(retriever, "embedding_provider", None) or getattr(
        retriever, "_embedding_provider", None
    )
    if provider is None:
        return None
    try:
        return provider.dimension
    except Exception:
        return None


def _build_default_retriever():
    """Build ``SimpleRetriever(BGEEmbeddingProvider(), PgVectorStore(conn))``.

    Returns:
        A ``(retriever, conn)`` tuple. The caller owns *conn* and must close it.

    Raises:
        RuntimeError: If the database is unreachable or the schema cannot be
            applied (psycopg/native error chained as the cause).
    """
    from docpilot.db.connection import ensure_schema, get_connection
    from docpilot.embeddings.provider import BGEEmbeddingProvider
    from docpilot.retrieval.retriever import SimpleRetriever
    from docpilot.retrieval.vector_store import PgVectorStore

    conn = None
    try:
        conn = get_connection()
        ensure_schema(conn)
        store = PgVectorStore(conn)
    except Exception as exc:
        if conn is not None:
            conn.close()
        raise RuntimeError(
            "Could not connect to the DocPilot PostgreSQL database / apply the "
            "pgvector schema — is the server running and is the 'vector' "
            f"extension available? ({exc})"
        ) from exc
    embedding_provider = BGEEmbeddingProvider()
    retriever = SimpleRetriever(embedding_provider, store)
    if config.RERANK_ENABLED:
        from docpilot.reranking.reranker import BCEReranker

        retriever = SimpleRetriever(
            embedding_provider, store, reranker=BCEReranker()
        )
    if config.HYBRID_ENABLED:
        from docpilot.retrieval.hybrid import HybridRetriever
        from docpilot.retrieval.lexical import PostgresFTSSearcher

        # Vector half = the (possibly reranked) dense retriever above;
        # lexical half = Postgres full-text over the same chunks table.
        retriever = HybridRetriever(
            retriever,
            PostgresFTSSearcher(conn),
            top_k_each=config.HYBRID_TOP_K_EACH,
            rrf_k=config.HYBRID_RRF_K,
            weight_vector=config.HYBRID_WEIGHT_VECTOR,
            weight_lexical=config.HYBRID_WEIGHT_LEXICAL,
        )
    return retriever, conn


def ask(
    question: str,
    *,
    retriever=None,
    generator=None,
    citation_engine=None,
    top_k: int | None = None,
    language: str | None = None,
) -> AskResult:
    """Retrieve context for *question*, generate a cited answer and return it.

    Args:
        question: The user's question.
        retriever: An optional ``Retriever``
            (default: ``SimpleRetriever`` + ``PgVectorStore``).
        generator: An optional ``Generator`` (default: ``GroqGenerator``).
        citation_engine: An optional ``CitationEngine``
            (default: ``StandardCitationEngine``).
        top_k: Number of chunks to retrieve; defaults to
            ``config.RETRIEVAL_TOP_K``.
        language: Retrieval language filter; defaults to
            ``config.RETRIEVAL_LANGUAGE``. The literal ``"any"`` disables
            filtering (retrieval across all languages).

    Returns:
        An :class:`AskResult` carrying the answer, footer, sources and raw
        observability fields (prompt, response, latency).
    """
    from docpilot.generation.generator import GroqGenerator

    conn = None
    if retriever is None:
        retriever, conn = _build_default_retriever()
    if generator is None:
        generator = GroqGenerator()
    if citation_engine is None:
        citation_engine = StandardCitationEngine()
    top_k = top_k if top_k is not None else config.RETRIEVAL_TOP_K

    # Language policy: explicit arg wins, else the config default; the
    # literal "any" means no filter at the retrieval layer.
    resolved_language = language if language is not None else config.RETRIEVAL_LANGUAGE
    filter_language = None if resolved_language == "any" else resolved_language
    logger.debug("Retrieval language filter: %s", filter_language)

    started = time.perf_counter()
    try:
        dim = _embedding_dimension(retriever)
        logger.debug(
            "Query embedding dimension: %s",
            dim if dim is not None else "unknown (retriever does not expose one)",
        )

        # ── retrieve ───────────────────────────────────────────────────────
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
        logger.info("Retrieved %d result(s)", len(results))

        # ── build context + sources (SPEC.md §3.9) ─────────────────────────
        if results:
            context_text = "\n\n".join(
                f"[{i + 1}] {r.chunk.content}" for i, r in enumerate(results)
            )
        else:
            context_text = _NO_CONTEXT_NOTE
            logger.warning(
                "No context retrieved — the documentation may not cover this question."
            )

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

        # ── generate ───────────────────────────────────────────────────────
        # The exact prompt is reconstructed here (identical to what the real
        # GroqGenerator builds via generate_answer) so it can be logged/kept
        # in AskResult without leaking secrets — the prompt contains only
        # retrieved documentation text.
        full_prompt = SYSTEM_PROMPT.format(
            context=context_text,
            sources=sources_text,
            question=question,
        )
        raw_response = generator.generate_answer(context_text, sources_text, question)
        logger.debug("Full prompt sent to LLM:\n%s", full_prompt)
        logger.debug("Raw LLM response:\n%s", raw_response)

        # ── cite ───────────────────────────────────────────────────────────
        answer, footer = citation_engine.format_answer(raw_response, sources)
    finally:
        if conn is not None:
            conn.close()

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Answer generated in %.1f ms total (retrieval + generation).", latency_ms)
    logger.debug("Total latency: %.1f ms", latency_ms)

    return AskResult(
        question=question,
        answer=answer,
        footer=footer,
        sources=sources,
        results=results,
        raw_prompt=full_prompt,
        raw_response=raw_response,
        latency_ms=latency_ms,
    )