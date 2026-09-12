"""Shared direct/fast ask core: retrieve → context+sources → generate → cite.

The Phase-1 ask path exists in two places today —
:func:`docpilot.pipeline_ask.ask` (the CLI pipeline, SPEC §3.7–§3.11) and
:func:`docpilot.api.service._run_direct` (the streamed SSE fast path,
SPEC §7).  Both perform the same retrieval / context construction /
generation / citation work; they differ only in observability: ``ask`` logs
retrieval detail and the reconstructed prompt, while the service emits
step/token SSE events.  This module is that shared core, so the two callers
stay byte-identical without duplicating the logic.

Design notes:

    * The core is **logging-free** — every log line keeps the provenance of
      its caller (the CLI's ``--debug`` stderr stays byte-identical); the
      callers pass hooks for anything they want to observe.
    * Optional ``emit`` hooks let the service drive its event sequence from
      the same code path: ``on_search`` fires immediately after retrieval
      (so the service can emit the debug-panel payload + ``search`` trace
      step before any token arrives) and ``on_generate_delta`` receives each
      streamed answer token.  With no hooks the core performs pure Phase-1
      behaviour — a single ``generate_answer`` call, never streaming —
      exactly what ``pipeline_ask.ask`` needs.
    * Latency is owned by the callers (``ask`` measures from its own start;
      the service measures from before its gate step).  The core only exposes
      the ``time.perf_counter()`` timestamps captured before retrieval and
      before generation so the service can build trace steps whose ``started_at``
      values land in the same places as before.

Import graph: this module depends only on :mod:`docpilot.core.models` and
:mod:`docpilot.generation.prompts`, so both ``pipeline_ask`` and
``api.service`` can import it without cycles.  The placeholder context note
(:data:`_NO_CONTEXT_NOTE`) is defined here but re-exported from
``docpilot.pipeline_ask``, which is where :mod:`docpilot.agent.graph` (and
the tests) import it from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from docpilot.core.models import (
    RetrieverResult,
    SourceRef,
    derive_source_kind,
)
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources

# Placeholder context shown to the LLM when retrieval returned nothing.
# Canonical home — re-exported from docpilot.pipeline_ask so the agent graph's
# import location stays unchanged.
_NO_CONTEXT_NOTE = "[no context retrieved — the documentation may not cover this question.]"


@dataclass
class DirectResult:
    """Everything the shared direct path computed, for either caller.

    Attributes:
        results: The raw :class:`RetrieverResult` list from the store.
        context_text: The numbered context block offered to the LLM (or the
            :data:`_NO_CONTEXT_NOTE` placeholder when retrieval was empty).
        sources: The numbered :class:`SourceRef` list derived from *results*.
        sources_text: The ``format_sources`` rendering of *sources*.
        raw_prompt: The full prompt built from ``SYSTEM_PROMPT`` with the
            context/sources/question substituted (for debugging).
        raw_response: The generator's raw output before citation formatting.
        answer: The citation-formatted answer string.
        footer: The source footer (``Sources:\\n...``) or ``""`` when the
            answer contains no valid citation markers.
        display: ``answer + "\\n\\n" + footer``, or just the bare answer when
            there is no footer (SPEC.md §3.11).
        search_started: ``time.perf_counter()`` captured just before
            retrieval (so the service can timestamp its ``search`` trace step
            identically to before).
        answer_started: ``time.perf_counter()`` captured just before
            generation (for the service's ``answer`` trace step).
    """

    results: list[RetrieverResult] = field(default_factory=list)
    context_text: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    sources_text: str = ""
    raw_prompt: str = ""
    raw_response: str = ""
    answer: str = ""
    footer: str = ""
    display: str = ""
    search_started: float = 0.0
    answer_started: float = 0.0


def _run_direct_core(
    question: str,
    *,
    retriever,
    generator,
    citation_engine,
    top_k: int,
    filter_language: str | None,
    on_search: Callable[[list[RetrieverResult], float], None] | None = None,
    on_generate_delta: Callable[[str], None] | None = None,
) -> DirectResult:
    """Run the shared retrieve → build context → generate → cite core.

    Args:
        question: The user's question.
        retriever: A ``Retriever`` (never ``None`` — callers resolve defaults
            before delegating).
        generator: A ``Generator`` (never ``None`` — callers resolve defaults
            before delegating).
        citation_engine: A ``CitationEngine`` (never ``None``).
        top_k: Number of chunks to retrieve (already resolved by the caller).
        filter_language: Language filter for the retrieval call; ``None``
            means no filtering (already resolved by the caller).
        on_search: Optional hook invoked with ``(results, search_started)``
            immediately after retrieval — before context is built, sources
            derived, or any token is generated.  The service uses it to emit
            the debug-panel payload and ``search`` trace step; the CLI wrapper
            uses it for its retrieval log lines.  ``None`` → no hook.
        on_generate_delta: Optional hook receiving each answer token as it is
            produced.  When given, the core tries ``generate_answer_stream``
            first and falls back to a single ``generate_answer`` when
            streaming is unsupported (``NotImplementedError``) or produced
            nothing usable — the fallback emits the whole response as one
            delta.  When ``None``, the core always uses the single
            ``generate_answer`` call (pure Phase-1 behaviour, never streams).

    Returns:
        A :class:`DirectResult` with everything both callers need, including
        the retrieval/generation start timestamps for trace-step latency.
    """
    # ── retrieve ───────────────────────────────────────────────────────────
    search_started = time.perf_counter()
    results = retriever.retrieve(question, top_k=top_k, language=filter_language)
    if on_search is not None:
        on_search(results, search_started)

    # ── build context + sources (SPEC.md §3.9) ─────────────────────────────
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

    # ── generate ───────────────────────────────────────────────────────────
    # The exact prompt is reconstructed here (identical to what the real
    # GroqGenerator builds via generate_answer) so callers can log it / keep
    # it (AskResult.raw_prompt) without leaking secrets — the prompt contains
    # only retrieved documentation text.
    full_prompt = SYSTEM_PROMPT.format(
        context=context_text,
        sources=sources_text,
        question=question,
    )
    answer_started = time.perf_counter()
    if on_generate_delta is not None:
        # Streaming path (service/UI): try the stream first, then fall back
        # to a single call when unsupported or the stream produced nothing.
        streamed: list[str] = []
        stream_ok = False
        try:
            for delta in generator.generate_answer_stream(
                context_text, sources_text, question
            ):
                streamed.append(delta)
                on_generate_delta(delta)
            stream_ok = True
        except NotImplementedError:
            stream_ok = False

        if stream_ok and streamed and "".join(streamed).strip():
            raw_response = "".join(streamed)
        else:
            # Streaming unsupported or produced nothing usable — single call.
            raw_response = generator.generate_answer(context_text, sources_text, question)
            on_generate_delta(raw_response)
    else:
        # Non-streaming path (Phase 1 / CLI) — always a single call.
        raw_response = generator.generate_answer(context_text, sources_text, question)

    # ── cite ───────────────────────────────────────────────────────────────
    answer, footer = citation_engine.format_answer(raw_response, sources)
    display = f"{answer}\n\n{footer}" if footer else answer

    return DirectResult(
        results=results,
        context_text=context_text,
        sources=sources,
        sources_text=sources_text,
        raw_prompt=full_prompt,
        raw_response=raw_response,
        answer=answer,
        footer=footer,
        display=display,
        search_started=search_started,
        answer_started=answer_started,
    )