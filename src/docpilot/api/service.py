"""Event orchestration — the streaming ask path (Phase 5, SPEC §7).

:func:`ask_events` runs one question end-to-end and reports progress through
the ``emit`` callback as type-tagged event dicts (protocol documented in
:mod:`docpilot.api.sse`).  It is **synchronous and duck-typed**, so:

    * the FastAPI app runs it in a worker thread and bridges events to an
      ``asyncio.Queue`` for the SSE generator (:mod:`docpilot.api.app`);
    * hermetic tests call it directly with injected fakes and a list
      collector for ``emit``;
    * the Chainlit UI calls it in-process with a callback that pumps Chainlit
      primitives (no HTTP hop, secrets stay server-side).

Routing mirrors :func:`docpilot.agent.pipeline_agentic.agentic_ask` exactly:
``auto`` lets the heuristic gate decide, ``direct`` / ``agentic`` force a
path.  On the agentic path the gate (and every node) step event is emitted by
``agentic_ask`` itself; on the direct path this module emits the gate step,
the retrieval debug payload, streamed answer tokens, and the final answer —
mirroring ``pipeline_ask.ask``'s SPEC §3.9 construction so output is
byte-identical to the CLI's direct path.

Every run ends with a terminal event (``answer`` then ``done``, or ``error``);
the returned dict is the same summary the ``done`` event carries, which the
app layer persists into the session store.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from docpilot import config
from docpilot.agent.gate import HeuristicQueryClassifier
from docpilot.agent.graph import trace_step_to_dict
from docpilot.agent.pipeline_agentic import agentic_ask
from docpilot.agent.types import LoopTraceStep
from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.models import SourceRef, derive_source_kind
from docpilot.generation.prompts import format_sources

logger = logging.getLogger(__name__)

# Excerpt length for the debug-panel "search" payload (identical constant in
# docpilot.agent.graph — the two event emitters stay shape-compatible).
_EXCERPT_CHARS = 220

_VALID_STRATEGIES = frozenset({"auto", "direct", "agentic"})


def _noop(_event: dict) -> None:
    """Default emit: drop events (service still returns the summary)."""


def _search_payload(turn: int, query: str, results, started: float) -> dict:
    """Debug-panel "search" event — same shape as the agentic graph's."""
    return {
        "type": "search",
        "turn": turn,
        "query": query,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "results": [
            {
                "file": r.chunk.source_file,
                "heading": r.chunk.heading_path,
                "kind": derive_source_kind(r.chunk.source_file),
                "score": round(r.score, 4),
                "excerpt": r.chunk.content[:_EXCERPT_CHARS],
            }
            for r in results
        ],
    }


def _sources_out(sources: list[SourceRef]) -> list[dict]:
    """Serialise SourceRefs for the answer/done events (incl. kind)."""
    return [
        {"ref": s.ref, "file": s.file, "heading": s.heading, "kind": s.kind}
        for s in sources
    ]


def ask_events(
    question: str,
    *,
    strategy: str | None = None,
    top_k: int | None = None,
    language: str | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    retriever=None,
    generator=None,
    citation_engine=None,
    judge=None,
    tool=None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    """Run *question* and return the run summary, streaming events via *emit*.

    Args:
        question: The user's question.
        strategy: ``"auto"`` (default; gate decides), ``"direct"`` (force the
            fast path) or ``"agentic"`` (force the loop).
        top_k: Retrieval count; defaults to ``config.RETRIEVAL_TOP_K`` on the
            direct path and ``config.AGENT_LOOP_TOP_K`` on the agentic path.
        language: Retrieval language filter; defaults to
            ``config.RETRIEVAL_LANGUAGE``; ``"any"`` disables filtering.
        model: Operator-side model knob (SPEC §7 hardening): overrides
            ``GROQ_MODEL`` for the answer generator when given.
        max_retries: Hard judge budget (agentic path only).
        retriever / generator / citation_engine / judge / tool: Injectable
            components (mirroring ``agentic_ask``); defaults are built lazily
            when ``None`` (production PG/Groq stack).
        emit: Live event callback (see module docstring); default drops events.

    Returns:
        A summary dict — ``{question, answer, sources, refused, direct,
        trace, latency_ms, usage}`` — identical to the ``done`` event payload
        minus the event type.

    Raises:
        ValueError: If *strategy* is not ``auto``/``direct``/``agentic``.
    """
    strategy = strategy or config.AGENT_DEFAULT_STRATEGY
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of 'auto', 'direct', 'agentic'; got {strategy!r}"
        )
    if emit is None:
        emit = _noop

    if strategy == "agentic":
        return _run_agentic(
            question,
            ask_strategy="agentic",
            top_k=top_k,
            language=language,
            max_retries=max_retries,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            judge=judge,
            tool=tool,
            model=model,
            emit=emit,
        )
    if strategy == "direct":
        return _run_direct(
            question,
            gate_decision="forced-direct",
            top_k=top_k,
            language=language,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            model=model,
            emit=emit,
        )

    # strategy == "auto": the heuristic gate decides (zero LLM calls).
    gate = HeuristicQueryClassifier()
    decision = gate.classify(question)
    if decision.agentic:
        # The agentic path emits its own gate step (decision "agentic") as
        # the first event — same shape as the direct path's.
        return _run_agentic(
            question,
            ask_strategy="auto",
            top_k=top_k,
            language=language,
            max_retries=max_retries,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            judge=judge,
            tool=tool,
            model=model,
            emit=emit,
        )
    gate_detail = decision.reason
    if decision.signals:
        gate_detail = f"signals={decision.signals}; {decision.reason}"
    return _run_direct(
        question,
        gate_decision="direct",
        gate_detail=gate_detail,
        top_k=top_k,
        language=language,
        retriever=retriever,
        generator=generator,
        citation_engine=citation_engine,
        model=model,
        emit=emit,
    )


# ---------------------------------------------------------------------------
# Direct (fast) path — service-owned, streams tokens (SPEC §7).
# ---------------------------------------------------------------------------


def _run_direct(
    question: str,
    *,
    gate_decision: str,
    gate_detail: str | None = None,
    top_k: int | None = None,
    language: str | None = None,
    retriever=None,
    generator=None,
    citation_engine=None,
    model: str | None = None,
    emit: Callable[[dict], None],
) -> dict:
    """Fast path with live streaming — mirrors ``pipeline_ask.ask`` (SPEC §3.9)
    but yields answer tokens to ``emit`` as they arrive, then emits the
    formatted answer, the trace and the done summary."""
    started = time.perf_counter()

    gate_started = time.perf_counter()
    gate_step = LoopTraceStep.new(
        "gate", question, gate_decision, detail=gate_detail, started_at=gate_started
    )
    emit({"type": "step", "step": trace_step_to_dict(gate_step)})

    conn = None
    if retriever is None:
        from docpilot.pipeline_ask import _build_default_retriever

        retriever, conn = _build_default_retriever()
    if generator is None:
        from docpilot.generation.generator import GroqGenerator

        generator = GroqGenerator(model=model)
    if citation_engine is None:
        citation_engine = StandardCitationEngine()
    top_k = top_k if top_k is not None else config.RETRIEVAL_TOP_K

    # Language policy mirrors pipeline_ask: arg wins, else config default;
    # the literal "any" means no filter.
    resolved_language = language if language is not None else config.RETRIEVAL_LANGUAGE
    filter_language = None if resolved_language == "any" else resolved_language

    try:
        # ── retrieve ──────────────────────────────────────────────────────
        search_started = time.perf_counter()
        results = retriever.retrieve(question, top_k=top_k, language=filter_language)
        logger.info("Direct path retrieved %d result(s)", len(results))
        emit(_search_payload(1, question, results, search_started))
        top_scores = [round(r.score, 4) for r in results[:3]]
        search_step = LoopTraceStep.new(
            "search",
            question,
            "retrieved",
            detail=(
                f"turn=1: retrieved {len(results)} chunks; top scores: {top_scores}"
            ),
            started_at=search_started,
        )
        emit({"type": "step", "step": trace_step_to_dict(search_step)})

        # ── context + sources (SPEC §3.9, mirrors pipeline_ask.ask) ───────
        if results:
            context_text = "\n\n".join(
                f"[{i + 1}] {r.chunk.content}" for i, r in enumerate(results)
            )
        else:
            from docpilot.pipeline_ask import _NO_CONTEXT_NOTE

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

        # ── generate (streamed when supported) ────────────────────────────
        answer_started = time.perf_counter()
        streamed: list[str] = []
        stream_ok = False
        try:
            for delta in generator.generate_answer_stream(
                context_text, sources_text, question
            ):
                streamed.append(delta)
                emit({"type": "token", "delta": delta})
            stream_ok = True
        except NotImplementedError:
            stream_ok = False

        if stream_ok and streamed and "".join(streamed).strip():
            raw_response = "".join(streamed)
        else:
            # Streaming unsupported or produced nothing usable — single call.
            raw_response = generator.generate_answer(context_text, sources_text, question)
            emit({"type": "token", "delta": raw_response})

        # ── cite ──────────────────────────────────────────────────────────
        answer, footer = citation_engine.format_answer(raw_response, sources)
        display = f"{answer}\n\n{footer}" if footer else answer
        answer_step = LoopTraceStep.new(
            "answer", question, "answer", started_at=answer_started
        )
        emit({"type": "step", "step": trace_step_to_dict(answer_step)})
    finally:
        if conn is not None:
            conn.close()

    latency_ms = (time.perf_counter() - started) * 1000.0
    sources_out = _sources_out(sources)
    trace = [
        trace_step_to_dict(gate_step),
        trace_step_to_dict(search_step),
        trace_step_to_dict(answer_step),
    ]
    emit(
        {
            "type": "answer",
            "text": display,
            "sources": sources_out,
            "refused": False,
            "direct": True,
        }
    )
    usage = getattr(generator, "last_usage", None)
    emit({"type": "done", "trace": trace, "latency_ms": round(latency_ms, 1), "usage": usage})
    return {
        "question": question,
        "answer": display,
        "sources": sources_out,
        "refused": False,
        "direct": True,
        "trace": trace,
        "latency_ms": latency_ms,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Agentic path — delegates to agentic_ask with the emit hook.
# ---------------------------------------------------------------------------


def _run_agentic(
    question: str,
    *,
    ask_strategy: str,
    top_k: int | None = None,
    language: str | None = None,
    max_retries: int | None = None,
    retriever=None,
    generator=None,
    citation_engine=None,
    judge=None,
    tool=None,
    model: str | None = None,
    emit: Callable[[dict], None],
) -> dict:
    """Agentic loop — ``agentic_ask`` emits the gate step, each node's step
    and streamed answer tokens through the same ``emit`` channel.  This module
    only adds the final ``answer`` and ``done`` events."""
    started = time.perf_counter()

    # Build the answer generator here (not inside agentic_ask) so the per-call
    # token usage is observable via generator.last_usage afterwards; the model
    # knob (SPEC §7) applies to the answer generator on this path too.
    if generator is None:
        from docpilot.generation.generator import GroqGenerator

        generator = GroqGenerator(model=model)

    result = agentic_ask(
        question,
        retriever=retriever,
        generator=generator,
        judge=judge,
        citation_engine=citation_engine,
        strategy=ask_strategy,
        top_k=top_k,
        language=language,
        max_retries=max_retries,
        tool=tool,
        emit=emit,
    )

    latency_ms = (time.perf_counter() - started) * 1000.0
    sources_out = _sources_out(result.sources)
    emit(
        {
            "type": "answer",
            "text": result.answer,
            "sources": sources_out,
            "refused": result.refused,
            "direct": result.direct,
        }
    )
    trace = [trace_step_to_dict(t) for t in result.trace]
    usage = getattr(generator, "last_usage", None)
    emit({"type": "done", "trace": trace, "latency_ms": round(latency_ms, 1), "usage": usage})
    return {
        "question": question,
        "answer": result.answer,
        "sources": sources_out,
        "refused": result.refused,
        "direct": result.direct,
        "trace": trace,
        "latency_ms": latency_ms,
        "usage": usage,
    }