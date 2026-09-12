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
the retrieval debug payload, streamed answer tokens, and the final answer.
The retrieve → context+sources → generate → cite work is delegated to the
shared direct core (:func:`docpilot.core.direct._run_direct_core`), the same
code the CLI pipeline (:func:`docpilot.pipeline_ask.ask`) runs, so output is
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
from docpilot.agent.code_route import run_code_route
from docpilot.agent.gate import HeuristicQueryClassifier
from docpilot.agent.graph import trace_step_to_dict
from docpilot.agent.pipeline_agentic import agentic_ask
from docpilot.agent.prompts import REFUSE_ANSWER
from docpilot.agent.types import LoopTraceStep
from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.direct import _run_direct_core
from docpilot.core.models import SourceRef, derive_source_kind
from docpilot.validation.verdict import CheckStatus, ValidationVerdict

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
    formatted answer, the trace and the done summary.  The retrieve →
    context+sources → generate → cite core is shared with the CLI pipeline
    (:func:`docpilot.core.direct._run_direct_core`); this wrapper keeps the
    connection lifecycle, the gate trace step, and the search / answer trace
    steps, driving the search and token events through the core's hooks."""
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

    # Pre-initialise so the post-finally trace construction never hits
    # UnboundLocalError if a mid-pipeline exception causes fall-through
    # (currently guarded by the bare `raise`, but future `except` clauses
    # could change this).
    search_step: LoopTraceStep | None = None
    answer_step: LoopTraceStep | None = None

    def _on_search(results, search_started) -> None:
        """Emit the debug-panel payload + ``search`` trace step right after
        retrieval, before any token is generated (SPEC §7 event order)."""
        nonlocal search_step
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
        if not results:
            logger.warning(
                "No context retrieved — the documentation may not cover this question."
            )

    try:
        # Shared core: retrieve → context+sources → generate → cite.  Its
        # streaming hook forwards answer tokens; its search hook carries the
        # debug payload and search trace step (see above).
        core = _run_direct_core(
            question,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            top_k=top_k,
            filter_language=filter_language,
            on_search=_on_search,
            on_generate_delta=lambda delta: emit({"type": "token", "delta": delta}),
        )
        answer_step = LoopTraceStep.new(
            "answer", question, "answer", started_at=core.answer_started
        )
        emit({"type": "step", "step": trace_step_to_dict(answer_step)})
    finally:
        if conn is not None:
            conn.close()

    latency_ms = (time.perf_counter() - started) * 1000.0
    sources_out = _sources_out(core.sources)
    trace = [
        trace_step_to_dict(s) for s in (gate_step, search_step, answer_step) if s is not None
    ]
    refused = REFUSE_ANSWER in core.raw_response  # mirrors eval benchmark §3.9
    emit(
        {
            "type": "answer",
            "text": core.display,
            "sources": sources_out,
            "refused": refused,
            "direct": True,
        }
    )
    usage = getattr(generator, "last_usage", None)
    emit({"type": "done", "trace": trace, "latency_ms": round(latency_ms, 1), "usage": usage})
    return {
        "question": question,
        "answer": core.display,
        "sources": sources_out,
        "refused": refused,
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


# ---------------------------------------------------------------------------
# Code route (Phase 6, PLAN §7.2 T6)
# ---------------------------------------------------------------------------


def _verdict_to_dict(verdict: ValidationVerdict) -> dict:
    """Serialize one validation verdict (checks keep name/status/detail)."""
    return {
        "passed": verdict.passed,
        "checks": [
            {"name": c.name, "status": c.status.value, "detail": c.detail}
            for c in verdict.checks
        ],
    }


def _code_payload(turn: int, request) -> dict:
    """The ``code`` SSE event — one attempt's validation verdict (§7.2 T6:
    "validation verdict is a new trace event").  Emitted per generation
    attempt; ``passed`` is ``None`` when nothing was validated (no code)."""
    return {
        "type": "code",
        "attempt": turn,
        "code_blocks": list(request.code_blocks),
        "verdicts": [_verdict_to_dict(v) for v in request.block_verdicts],
        "passed": None if request.verdict is None else request.verdict.passed,
        "validation_failed": request.validation_failed,
        "refused": request.refused,
        "reasons": list(request.validation_reasons),
        "attempt_latency_ms": round(request.latency_ms, 1),
    }


def _code_step_decision(request) -> str:
    """Trace-step decision for one attempt: validated / failed / no_code."""
    if not request.code_blocks:
        return "no_code"
    verdict = request.verdict
    if verdict is not None and verdict.passed and all(
        check.status is CheckStatus.PASS for check in verdict.checks
    ):
        return "validated"
    return "failed"


def code_events(
    question: str,
    *,
    top_k: int | None = None,
    language: str | None = None,
    model: str | None = None,
    max_validation_turns: int | None = None,
    retriever=None,
    generator=None,
    citation_engine=None,
    validator=None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    """Run *question* through the **explicit** code route (PLAN §7.2 T6) and
    stream the run as SSE-shaped events via *emit*.

    This endpoint is the opt-in code channel: ``explicit_code=True`` selects
    the code route regardless of the ``CODE_ROUTE_ENABLED`` lever (§7.5 — a
    caller asking on this endpoint is by definition asserting code intent).
    The T4 loop's lifecycle is streamed live through :func:`run_code_route`'s
    ``on_event`` hook:

        step(gate) → [search + step(search) + code + step(code_validation)]×N
        → answer → done

    * ``search`` — the debug-panel retrieval payload (true retrieval time);
    * ``code`` — one attempt's validation verdict (blocks + per-check status),
      the new §7.2 T6 trace event; one per generation attempt, including the
      final refusal-with-sources attempt;
    * ``answer`` — the citation-formatted display, plus ``code``,
      ``validation_failed``, ``validation_reasons``, ``validation_turns`` and
      the overall ``verdict`` so the UI can render refusal-vs-code honestly.

    Returns the run summary dict (identical to the ``done`` payload minus the
    event type).

    Args:
        question: The user's code request.
        top_k / language / model / max_validation_turns: Operable knobs —
            retrieval count, language filter, the operator-side model knob
            (SPEC §7, applied when the generator is built lazily), and the T4
            reformulation budget.
        retriever / generator / citation_engine / validator: Injectable
            components (defaults build the production PG/Groq/structural stack
            lazily).
        emit: Live event callback (default drops events).
    """
    emit = emit or _noop
    started = time.perf_counter()

    if generator is None and model is not None:
        from docpilot.generation.generator import GroqGenerator

        generator = GroqGenerator(model=model)

    trace_steps: list[LoopTraceStep] = []

    def on_event(kind: str, payload: dict) -> None:
        if kind == "gated":
            decision = payload["code"]
            gate_step = LoopTraceStep.new(
                "gate",
                question,
                "code" if decision else "ask",
                detail=payload["reason"],
                started_at=started,
            )
            trace_steps.append(gate_step)
            emit({"type": "step", "step": trace_step_to_dict(gate_step)})
            return
        if kind == "attempt":
            request = payload["request"]
            turn = payload["turn"]
            # True retrieval latency: request.retrieval_latency_ms was recorded
            # by ask_code; we emit right after the attempt, so now - latency ≈
            # the retrieval start (same shape as the direct path's payload).
            emit(
                _search_payload(
                    turn,
                    question,
                    request.results,
                    started=time.perf_counter() - request.retrieval_latency_ms / 1000.0,
                )
            )
            search_step = LoopTraceStep.new(
                "search",
                question,
                "retrieved",
                detail=(
                    f"turn={turn}: retrieved {len(request.results)} chunks; "
                    f"top scores: {[round(r.score, 4) for r in request.results[:3]]}"
                ),
                started_at=started,
            )
            trace_steps.append(search_step)
            emit({"type": "step", "step": trace_step_to_dict(search_step)})

            emit(_code_payload(turn, request))
            code_step = LoopTraceStep.new(
                "code_validation",
                question,
                _code_step_decision(request),
                detail=_code_step_detail(request),
                started_at=started,
            )
            trace_steps.append(code_step)
            emit({"type": "step", "step": trace_step_to_dict(code_step)})

    result = run_code_route(
        question,
        explicit_code=True,
        retriever=retriever,
        generator=generator,
        citation_engine=citation_engine,
        validator=validator,
        max_validation_turns=max_validation_turns,
        top_k=top_k,
        language=language,
        on_event=on_event,
    )

    request = result.code
    if request is None:
        # Unreachable on this endpoint (explicit opt-in always selects the
        # code route) — fail honestly, never fabricate an answer.
        emit({"type": "error", "message": "code route unavailable"})
        return {
            "question": question,
            "answer": "",
            "sources": [],
            "refused": True,
            "code": True,
            "trace": [trace_step_to_dict(t) for t in trace_steps],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "usage": getattr(generator, "last_usage", None),
        }

    latency_ms = (time.perf_counter() - started) * 1000.0
    sources_out = _sources_out(request.sources)
    answer = {
        "type": "answer",
        "text": request.display,
        "sources": sources_out,
        "refused": request.refused,
        "code": True,
        "validation_failed": request.validation_failed,
        "validation_reasons": list(request.validation_reasons),
        "validation_turns": request.generation_attempts,
        "verdict": None if request.verdict is None else _verdict_to_dict(request.verdict),
    }
    emit(answer)
    trace = [trace_step_to_dict(t) for t in trace_steps]
    usage = getattr(generator, "last_usage", None)
    emit(
        {
            "type": "done",
            "trace": trace,
            "latency_ms": round(latency_ms, 1),
            "usage": usage,
            "validation_turns": request.generation_attempts,
            "validation_failed": request.validation_failed,
        }
    )
    return {
        "question": question,
        "answer": request.display,
        "sources": sources_out,
        "refused": request.refused,
        "code": True,
        "trace": trace,
        "latency_ms": latency_ms,
        "usage": usage,
        "validation_turns": request.generation_attempts,
        "validation_failed": request.validation_failed,
        "verdict": None if request.verdict is None else _verdict_to_dict(request.verdict),
    }


def _code_step_detail(request) -> str:
    """Human-readable detail for the per-attempt trace step."""
    parts = [f"{len(request.code_blocks)} code block(s)"]
    if request.validation_reasons:
        parts.append("; ".join(request.validation_reasons))
    if request.validation_failed:
        parts.append("validation budget exhausted — refusing to return code")
    return " · ".join(parts)