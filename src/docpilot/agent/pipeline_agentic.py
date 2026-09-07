"""Agentic vs. direct orchestration entry point (SPEC §4, PLAN §3.2).

:func:`agentic_ask` is the phase-2 orchestrator and the CLI / tests entry
point. It applies the strategy (`auto|direct|agentic`), then either:

    * **direct** (or ``auto``-with-simple-gate) — delegates straight to the
      untouched Phase 1 :func:`docpilot.pipeline_ask.ask` fast path and wraps
      its output into an :class:`AgentResult`. Output is byte-identical to a
      Phase 1 ``ask()`` (``direct=True``, gate-only trace).
    * **agentic** (or ``auto``-with-complex-gate) — runs the compiled LangGraph
      loop over an :class:`AgentLoopState` and converts the final state back
      into an :class:`AgentResult` (full trace, ``direct=False``).

The gate decision is always recorded first in the trace, with one of the
decisions ``"direct"`` / ``"agentic"`` / ``"forced-direct"`` /
``"forced-agentic"``.
"""

from __future__ import annotations

import logging
import time

from docpilot import config
from docpilot.agent.gate import HeuristicQueryClassifier
from docpilot.agent.graph import (
    build_graph,
    dict_to_source,
    dict_to_trace_step,
    trace_step_to_dict,
)
from docpilot.agent.interface import Agent, AgentResult
from docpilot.agent.judge import LLMSufficiencyJudge
from docpilot.agent.types import DEFAULT_MAX_RETRIES, AgentLoopState, LoopTraceStep
from docpilot.citations.engine import StandardCitationEngine
from docpilot.generation.generator import GroqGenerator

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = frozenset({"auto", "direct", "agentic"})


def _build_default_judge() -> LLMSufficiencyJudge:
    """Build the production judge over a Groq-backed generator.

    The judge model is ``AGENT_JUDGE_MODEL`` when set, else the shared
    ``GROQ_MODEL`` (SPEC §4.3).
    """
    model = config.AGENT_JUDGE_MODEL or config.GROQ_MODEL
    return LLMSufficiencyJudge(GroqGenerator(model=model))


def _resolve_language(language: str | None) -> str:
    """Mirror Phase 1 language resolution (SPEC §3.11, PLAN §3.6)."""
    return language if language is not None else config.RETRIEVAL_LANGUAGE


def agentic_ask(
    question: str,
    *,
    retriever=None,
    generator=None,
    judge=None,
    citation_engine=None,
    strategy: str = "auto",
    top_k: int | None = None,
    language: str | None = None,
    max_retries: int | None = None,
) -> AgentResult:
    """Answer *question* under the chosen strategy.

    Args:
        question: The user's natural-language question.
        retriever: A ``Retriever`` (default: ``SimpleRetriever`` + pgvector).
        generator: A ``Generator`` (default: ``GroqGenerator``).
        judge: A ``SufficiencyJudge`` (default: ``LLMSufficiencyJudge`` over
            ``GroqGenerator``); only used on the agentic path.
        citation_engine: A ``CitationEngine`` (default: ``StandardCitationEngine``).
        strategy: ``"auto"`` (gate decides), ``"direct"`` (force fast path) or
            ``"agentic"`` (force the loop).
        top_k: Retrieval count.  On the direct/fast path defaults to
            ``config.RETRIEVAL_TOP_K``; on the agentic loop path defaults to
            ``config.AGENT_LOOP_TOP_K`` (a caller-supplied value overrides
            both).
        language: Retrieval language filter; defaults to
            ``config.RETRIEVAL_LANGUAGE``; ``"any"`` disables filtering.
        max_retries: Hard judge budget; defaults to ``config.AGENT_MAX_RETRIES``
            (falling back to ``agent.types.DEFAULT_MAX_RETRIES``).

    Returns:
        An :class:`AgentResult` whose ``answer`` is the final display string
        (answer + footer on the answering path, or the verbatim SPEC §3.9
        refusal sentence when refused) and whose ``trace`` always starts with
        the gate step.

    Raises:
        ValueError: If *strategy* is not ``auto``/``direct``/``agentic``.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of 'auto', 'direct', 'agentic'; got {strategy!r}"
        )

    # Fast/direct path keeps RETRIEVAL_TOP_K; the agentic loop broadens to
    # AGENT_LOOP_TOP_K unless the caller explicitly overrode ``top_k``.
    fast_top_k = top_k if top_k is not None else config.RETRIEVAL_TOP_K
    loop_top_k = top_k if top_k is not None else config.AGENT_LOOP_TOP_K
    resolved_language = _resolve_language(language)
    max_retries = max_retries if max_retries is not None else (
        config.AGENT_MAX_RETRIES or DEFAULT_MAX_RETRIES
    )

    gate = HeuristicQueryClassifier()
    decision = gate.classify(question)

    forced_direct = strategy == "direct"
    forced_agentic = strategy == "agentic"
    agentic = forced_agentic or (strategy == "auto" and decision.agentic)

    # ── gate trace step (always recorded first) ──────────────────────────
    if agentic:
        if forced_agentic:
            gate_decision = "forced-agentic"
        else:
            gate_decision = "agentic"
    else:
        gate_decision = "forced-direct" if forced_direct else "direct"
    gate_detail = (
        f"signals={decision.signals}; {decision.reason}" if decision.signals else decision.reason
    )
    gate_started = time.perf_counter()

    if not agentic:
        # ── fast path: identical to Phase 1 ask() ───────────────────────
        from docpilot.pipeline_ask import ask

        result = ask(
            question,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            top_k=fast_top_k,
            language=resolved_language,
        )
        gate_step = LoopTraceStep.new(
            "gate", question, gate_decision, detail=gate_detail, started_at=gate_started
        )
        logger.debug("Gate decision: %s → direct fast path", gate_decision)
        return AgentResult(
            question=question,
            answer=result.display,
            sources=result.sources,
            trace=[gate_step],
            refused=False,
            max_retries=max_retries,
            direct=True,
        )

    # ── agentic path: run the compiled graph ─────────────────────────────
    conn = None
    if retriever is None:
        from docpilot.pipeline_ask import _build_default_retriever

        retriever, conn = _build_default_retriever()
    if generator is None:
        generator = GroqGenerator()
    if citation_engine is None:
        citation_engine = StandardCitationEngine()
    if judge is None:
        judge = _build_default_judge()

    gate_step = LoopTraceStep.new(
        "gate", question, gate_decision, detail=gate_detail, started_at=gate_started
    )
    app = build_graph(
        retriever=retriever,
        judge=judge,
        generator=generator,
        citation_engine=citation_engine,
        top_k=loop_top_k,
        language=None if resolved_language == "any" else resolved_language,
        max_retries=max_retries,
    )
    initial: AgentLoopState = {
        "question": question,
        "original_question": question,
        "current_query": question,
        "results": [],
        "sources": [],
        "attempts": 0,
        "trace": [trace_step_to_dict(gate_step)],
        "answer": None,
        "refused": False,
        "direct": False,
    }
    try:
        final = app.invoke(initial)
    finally:
        if conn is not None:
            conn.close()

    logger.debug("Gate decision: %s → agentic loop engaged", gate_decision)
    return AgentResult(
        question=question,
        answer=final["answer"],
        sources=[dict_to_source(d) for d in final["sources"]],
        trace=[dict_to_trace_step(d) for d in final["trace"]],
        refused=final["refused"],
        max_retries=max_retries,
        direct=False,
    )


class AgenticAgent(Agent):
    """Agent implementation backed by :func:`agentic_ask`.

    Bundles default components and strategy; per-call overrides may be passed
    to :meth:`run`. Simple questions keep routing through the Phase 1 fast
    path (the ``auto`` gate), so the loop is only engaged when warranted.
    """

    def __init__(
        self,
        *,
        retriever=None,
        generator=None,
        judge=None,
        citation_engine=None,
        strategy: str = "auto",
        top_k: int | None = None,
        language: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._judge = judge
        self._citation_engine = citation_engine
        self._strategy = strategy
        self._top_k = top_k
        self._language = language
        self._max_retries = max_retries

    def run(self, question: str, **kwargs) -> AgentResult:
        """Answer *question*, honouring per-call overrides in **kwargs.

        Recognised override keys: ``retriever``, ``generator``, ``judge``,
        ``citation_engine``, ``strategy``, ``top_k``, ``language``,
        ``max_retries``.
        """
        params = {
            "retriever": kwargs.get("retriever", self._retriever),
            "generator": kwargs.get("generator", self._generator),
            "judge": kwargs.get("judge", self._judge),
            "citation_engine": kwargs.get("citation_engine", self._citation_engine),
            "strategy": kwargs.get("strategy", self._strategy),
            "top_k": kwargs.get("top_k", self._top_k),
            "language": kwargs.get("language", self._language),
            "max_retries": kwargs.get("max_retries", self._max_retries),
        }
        return agentic_ask(question, **params)
