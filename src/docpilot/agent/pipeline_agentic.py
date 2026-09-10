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
from docpilot.tools import GitHubTool

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = frozenset({"auto", "direct", "agentic"})


def _build_default_judge() -> LLMSufficiencyJudge:
    """Build the production judge over a Groq-backed generator.

    The judge model is ``AGENT_JUDGE_MODEL`` when set, else the shared
    ``GROQ_MODEL`` (SPEC §4.3).  When ``AGENT_JUDGE_SCORE_FLOOR`` is
    enabled, the raw LLM judge is wrapped with
    :class:`ScoreFloorBackstopJudge` (PLAN §H finding 6).
    """
    from docpilot.agent.judge import ScoreFloorBackstopJudge

    model = config.AGENT_JUDGE_MODEL or config.GROQ_MODEL
    judge = LLMSufficiencyJudge(GroqGenerator(model=model))
    if config.AGENT_JUDGE_SCORE_FLOOR > 0.0:
        return ScoreFloorBackstopJudge(judge, config.AGENT_JUDGE_SCORE_FLOOR)
    return judge


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
    tool=None,
    emit=None,
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
        tool: Phase 3 external-data ``Tool`` (e.g. ``GitHubTool``). When
            ``None``, defaults to a production ``GitHubTool()`` if
            ``config.GITHUB_PAT`` is truthy, else ``None`` — with no PAT the
            graph is exactly Phase 2 (judge told tools are unavailable, no
            tool node exists).
        emit: Optional live-view event callback (Phase 5, SPEC §7).  ``None``
            (default) → byte-identical Phase 2–4 behaviour — the CLI and eval
            never pass one.  When set, the gate ``step`` event is emitted
            first, then the graph emits every node's ``step`` event as it
            completes and streams answer deltas as ``token`` events (see
            :func:`docpilot.agent.graph.make_nodes`).

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

    # The gate step is built once and shared by both paths; the emit hook (if
    # any) sees it first so the UI gets the routing decision immediately.
    gate_step = LoopTraceStep.new(
        "gate", question, gate_decision, detail=gate_detail, started_at=gate_started
    )
    if emit is not None:
        emit({"type": "step", "step": trace_step_to_dict(gate_step)})

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
    # Phase 3: default to the production GitHub tool only when a PAT exists;
    # otherwise keep tool=None so the loop is byte-identical to Phase 2 (the
    # judge is told tools are unavailable and no tool node exists).
    if tool is None and config.GITHUB_PAT:
        tool = GitHubTool()

    app = build_graph(
        retriever=retriever,
        judge=judge,
        generator=generator,
        citation_engine=citation_engine,
        top_k=loop_top_k,
        language=None if resolved_language == "any" else resolved_language,
        max_retries=max_retries,
        tool=tool,
        emit=emit,
        judge_skip_score=config.AGENT_JUDGE_SKIP_MIN_SCORE,
    )
    initial: AgentLoopState = {
        "question": question,
        "original_question": question,
        "current_query": question,
        "results": [],
        "sources": [],
        "attempts": 0,
        "trace": [trace_step_to_dict(gate_step)],
        "tool_request": None,
        "tool_results": [],
        "tool_error": None,
        "answer": None,
        "refused": False,
        "direct": False,
        "skip_judge": False,
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
        tool=None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._judge = judge
        self._citation_engine = citation_engine
        self._strategy = strategy
        self._top_k = top_k
        self._language = language
        self._max_retries = max_retries
        self._tool = tool

    def run(self, question: str, **kwargs) -> AgentResult:
        """Answer *question*, honouring per-call overrides in **kwargs.

        Recognised override keys: ``retriever``, ``generator``, ``judge``,
        ``citation_engine``, ``strategy``, ``top_k``, ``language``,
        ``max_retries``, ``tool``, ``emit``.
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
            "tool": kwargs.get("tool", self._tool),
            "emit": kwargs.get("emit", None),
        }
        return agentic_ask(question, **params)
