"""Phase 2 shared contract — write-once, read-only for all other modules.

AGENT G imports everything from here; no langgraph imports anywhere in this
file.  All types are plain dataclasses / TypedDict so the graph wire-up can
serialise them without pulling in third-party state frameworks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypedDict

from docpilot.core.models import Chunk, RetrieverResult, SourceRef
from docpilot.tools.base import ToolResult

# ---------------------------------------------------------------------------
# Budget constant (SPEC §4.1 — never hard-coded elsewhere)
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 2
"""Maximum number of judge/reformulate iterations before refusing."""


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass
class GateDecision:
    """Output of the heuristic query classifier.

    Attributes:
        agentic: ``True`` when the question should be routed through the
            agentic loop; ``False`` for the direct / fast path.
        signals: Which heuristic signals fired (empty when the question is
            simple and goes straight to the fast path).
        reason: One-line human-readable explanation of the decision.
    """

    agentic: bool
    signals: list[str]
    reason: str


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


@dataclass
class Judgment:
    """Structured output of the sufficiency judge.

    Attributes:
        verdict: Exactly ``"sufficient"`` or ``"insufficient"``.
        reason: Human-readable explanation from the judge.
        reformulated_query: A rewritten / expanded retrieval query when the
            evidence is insufficient and a retry is warranted.  ``None`` when
            the current query is already adequate (or when the verdict is
            sufficient).
        needs_tool: ``True`` only when the verdict is ``"insufficient"`` and a
            live external call (Phase 3, e.g. GitHub) could genuinely provide
            evidence static docs cannot (live issue state, repo state, recent
            commits).  Default ``False`` — doc-content gaps retry the
            retrievers instead.
        tool_request: The validated tool invocation
            ``{"name": "<github action>", "params": {...}}`` the agent should
            run when ``needs_tool`` fires; ``None`` otherwise.  Only meaningful
            together with ``needs_tool=True``; the graph also refuses to fire
            the tool when no tool is wired.
    """

    verdict: str
    reason: str
    reformulated_query: str | None = None
    needs_tool: bool = False
    tool_request: dict | None = None


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class LoopTraceStep:
    """One decision point in the agentic loop (Phase 2's only trace mechanism).

    Attributes:
        step: One of ``"gate"``, ``"search"``, ``"judge"``, ``"tool_call"``,
            ``"answer"``, ``"refuse"``.
        query: The query string used at this step.
        decision: Short label, e.g. ``"agentic"``, ``"direct"``,
            ``"sufficient"``, ``"insufficient/retry-1"``, ``"refuse"``.
        detail: Optional extra information.
        latency_ms: Wall-clock time for this step (inspectability only;
            no aggregates / comparisons in Phase 2).
    """

    step: str
    query: str
    decision: str
    detail: str | None = None
    latency_ms: int | None = None

    @classmethod
    def new(
        cls,
        step: str,
        query: str,
        decision: str,
        detail: str | None = None,
        started_at: float | None = None,
    ) -> "LoopTraceStep":
        """Build a trace step, optionally tagging it with its latency.

        Args:
            step: Step label (``"gate"``, ``"search"``, ``"judge"``,
                ``"answer"``, ``"refuse"``).
            query: Query string used at this step.
            decision: Short decision label (e.g. ``"agentic"``,
                ``"sufficient"``, ``"refuse"``).
            detail: Optional extra information.
            started_at: A ``time.perf_counter()`` timestamp captured before
                the step ran; when given, ``latency_ms`` is computed from it.
                ``None`` → latency left unset (inspectability only).
        """
        latency_ms = None
        if started_at is not None:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
        return cls(
            step=step,
            query=query,
            decision=decision,
            detail=detail,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# LangGraph state shape  (AGENT G builds the StateGraph around this)
# ---------------------------------------------------------------------------


class AgentLoopState(TypedDict, total=False):
    """State dictionary shared across graph nodes.

    AGENT G reads / writes these fields via the LangGraph StateGraph.  Every
    field is serialisable (no custom classes) so the graph can persist /
    checkpoint state without vendor-specific serialisation.

    Attributes:
        question: The user's original question.
        original_question: Preserved copy of *question* (never modified).
        current_query: The query used for the current retrieval round
            (may differ from *question* after judge reformulation).
        results: Serialised :class:`RetrieverResult` list — each entry is a
            dict ``{content, score, source_file, heading}``.
        sources: Serialised :class:`SourceRef` list — each entry is a dict
            ``{ref, file, heading}``.
        attempts: How many judge calls have been made so far.
        trace: Serialised :class:`LoopTraceStep` list.
        tool_request: Validated tool invocation ``{"name", "params"}`` stashed
            by the judge node when ``needs_tool`` fired on a within-budget
            round (``None`` otherwise — reset every round).
        tool_results: Serialised :class:`ToolResult` list (``ok`` results from
            the Phase 3 tool call, consumed by the answer node's
            ``LIVE GITHUB EVIDENCE`` context section).
        tool_error: Error message when the tool call failed (``ok=False``);
            ``None`` when no tool ran or the tool succeeded — routes the graph
            to ``refuse`` / ``answer`` respectively.
        answer: The final answer string (``None`` until the graph reaches
            the ``answer`` or ``refuse`` node).
        refused: ``True`` when the budget is exhausted and the agent says
            "I don't know".
        direct: ``True`` when the gate decided the fast path (the loop was
            never entered).
    """

    question: str
    original_question: str
    current_query: str
    results: list[dict]
    sources: list[dict]
    attempts: int
    trace: list[dict]
    tool_request: dict | None
    tool_results: list[dict]
    tool_error: str | None
    answer: str | None
    refused: bool
    direct: bool
    # Phase 5 hardening (SPEC §7): set by the retrieve node when a
    # configured judge-skip threshold is cleared — the judge LLM call is
    # skipped and the router answers directly.  Default False; only ever
    # True when AGENT_JUDGE_SKIP_MIN_SCORE > 0.
    skip_judge: bool


# ---------------------------------------------------------------------------
# Tool result serialisation (Phase 3 — graph state stores plain dicts)
# ---------------------------------------------------------------------------


def tool_result_to_dict(result: ToolResult) -> dict:
    """Serialise a :class:`ToolResult` to a plain dict for graph state.

    Fields needed downstream: ``ok`` / ``summary`` / ``error`` for routing and
    the answer node's ``LIVE GITHUB EVIDENCE`` context section, plus the
    per-item records (carrying the per-record ``source_label`` the tool_call
    node turns into :class:`SourceRef` citation entries).
    """
    return {
        "ok": result.ok,
        "summary": result.summary,
        "error": result.error,
        "source_label": result.source_label,
        "items": list(result.items),
    }


def dict_to_tool_result(d: dict) -> ToolResult:
    """Rebuild a :class:`ToolResult` from a graph-state dict."""
    return ToolResult(
        ok=bool(d.get("ok", False)),
        summary=d.get("summary", ""),
        error=d.get("error"),
        source_label=d.get("source_label"),
        items=list(d.get("items") or []),
    )
