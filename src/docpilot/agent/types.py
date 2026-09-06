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
    """

    verdict: str
    reason: str
    reformulated_query: str | None = None


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class LoopTraceStep:
    """One decision point in the agentic loop (Phase 2's only trace mechanism).

    Attributes:
        step: One of ``"gate"``, ``"search"``, ``"judge"``, ``"answer"``,
            ``"refuse"``.
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
    answer: str | None
    refused: bool
    direct: bool
