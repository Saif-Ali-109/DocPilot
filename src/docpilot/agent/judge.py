"""LLM-backed sufficiency judge (SPEC §4.1, PLAN §3.2).

:class:`LLMSufficiencyJudge` makes **one** structured LLM call per
``judge()`` invocation.  The call is routed through the injected
:class:`~docpilot.generation.generator.Generator` interface — no direct
vendor calls here.

On parse failure the judge returns a defensive ``"sufficient"`` verdict so
the loop never crashes (PLAN §3.2 guard).  Every fallback is also counted
per cause and logged at INFO — the Phase 4 flip-condition signal that the
judge's output is drifting from the JSON contract (SPEC §6.3).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter

from docpilot.agent.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from docpilot.agent.types import Judgment
from docpilot.core.models import RetrieverResult
from docpilot.generation.generator import Generator

logger = logging.getLogger(__name__)

# Per-cause count of defensive "sufficient" parse fallbacks, keyed by cause
# ("empty", "unparseable", "bad_verdict").  Accumulates over the judge's
# lifetime in this process; inspect via :func:`judge_parse_fallback_counts`
# and reset with :func:`reset_judge_parse_fallback_counts`.
_PARSE_FALLBACK_COUNTER: Counter[str] = Counter()


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class SufficiencyJudge(ABC):
    """Interface for retrieval-sufficiency evaluation."""

    @abstractmethod
    def judge(
        self,
        question: str,
        results: list[RetrieverResult],
        query_used: str,
        *,
        tools_available: bool = True,
    ) -> Judgment:
        """Evaluate whether *results* provide sufficient evidence.

        Args:
            question: The original user question.
            results: The retriever results from this round.
            query_used: The retrieval query that produced *results*.
            tools_available: Whether a Phase 3 external tool is wired into the
                graph — when ``False`` the judge must keep ``needs_tool``
                false (the graph would have nothing to call).

        Returns:
            A :class:`Judgment` with ``verdict``, ``reason``, an optional
            ``reformulated_query`` and (Phase 3) an optional ``needs_tool`` /
            ``tool_request`` pair.
        """
        ...


# ---------------------------------------------------------------------------
# LLM implementation
# ---------------------------------------------------------------------------


class LLMSufficiencyJudge(SufficiencyJudge):
    """LLM-backed judge — one call per ``judge()`` invocation.

    All LLM interaction goes through the injected ``generator`` (the only
    route to the LLM in Phase 2).  The judge prompt is built via helpers
    in :mod:`docpilot.agent.prompts`.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        system_prompt: str | None = None,
    ) -> None:
        """Build the judge over *generator*.

        Args:
            generator: The ``Generator`` the judge's single LLM call is
                routed through (SPEC §4.1).
            system_prompt: The judge system prompt to use; defaults to
                :data:`~docpilot.agent.prompts.JUDGE_SYSTEM_PROMPT`.  Pass a
                variant (e.g. ``JUDGE_SYSTEM_PROMPT_B``) for the Phase 4
                two-prompt calibration run (SPEC §6.2).
        """
        self._generator = generator
        self._system_prompt = (
            system_prompt if system_prompt is not None else JUDGE_SYSTEM_PROMPT
        )

    def judge(
        self,
        question: str,
        results: list[RetrieverResult],
        query_used: str,
        *,
        tools_available: bool = True,
    ) -> Judgment:
        """Evaluate retrieval sufficiency with exactly one LLM call.

        On JSON parse failure a defensive ``"sufficient"`` verdict is
        returned so the loop never crashes.
        """
        logger.debug(
            "Judging sufficiency of %d chunk(s) for query %r (tools_available=%r)",
            len(results),
            query_used,
            tools_available,
        )
        user_prompt = build_judge_user_prompt(
            question, query_used, results, tools_available=tools_available
        )

        raw = self._generator.generate_system_user(self._system_prompt, user_prompt)

        judgment = _parse_judgment(raw)
        logger.debug(
            "Judge verdict=%r reason=%r needs_tool=%r tool_request=%r",
            judgment.verdict,
            judgment.reason,
            judgment.needs_tool,
            judgment.tool_request,
        )
        return judgment


class ScoreFloorBackstopJudge(SufficiencyJudge):
    """Score-floor sanity backstop around a wrapped :class:`SufficiencyJudge`.

    An LLM-as-judge is vulnerable to prompt drift and overconfidence on weak
    retrieval.  This wrapper adds a cheap, non-LLM cross-check (pre-Phase-6
    hardening finding): when the *top retrieval score* sits below a
    configurable floor, the verdict is forced to ``"insufficient"`` so the
    loop can never route straight to ``answer`` on thin evidence — exactly
    the "I don't know" trustworthiness gap SPEC §6.3 targets.

    Behaviour matrix:

    * ``score_floor <= 0.0`` (default) → pass-through, zero behaviour change.
    * top score ≥ floor → pass-through (the LLM verdict stands).
    * top score < floor → verdict forced to ``"insufficient"``; the wrapped
      judge's ``needs_tool`` / ``tool_request`` / ``reformulated_query`` are
      **preserved** so live-state questions can still reach the GitHub tool /
      retry — the backstop only kills unconditional answering on weak
      evidence, it never blocks the tool or reformulation path.
    * empty ``results`` → verdict forced to ``"insufficient"`` when a floor
      is active (nothing to answer from).
    """

    def __init__(self, inner: SufficiencyJudge, score_floor: float) -> None:
        """Wrap *inner* with a top-score floor.

        Args:
            inner: The LLM (or any) judge whose verdicts get sanity-checked.
            score_floor: Minimum acceptable top retrieval score.  ``0.0`` or
                negative disables the backstop (pure pass-through).
        """
        self._inner = inner
        self._score_floor = float(score_floor)

    def judge(
        self,
        question: str,
        results: list[RetrieverResult],
        query_used: str,
        *,
        tools_available: bool = True,
    ) -> Judgment:
        """Apply the score floor, then delegate to the wrapped judge."""
        judgment = self._inner.judge(
            question, results, query_used, tools_available=tools_available
        )
        if self._score_floor <= 0.0:
            return judgment

        top_score = results[0].score if results else None
        overridden = top_score is None or top_score < self._score_floor
        if not overridden:
            return judgment

        logger.info(
            "Judge backstop: top_score=%r < floor=%r — forcing insufficient "
            "(wrapped verdict=%r, needs_tool=%r)",
            top_score,
            self._score_floor,
            judgment.verdict,
            judgment.needs_tool,
        )
        return Judgment(
            verdict="insufficient",
            reason=(
                f"score floor {self._score_floor} not met "
                f"(top_score={top_score}); LLM verdict was {judgment.verdict!r}"
            ),
            # Preserve the wrapped judge's recovery signals — the backstop
            # only neutralises unconditional answering on weak evidence.
            reformulated_query=judgment.reformulated_query,
            needs_tool=judgment.needs_tool,
            tool_request=judgment.tool_request,
        )


# ---------------------------------------------------------------------------
# JSON parsing (tolerant)
# ---------------------------------------------------------------------------

# Regex that finds candidate ``{…}`` JSON objects — handles markdown-fenced
# output, trailing prose, etc.  Since Phase 3 the judge's object may carry a
# nested ``tool_request`` dict, so the pattern tolerates exactly one level of
# ``{…}`` nesting (the documented ``params`` shape has scalar/JSON-ish values).
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

# Matches a ```json …``` fenced block (also bare ``` fences).
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# Keys the judge's JSON object is expected to contain.
_JUDGE_KEYS = frozenset(
    {"verdict", "reason", "reformulated_query", "needs_tool", "tool_request"}
)


def judge_parse_fallback_counts() -> dict[str, int]:
    """Snapshot of per-cause parse-fallback counts (read-only copy).

    Causes: ``empty`` (blank raw output), ``unparseable`` (none of the
    tolerant-parse attempts succeeded), ``bad_verdict`` (JSON parsed but the
    ``verdict`` value was neither ``sufficient`` nor ``insufficient``).
    """
    return dict(_PARSE_FALLBACK_COUNTER)


def reset_judge_parse_fallback_counts() -> None:
    """Reset the parse-fallback counter (test/phase boundary hook)."""
    _PARSE_FALLBACK_COUNTER.clear()


def _record_parse_fallback(cause: str) -> None:
    """Record one defensive fallback and log it at INFO.

    The INFO line is deliberately stable — the Phase 4 flip-condition check
    (SPEC §6.3) greps for it:

        Judge parse fallback: cause=<cause> -> sufficient, total=<n>

    where ``total`` is the running count across all causes in this process.
    """
    _PARSE_FALLBACK_COUNTER[cause] += 1
    logger.info(
        "Judge parse fallback: cause=%s -> sufficient, total=%d",
        cause,
        sum(_PARSE_FALLBACK_COUNTER.values()),
    )


def _parse_judgment(raw: str) -> Judgment:
    """Parse the judge's raw output into a :class:`Judgment`.

    Tolerant parsing order:
    1. Try ``json.loads`` on the raw string (model followed instructions).
    2. Try ``json.loads`` after stripping a markdown fence (covers nested
       ``tool_request`` dicts the flat-object regex cannot capture).
    3. Try regex extraction of the first ``{…}`` block that parses and looks
       like judge output (contains any of the judge keys).
    4. On any failure → defensive ``"sufficient"``.

    Never raises — always returns a valid :class:`Judgment`.
    """
    if not raw or not raw.strip():
        _record_parse_fallback("empty")
        logger.debug("Judge returned empty output — defaulting to sufficient")
        return Judgment(
            verdict="sufficient",
            reason="judge output empty; defaulting to answering",
        )

    # Attempt 1: direct parse
    obj = _try_parse_json(raw)
    if obj is not None:
        return _extract_judgment(obj)

    # Attempt 2: direct parse after stripping a ```json …``` fence.
    unfenced = _FENCE_RE.sub(r"\1", raw.strip())
    if unfenced != raw:
        obj = _try_parse_json(unfenced)
        if obj is not None:
            return _extract_judgment(obj)

    # Attempt 3: collect candidate JSON objects and use the first one that
    # parses and carries judge keys (skips stray "{}" fragments).
    for match in _JSON_RE.finditer(raw):
        candidate = _try_parse_json(match.group(0))
        if candidate is not None and _JUDGE_KEYS & set(candidate.keys()):
            return _extract_judgment(candidate)

    # Attempt 4: defensive fallback
    _record_parse_fallback("unparseable")
    logger.debug("Judge output unparseable — defaulting to sufficient")
    return Judgment(
        verdict="sufficient",
        reason="judge output unparseable; default to answering",
    )


def _try_parse_json(text: str) -> dict | None:
    """Attempt ``json.loads``; return ``None`` on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_judgment(obj: dict) -> Judgment:
    """Build a :class:`Judgment` from a parsed dict.

    Missing or malformed keys fall back to safe defaults.
    """
    verdict = obj.get("verdict", "")
    if verdict not in ("sufficient", "insufficient"):
        # Unknown verdict — lean toward sufficient (PLAN §3.2 guard)
        _record_parse_fallback("bad_verdict")
        logger.debug("Unknown judge verdict %r — defaulting to sufficient", verdict)
        verdict = "sufficient"

    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    reformulated_query = obj.get("reformulated_query")
    if reformulated_query is not None and not isinstance(reformulated_query, str):
        reformulated_query = None
    # Empty string → None (no reformulation needed)
    if reformulated_query == "":
        reformulated_query = None

    # Phase 3 fields — tolerant, never raising.
    needs_tool = _parse_needs_tool(obj.get("needs_tool"))
    tool_request = _parse_tool_request(obj.get("tool_request"))

    # Cross-checks keep the Judgment self-consistent with the graph routing:
    # * a tool_request without needs_tool is meaningless → drop it;
    # * needs_tool=true rules out a reformulated_query — the tool replies with
    #   live evidence and never loops back to the judge.
    if needs_tool:
        reformulated_query = None
    else:
        tool_request = None

    return Judgment(
        verdict=verdict,
        reason=reason,
        reformulated_query=reformulated_query,
        needs_tool=needs_tool,
        tool_request=tool_request,
    )


def _parse_needs_tool(value: object) -> bool:
    """Coerce the judge's ``needs_tool`` output to a bool.

    Accepts JSON booleans and common truthy/falsy strings; anything malformed
    (or missing) defaults to ``False`` so the loop never misroutes on prompt
    drift.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y")
    return False


def _parse_tool_request(value: object) -> dict | None:
    """Validate the judge's ``tool_request`` shape.

    Must be a dict with a non-empty string ``name`` and a dict ``params``;
    anything else → ``None`` (safe default).  The action name itself is *not*
    validated here — an unknown action surfaces as a non-``ok`` ToolResult at
    execution time and routes to the refusal path.
    """
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    params = value.get("params")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(params, dict):
        return None
    return {"name": name, "params": params}
