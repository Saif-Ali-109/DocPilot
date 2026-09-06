"""LLM-backed sufficiency judge (SPEC §4.1, PLAN §3.2).

:class:`LLMSufficiencyJudge` makes **one** structured LLM call per
``judge()`` invocation.  The call is routed through the injected
:class:`~docpilot.generation.generator.Generator` interface — no direct
vendor calls here.

On parse failure the judge returns a defensive ``"sufficient"`` verdict so
the loop never crashes (PLAN §3.2 guard).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from docpilot.agent.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from docpilot.agent.types import Judgment
from docpilot.core.models import RetrieverResult
from docpilot.generation.generator import Generator

logger = logging.getLogger(__name__)


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
    ) -> Judgment:
        """Evaluate whether *results* provide sufficient evidence.

        Args:
            question: The original user question.
            results: The retriever results from this round.
            query_used: The retrieval query that produced *results*.

        Returns:
            A :class:`Judgment` with ``verdict``, ``reason``, and an optional
            ``reformulated_query``.
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

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def judge(
        self,
        question: str,
        results: list[RetrieverResult],
        query_used: str,
    ) -> Judgment:
        """Evaluate retrieval sufficiency with exactly one LLM call.

        On JSON parse failure a defensive ``"sufficient"`` verdict is
        returned so the loop never crashes.
        """
        logger.debug(
            "Judging sufficiency of %d chunk(s) for query %r",
            len(results),
            query_used,
        )
        user_prompt = build_judge_user_prompt(question, query_used, results)
        full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

        raw = self._generator.generate(full_prompt)

        judgment = _parse_judgment(raw)
        logger.debug("Judge verdict=%r reason=%r", judgment.verdict, judgment.reason)
        return judgment


# ---------------------------------------------------------------------------
# JSON parsing (tolerant)
# ---------------------------------------------------------------------------

# Regex that finds candidate ``{…}`` JSON objects — handles markdown-fenced
# output, trailing prose, etc.  Nested objects are not expected (the judge
# returns a flat object).
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Keys the judge's JSON object is expected to contain.
_JUDGE_KEYS = frozenset({"verdict", "reason", "reformulated_query"})


def _parse_judgment(raw: str) -> Judgment:
    """Parse the judge's raw output into a :class:`Judgment`.

    Tolerant parsing order:
    1. Try ``json.loads`` on the raw string (model followed instructions).
    2. Try regex extraction of the first ``{…}`` block that parses and looks
       like judge output (contains ``verdict`` / ``reason`` /
       ``reformulated_query``).
    3. On any failure → defensive ``"sufficient"``.

    Never raises — always returns a valid :class:`Judgment`.
    """
    if not raw or not raw.strip():
        logger.debug("Judge returned empty output — defaulting to sufficient")
        return Judgment(
            verdict="sufficient",
            reason="judge output empty; defaulting to answering",
        )

    # Attempt 1: direct parse
    obj = _try_parse_json(raw)
    if obj is not None:
        return _extract_judgment(obj)

    # Attempt 2: collect candidate JSON objects and use the first one that
    # parses and carries judge keys (skips stray "{}" fragments).
    for match in _JSON_RE.finditer(raw):
        candidate = _try_parse_json(match.group(0))
        if candidate is not None and _JUDGE_KEYS & set(candidate.keys()):
            return _extract_judgment(candidate)

    # Attempt 3: defensive fallback
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

    return Judgment(
        verdict=verdict,
        reason=reason,
        reformulated_query=reformulated_query,
    )
