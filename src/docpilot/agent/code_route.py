"""Code-route dispatch for Phase 6 (PLAN §7.2 T3, SPEC §8).

The code route is an **explicit opt-in** that never shadows the standard
answer path.  A query reaches code generation only when the dispatch says so:

1. a per-request ``explicit`` opt-in always selects the code route; or
2. the route is armed (``config.CODE_ROUTE_ENABLED`` / caller ``enabled``)
   **and** the query looks like a code request (deterministic intent-phrase
   match, zero LLM calls — mirroring the Phase 2 heuristic gate); otherwise
3. the request falls back to the standard ask path, byte-identical to a
   plain ``ask()`` (§7.5: non-code queries never produce a code answer).

:func:`run_code_route` is the gated route: it runs the T2
:func:`docpilot.codegen.pipeline_ask_code.ask_code` pipeline, validates every
emitted code block against the retrieved evidence with the T1 validator, and
runs the T4 reformulation loop (capped) — on a failed verdict the failure
reasons are fed back for a rewrite until the code is fully validated or the
budget is exhausted, in which case the route refuses with the "couldn't
validate" message plus the retrieved sources (§7.5: unvalidated code is never
returned).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docpilot import config
from docpilot.agent.gate import _normalise
from docpilot.validation.verdict import (
    CheckStatus,
    ValidationVerdict,
    combine_verdicts,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from docpilot.codegen.pipeline_ask_code import CodeRequest
    from docpilot.pipeline_ask import AskResult
    from docpilot.validation.validator import CodeValidator


@dataclass(frozen=True)
class CodeRouteDecision:
    """Outcome of the code-route dispatch for one query."""

    code: bool
    reason: str


class CodeIntentClassifier(ABC):
    """Interface for deciding whether a query itself is a code request."""

    @abstractmethod
    def classify(self, question: str) -> CodeRouteDecision:
        """Return ``code=True`` iff *question* looks like a code request."""
        ...


class HeuristicCodeIntentClassifier(CodeIntentClassifier):
    """Zero-LLM code-intent detection: seed-phrase match on the normalised
    query.

    Deterministic and inspectable, matching the Phase 2 gate's style
    (``agent/gate.py``).  Phrase list is ``config.CODE_INTENT_PHRASES``
    (documented seed, tuned like the gate's connector vocabulary) and
    overridable per instance for tests.
    """

    def __init__(self, phrases: tuple[str, ...] | None = None) -> None:
        self._phrases = phrases if phrases is not None else config.CODE_INTENT_PHRASES

    def classify(self, question: str) -> CodeRouteDecision:
        if not question or not question.strip():
            return CodeRouteDecision(False, "empty input")
        normalised = _normalise(question)
        for phrase in self._phrases:
            if phrase in normalised:
                return CodeRouteDecision(True, f"code-intent phrase: {phrase!r}")
        return CodeRouteDecision(False, "no code-intent phrase in question")


def decide_code_route(
    question: str,
    *,
    enabled: bool | None = None,
    explicit: bool = False,
    classifier: CodeIntentClassifier | None = None,
) -> CodeRouteDecision:
    """Dispatch decision: should *question* take the code route?

    Rules (PLAN §7.2 T3, §7.5):

        1. ``explicit`` per-request opt-in always selects the code route
           (a caller asserting code intent is by definition not "the default
           answer path");
        2. otherwise the route must be armed — ``config.CODE_ROUTE_ENABLED``
           or the caller's ``enabled`` override — AND a code-intent match
           must fire; and
        3. anything else stays on the standard answer path.

    The decision is pure and deterministic; the ``reason`` always explains
    the outcome for the debug/trace layer.
    """
    if explicit:
        return CodeRouteDecision(True, "explicit code opt-in")
    armed = config.CODE_ROUTE_ENABLED if enabled is None else enabled
    if not armed:
        return CodeRouteDecision(False, "code route not armed (CODE_ROUTE_ENABLED off)")
    classifier = classifier or HeuristicCodeIntentClassifier()
    return classifier.classify(question)


@dataclass
class CodeRouteResult:
    """Outcome of one routed request: the standard ask result or the code
    result, whichever the dispatch selected."""

    decision: CodeRouteDecision
    ask: AskResult | None = None
    code: CodeRequest | None = None

    @property
    def took_code_route(self) -> bool:
        """True when the request actually ran through the code pipeline."""
        return self.code is not None

    @property
    def answer(self) -> str:
        """The display string for whichever route ran (else ``""``)."""
        if self.code is not None:
            return self.code.display
        if self.ask is not None:
            return self.ask.display
        return ""


def run_code_route(
    question: str,
    *,
    enabled: bool | None = None,
    explicit_code: bool = False,
    retriever=None,
    generator=None,
    citation_engine=None,
    validator: CodeValidator | None = None,
    max_validation_turns: int | None = None,
    top_k: int | None = None,
    language: str | None = None,
) -> CodeRouteResult:
    """Route *question* through the validated code path only when allowed.

    Non-code requests fall back to the standard answer path, byte-identical
    to a plain ``ask()`` call (§7.5 — the fast/cheap path never changes).

    On the code route the T4 loop runs: generate → validate every emitted
    code block against the retrieved evidence (T1) → if a verdict fails,
    feed the failure reasons back for a rewrite (``CODE_FIX_PROMPT``) — up to
    ``max_validation_turns`` reformulations, then **refuse** with the
    "couldn't validate" message plus the retrieved sources (§7.5: unvalidated
    code is never returned).  The failed candidate remains attached to the
    :class:`CodeRequest` for the debug layer.

    Args:
        question: The user's question / code request.
        enabled: Override for ``config.CODE_ROUTE_ENABLED`` (arm lever).
        explicit_code: Per-request opt-in — the caller asserts code intent.
        validator: A ``CodeValidator``; ``None`` defaults to the structural
            ``RetrieveThenValidate``.  Pass the explicit ``None`` sentinel
            only to get T2-style unvalidated output (tests / pre-T4 callers).
        max_validation_turns: Reformulation budget — ``None`` reads
            ``config.CODE_VALIDATE_MAX_TURNS`` (2).  Total generation attempts
            on the code route = 1 + max turns.
        retriever / generator / citation_engine: Injectable components
            (defaults build the production PG/Groq stack, lazily).
        top_k / language: Retrieval knobs, same semantics as ``ask()``.

    Returns:
        A :class:`CodeRouteResult`: ``code`` set on the code route (validated
        or validation-refused with sources), else ``ask`` set.
    """
    decision = decide_code_route(
        question, enabled=enabled, explicit=explicit_code
    )
    if not decision.code:
        from docpilot.pipeline_ask import ask

        result = ask(
            question,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            top_k=top_k,
            language=language,
        )
        return CodeRouteResult(decision=decision, ask=result)

    if validator is None:
        from docpilot.validation.validator import RetrieveThenValidate

        validator = RetrieveThenValidate()
    if max_validation_turns is None:
        max_validation_turns = config.CODE_VALIDATE_MAX_TURNS

    request = _run_code_validation_loop(
        question,
        retriever=retriever,
        generator=generator,
        citation_engine=citation_engine,
        validator=validator,
        max_turns=max_validation_turns,
        top_k=top_k,
        language=language,
    )
    return CodeRouteResult(decision=decision, code=request)


def _run_code_validation_loop(
    question: str,
    *,
    retriever,
    generator,
    citation_engine,
    validator: CodeValidator,
    max_turns: int,
    top_k: int | None,
    language: str | None,
) -> CodeRequest:
    """The T4 loop: generate → validate → reformulate, capped, then refuse.

    Returns the final :class:`CodeRequest`.  Outcomes:

        * fully validated (every check ``PASS``) → returned as-is;
        * model refused / emitted no code → returned as-is (``refused``);
        * still-failing after the budget → :meth:`CodeRequest.refuse_code`
          replaces the returned answer with ``VALIDATION_REFUSAL`` + sources.
    """
    from docpilot.codegen.pipeline_ask_code import ask_code

    reasons: list[str] = []
    request: CodeRequest | None = None

    for turn in range(max_turns + 1):
        request = ask_code(
            question,
            retriever=retriever,
            generator=generator,
            citation_engine=citation_engine,
            top_k=top_k,
            language=language,
            fix_reasons=reasons,
        )
        if request.code_blocks:
            evidence = [r.chunk.content for r in request.results]
            request.block_verdicts = [
                validator.validate(block, evidence) for block in request.code_blocks
            ]
            request.verdict = combine_verdicts(request.block_verdicts)
            request.generation_attempts = turn + 1
            request.validation_reasons = sorted(
                {
                    reason
                    for verdict in request.block_verdicts
                    for reason in verdict.reasons
                }
            )
        else:
            # No code emitted (model refusal / prose-only) — nothing to
            # validate, nothing to loop on.
            break

        if _fully_validated(request.verdict):
            break
        if turn < max_turns:
            reasons = request.validation_reasons
            logger.debug(
                "Code validation failed on turn %d/%d — reformulating "
                "with %d reason(s): %s",
                turn + 1,
                max_turns + 1,
                len(reasons),
                "; ".join(reasons),
            )
        else:
            # Budget exhausted — refuse honestly instead of returning code.
            logger.warning(
                "Code remained unvalidated after %d attempt(s) — refusing "
                "to return code (reasons: %s)",
                max_turns + 1,
                "; ".join(request.validation_reasons),
            )
            request.refuse_code()
            break

    assert request is not None
    return request


def _fully_validated(verdict: ValidationVerdict | None) -> bool:
    """True when the verdict proves the code valid — every check PASS.

    A passed verdict with ``SKIP`` checks is *not* fully validated (the
    check never ran / no evidence to ground against) and must not be treated
    as validated code (§7.5: never return unvalidated code).
    """
    if verdict is None:
        return False
    return verdict.passed and all(
        check.status is CheckStatus.PASS for check in verdict.checks
    )