"""Code-route dispatch for Phase 6 (PLAN §7.2 T3, SPEC §8).

The code route is an **explicit opt-in** that never shadows the standard
answer path.  A query reaches code generation only when the dispatch says so:

1. a per-request ``explicit`` opt-in always selects the code route; or
2. the route is armed (``config.CODE_ROUTE_ENABLED`` / caller ``enabled``)
   **and** the query looks like a code request (deterministic intent-phrase
   match, zero LLM calls — mirroring the Phase 2 heuristic gate); otherwise
3. the request falls back to the standard ask path, byte-identical to a
   plain ``ask()`` (§7.5: non-code queries never produce a code answer).

:func:`run_code_route` is the T3 "gated route": it runs the T2
:func:`docpilot.codegen.pipeline_ask_code.ask_code` pipeline and then wires
the T1 validator in — every emitted code block is validated against the
retrieved evidence and the verdicts are attached to the
:class:`~docpilot.codegen.pipeline_ask_code.CodeRequest`.  There is **no
reformulation loop yet** — that is T4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docpilot import config
from docpilot.agent.gate import _normalise
from docpilot.validation.verdict import combine_verdicts

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
    top_k: int | None = None,
    language: str | None = None,
) -> CodeRouteResult:
    """Route *question* through the code path only when the dispatch allows.

    Args:
        question: The user's question / code request.
        enabled: Override for ``config.CODE_ROUTE_ENABLED`` (arm lever).
        explicit_code: Per-request opt-in — the caller asserts code intent.
        validator: A ``CodeValidator``; when given, every emitted code block
            is validated against the retrieved evidence (T1 wiring).  ``None``
            → verdicts are left empty (T2-only behaviour).
        retriever / generator / citation_engine: Injectable components
            (defaults build the production PG/Groq stack, lazily).
        top_k / language: Retrieval knobs, same semantics as ``ask()``.

    Returns:
        A :class:`CodeRouteResult`: ``code`` set on the code route (with
        per-block verdicts attached when *validator* was given), else ``ask``
        — the standard answer, byte-identical to a plain ``ask()`` call.
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

    from docpilot.codegen.pipeline_ask_code import ask_code

    request = ask_code(
        question,
        retriever=retriever,
        generator=generator,
        citation_engine=citation_engine,
        top_k=top_k,
        language=language,
    )
    if validator is not None:
        evidence = [r.chunk.content for r in request.results]
        request.block_verdicts = [
            validator.validate(block, evidence) for block in request.code_blocks
        ]
        request.verdict = (
            combine_verdicts(request.block_verdicts) if request.block_verdicts else None
        )
    return CodeRouteResult(decision=decision, code=request)