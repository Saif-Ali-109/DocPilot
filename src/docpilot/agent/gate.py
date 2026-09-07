"""Heuristic query classifier — zero LLM calls (SPEC §4.1, PLAN §3.2).

:class:`HeuristicQueryClassifier` decides whether a question should take the
fast/direct path (identical to Phase 1 ``ask()``) or be routed through the
agentic loop.  The classification is **pure and deterministic**: no I/O, no
LLM, no randomness.

Signal definitions (PLAN §3 locked decisions, tuned against the §3.8 seed
set — PLAN wins over the task's draft proposal where they conflict):

    1. **long_question** — more than :data:`SIMPLE_WORD_LIMIT` (18) words.
    2. **connectors** — the normalised question contains reasoning /
       comparison / joiner language from :data:`CONNECTORS` (e.g. ``and``,
       ``or``, ``when``, ``because``, ``without``, ``combine``,
       ``"difference between"``, ``vs``, ``versus``).  Canonical question
       openers (``"how do i"``, ``"what is"``, ``"why is"``, …) are stripped
       *before* connector matching so that simple questions such as the
       §3.8 ``simple`` seeds ("How do I install FastAPI?") stay on the fast
       path — a bare ``how``/``what`` opener is not a reasoning connector.
    3. **multi_part** — two or more ``"how do i"`` / ``"what is"`` /
       ``"how can i"`` conjuncts joined by ``and`` (e.g. *"How do I add a
       path parameter and what is a query parameter?"*).
    4. **multi_concept** — STANDALONE trigger: at least
       :data:`MIN_AGENTIC_CONCEPTS` (3) **distinct** framework concepts from
       :data:`TECH_TERMS` are present (case-insensitive, word-boundary
       matches).  Juxtaposing three or more concepts is a strong multi-hop
       signal even without connector language — this routes the §3.8 seed
       questions 3/7/8 ("…exception handlers/middleware?",
       "WebSocket endpoint + HTTP route … auth dependency",
       "OAuth2 security scopes + custom dependency … routes") agentically.
    5. **multi_tech_terms** — corroborating ONLY: exactly two distinct
       concepts (:data:`MIN_TECH_TERMS`) appear **alongside** another
       complexity signal.  Two terms alone usually form a single compound
       concept ("query parameter"), so they never stand alone — this keeps
       the §3.8 simple seed "What is a query parameter in FastAPI?" direct.
       (Concept counts at or above :data:`MIN_AGENTIC_CONCEPTS` fire signal 4
       instead, so this corroborating signal never double-fires.)

Aggregation: the question is **agentic when at least one signal fires**;
empty or whitespace-only input returns ``agentic=False`` (never throws).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from docpilot.agent.types import GateDecision

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SIMPLE_WORD_LIMIT: int = 18
"""Word-count threshold — questions longer than this fire the word-count signal."""

MIN_TECH_TERMS: int = 2
"""Minimum distinct tech-term matches (alongside another signal) to fire the
corroborating multi-tech-term signal."""

MIN_AGENTIC_CONCEPTS: int = 3
"""Minimum DISTINCT matched concepts to fire the standalone multi_concept
signal.  Three or more juxtaposed framework concepts is treated as a strong
multi-hop indicator even without connector/reasoning language."""

# Reasoning / comparison / joiner words that suggest the question compares,
# combines or conditions over multiple things.  Question openers ("how do i",
# "what is", …) are stripped before matching, so bare "how"/"why" here are
# mid-question reasoning words, not simple-question openers.
CONNECTORS: frozenset[str] = frozenset({
    # joiners / comparators
    "and", "or", "both",
    "combine", "difference", "differences",
    "compare", "compared", "versus", "vs",
    # reasoning / condition
    "because", "if", "then", "while", "when",
    "whereas", "although", "unless", "without",
    # mid-question reasoning (openers are stripped before matching)
    "how", "why",
})

# Phrase-level connector patterns (checked against the normalised question).
_CONNECTOR_PHRASES: frozenset[str] = frozenset({
    "difference between", "compared to", "compared with",
    "as well as", "in order to",
})

# Canonical question openers stripped before connector/phrase matching — a
# simple how/what question is not a reasoning/comparison signal by itself.
_QUESTION_OPENERS: tuple[str, ...] = (
    "how do i", "how can i", "how do you", "how does",
    "what is", "what are", "what does",
    "why is", "why do", "why does",
    "when is", "when do", "when does",
    "can i", "do i",
)

# Technical terms drawn from the FastAPI corpus (PLAN §3.2 locked list
# expanded with terms common in the actual docs).
TECH_TERMS: frozenset[str] = frozenset({
    "path", "query", "body", "parameter", "parameters",
    "dependency", "dependencies", "depends",
    "middleware", "security", "websocket", "websockets",
    "background", "sub-application", "mount", "mounts",
    "response", "response_model",
    "oauth", "oauth2", "jwt", "cors", "testing", "deployment",
    "uvicorn", "validation", "pydantic",
    "header", "cookie", "form", "file",
    "exception", "exception_handler",
    "annotated", "scope", "scopes",
    "openapi", "schema", "model",
    # concepts added for the multi_concept standalone signal (PLAN §3.7 fix)
    "endpoint", "route", "routes", "http", "auth", "authentication",
    "handler", "task", "cleanup", "request",
})

# Regex for explicit multi-part questions:
#   "how do i / what is / how can i" … "and" … (second conjunct)
_MULTI_PART_RE = re.compile(
    r"(?:how\s+do\s+i|what\s+is|how\s+can\s+i)\b.*?\band\b.*?"
    r"(?:how\s+do\s+i|what\s+is|how\s+can\s+i)\b",
    re.IGNORECASE,
)

_OPENER_RE = re.compile(
    r"^(?:" + "|".join(_QUESTION_OPENERS) + r")\b",
    re.IGNORECASE,
)

# Word-boundary compound alternation over TECH_TERMS — used to count DISTINCT
# matched concepts.  ``findall`` returns the matched term strings (no
# capturing group, so no group extraction is needed).
_TECH_TERMS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in TECH_TERMS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class QueryClassifier(ABC):
    """Interface for query classification (direct vs. agentic)."""

    @abstractmethod
    def classify(self, question: str) -> GateDecision:
        """Classify *question* and return a :class:`GateDecision`."""
        ...


# ---------------------------------------------------------------------------
# Heuristic implementation
# ---------------------------------------------------------------------------


class HeuristicQueryClassifier(QueryClassifier):
    """Pure, deterministic, zero-LLM query classifier.

    See the module docstring for the aggregation logic and signal
    definitions.
    """

    def classify(self, question: str) -> GateDecision:
        """Classify *question* using heuristic signals.

        Returns:
            A :class:`GateDecision` with ``agentic=True`` when at least one
            signal fires, ``agentic=False`` otherwise.
        """
        # Guard: empty / whitespace-only
        if not question or not question.strip():
            return GateDecision(agentic=False, signals=[], reason="empty input")

        normalised = _normalise(question)
        stripped = _strip_opener(normalised)
        signals: list[str] = []

        # 1. Word count
        words = normalised.split()
        if len(words) > SIMPLE_WORD_LIMIT:
            signals.append("long_question")

        # 2. Connector words / phrases (question openers already removed)
        connector_hits = _detect_connectors(stripped)
        if connector_hits:
            signals.append(f"connectors:{','.join(sorted(connector_hits))}")

        # 3. Explicit multi-part pattern
        multi_part = _MULTI_PART_RE.search(normalised) is not None
        if multi_part:
            signals.append("multi_part")

        # 4. Distinct framework concepts — counted once per concept via
        #    word-boundary matches against the normalised (opener-stripped)
        #    text.
        tech_matches = sorted(set(_TECH_TERMS_RE.findall(stripped)))
        num_concepts = len(tech_matches)

        # STANDALONE: three or more distinct concepts (multi-hop even without
        # connector language).
        if num_concepts >= MIN_AGENTIC_CONCEPTS:
            signals.append(f"multi_concept:{','.join(tech_matches[:5])}")
        # CORROBORATING: exactly two concepts — only contributes when another
        # complexity signal has already fired ("query parameter" is a single
        # compound concept, not a standalone trigger).
        elif num_concepts >= MIN_TECH_TERMS and (signals or multi_part):
            signals.append(f"multi_tech_terms:{','.join(tech_matches[:5])}")

        if signals:
            return GateDecision(
                agentic=True,
                signals=signals,
                reason=f"signals fired: {', '.join(signals)}",
            )

        return GateDecision(
            agentic=False,
            signals=[],
            reason="simple question — no heuristics triggered",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation (except spaces), collapse whitespace."""
    text = text.lower()
    # Remove punctuation characters that aren't spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_opener(normalised: str) -> str:
    """Remove a leading canonical question opener from *normalised*.

    "How do I install FastAPI?" → "install fastapi".  This is applied before
    connector/tech-term matching so that simple how/what questions do not
    fire the reasoning-connector signal through their opener.
    """
    return _OPENER_RE.sub("", normalised, count=1).strip()


def _detect_connectors(normalised: str) -> set[str]:
    """Return the set of connector words/phrases present in *normalised*."""
    hits: set[str] = set()
    for phrase in _CONNECTOR_PHRASES:
        if phrase in normalised:
            hits.add(phrase)
    for token in normalised.split():
        if token in CONNECTORS:
            hits.add(token)
    return hits