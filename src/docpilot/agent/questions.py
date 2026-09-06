"""Phase 2 seed question set (PLAN §3.8).

These questions are used for manual QA and gate classification validation
during Phase 2.  They are **not** the formal Phase 4 benchmark dataset.

- 8 multi-hop  (need multiple docs / comparison)
- 2 simple     (single-hop factual)
- 2 refuse     (out-of-corpus)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeedQuestion:
    """A single seed question for Phase 2 development and QA.

    Attributes:
        question: The natural-language question.
        category: ``"multi_hop"`` | ``"simple"`` | ``"refuse"``.
        note: What the question stresses or the expected answer shape.
    """

    question: str
    category: str  # "multi_hop" | "simple" | "refuse"
    note: str


SEED_QUESTIONS: list[SeedQuestion] = [
    # ── multi_hop (8) ─────────────────────────────────────────────────
    SeedQuestion(
        question=(
            "Combine path, query, and body parameters in one endpoint — "
            "what are the validation rules for each kind?"
        ),
        category="multi_hop",
        note=(
            "Requires retrieval from multiple parameter-related docs "
            "(path, query, body) and synthesising validation differences."
        ),
    ),
    SeedQuestion(
        question=(
            "Do dependencies interact with path operations — can one "
            "dependency validate params and help produce the response, "
            "with shared state passing?"
        ),
        category="multi_hop",
        note=(
            "Needs dependency docs + path operation docs; answer covers "
            "injection, shared state, and execution order."
        ),
    ),
    SeedQuestion(
        question=(
            "What happens when an exception is raised inside a dependency — "
            "how does it interact with exception handlers and middleware?"
        ),
        category="multi_hop",
        note=(
            "Crosses dependency docs, exception handler docs, and middleware "
            "docs; the answer must combine information from all three."
        ),
    ),
    SeedQuestion(
        question=(
            "Can the same Pydantic model be used for request-body validation "
            "and response_model — what are the differences?"
        ),
        category="multi_hop",
        note=(
            "Needs body-validation docs and response_model docs; answer "
            "explains field inclusion/exclusion differences."
        ),
    ),
    SeedQuestion(
        question=(
            "Background tasks vs yield-dependencies — when does cleanup "
            "really run?"
        ),
        category="multi_hop",
        note=(
            "Requires background task docs and dependency yield/lifecycle "
            "docs; answer compares execution timing."
        ),
    ),
    SeedQuestion(
        question=(
            "Annotated[...] = Depends(...) vs legacy = Depends(...) — "
            "what is the difference and are they mixable?"
        ),
        category="multi_hop",
        note=(
            "Needs Annotated-style docs and legacy Depends docs; answer "
            "covers equivalence and mixing rules."
        ),
    ),
    SeedQuestion(
        question=(
            "Can a WebSocket endpoint and an HTTP route share one auth "
            "dependency — how is the wiring done?"
        ),
        category="multi_hop",
        note=(
            "Crosses WebSocket docs and dependency/security docs; answer "
            "shows shared-dependency wiring."
        ),
    ),
    SeedQuestion(
        question=(
            "How do OAuth2 security scopes and a custom dependency work "
            "together to restrict routes?"
        ),
        category="multi_hop",
        note=(
            "Needs OAuth2 scope docs and custom dependency docs; answer "
            "combines both for a complete picture."
        ),
    ),

    # ── simple (2) ────────────────────────────────────────────────────
    SeedQuestion(
        question="How do I install FastAPI?",
        category="simple",
        note="Single-hop factual: installation docs only.",
    ),
    SeedQuestion(
        question="What is a query parameter in FastAPI?",
        category="simple",
        note="Single-hop factual: query-parameter definition.",
    ),

    # ── refuse (2) ────────────────────────────────────────────────────
    SeedQuestion(
        question=(
            "How do I auto-cache database queries in FastAPI without "
            "writing any extra code?"
        ),
        category="refuse",
        note=(
            "Out-of-corpus: FastAPI docs do not cover automatic DB query "
            "caching. Must refuse with the SPEC §3.9 sentence."
        ),
    ),
    SeedQuestion(
        question=(
            "What is the current status of the OAuth token-expiration "
            "bug in the FastAPI GitHub repository?"
        ),
        category="refuse",
        note=(
            "Live repo state / issue tracker — not covered by static docs. "
            "Phase 3 territory; must refuse in Phase 2."
        ),
    ),
]
