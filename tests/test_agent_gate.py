"""Tests for agent/gate.py — HeuristicQueryClassifier (zero LLM calls).

Gate correctness on the PLAN §3.8 seed set is a Phase 2 exit criterion
(PLAN §3.7): multi-hop seeds → agentic, simple seeds → direct.
"""

from __future__ import annotations

from docpilot.agent.gate import (
    HeuristicQueryClassifier,
    MIN_TECH_TERMS,
    SIMPLE_WORD_LIMIT,
)
from docpilot.agent.questions import SEED_QUESTIONS


class TestHeuristicGate:
    def setup_method(self) -> None:
        self.gate = HeuristicQueryClassifier()

    # ── robustness ──────────────────────────────────────────────────────

    def test_empty_input_does_not_throw(self) -> None:
        for question in ("", "   ", "\n\t", "  \n  "):
            decision = self.gate.classify(question)
            assert decision.agentic is False
            assert decision.signals == []
            assert decision.reason == "empty input"

    def test_punctuation_only_and_numeric_input_do_not_throw(self) -> None:
        for question in ("!!!", "????", "12345", "...", "− — –"):
            decision = self.gate.classify(question)
            assert decision.agentic is False
            assert decision.signals == []

    def test_classify_is_deterministic(self) -> None:
        question = (
            "Combine path, query, and body parameters in one endpoint — "
            "what are the validation rules for each kind?"
        )
        first = self.gate.classify(question)
        second = self.gate.classify(question)
        assert first == second

    # ── simple questions stay on the fast path ──────────────────────────

    def test_simple_short_question_is_direct(self) -> None:
        # PLAN §3.8 simple seed — must NOT enter the loop.
        decision = self.gate.classify("How do I install FastAPI?")
        assert decision.agentic is False
        assert decision.signals == []

    def test_query_parameter_definition_is_direct(self) -> None:
        # PLAN §3.8 simple seed — 2 tech terms but no corroborating signal
        # ("query parameter" is a single compound concept).
        decision = self.gate.classify("What is a query parameter in FastAPI?")
        assert decision.agentic is False
        assert decision.signals == []

    # ── word-count signal ───────────────────────────────────────────────

    def test_word_count_signal_fires(self) -> None:
        long_question = (
            "I want to know the most efficient way to deploy a small "
            "fastapi application on a single low cost server instance"
        )
        assert len(long_question.split()) > SIMPLE_WORD_LIMIT
        decision = self.gate.classify(long_question)
        assert decision.agentic is True
        assert any(s.startswith("long_question") for s in decision.signals)

    # ── connector signal ────────────────────────────────────────────────

    def test_connector_phrase_difference_between_fires(self) -> None:
        decision = self.gate.classify(
            "What is the difference between path and query parameters in FastAPI?"
        )
        assert decision.agentic is True
        assert any(s.startswith("connectors:") for s in decision.signals)

    def test_connector_word_combine_fires(self) -> None:
        decision = self.gate.classify(
            "Combine path, query, and body parameters in one endpoint — "
            "what are the validation rules for each kind?"
        )
        assert decision.agentic is True
        assert any(s.startswith("connectors:") for s in decision.signals)

    def test_connector_vs_fires(self) -> None:
        decision = self.gate.classify("Background tasks vs yield-dependencies — when does cleanup run?")
        assert decision.agentic is True
        assert any(s.startswith("connectors:") for s in decision.signals)

    # ── multiple-technical-terms signal ─────────────────────────────────

    def test_multiple_technical_terms_fire(self) -> None:
        decision = self.gate.classify("Path and Query — when to use each?")
        assert decision.agentic is True
        assert any(s.startswith("multi_tech_terms:") for s in decision.signals)

    def test_multiple_technical_terms_constant(self) -> None:
        assert MIN_TECH_TERMS == 2

    # ── explicit multi-part pattern ─────────────────────────────────────

    def test_multi_part_pattern_fires(self) -> None:
        decision = self.gate.classify(
            "How do I add a path parameter and what is a query parameter?"
        )
        assert decision.agentic is True
        assert "multi_part" in decision.signals

    # ── seed-set coverage (PLAN §3.7 exit criterion) ────────────────────

    def test_seed_set_gate_classification(self) -> None:
        """All multi-hop seeds → agentic; all simple seeds → direct.

        Refuse seeds must still end in "I don't know"; whether the gate
        routes them through the loop or the fast path is documented in
        AGENT F's report (auto-cache seed → loop; live-status seed → direct).
        """
        by_category: dict[str, list[str]] = {}
        for seed in SEED_QUESTIONS:
            by_category.setdefault(seed.category, []).append(seed.question)

        for question in by_category["multi_hop"]:
            decision = self.gate.classify(question)
            assert decision.agentic is True, f"multi_hop seed not agentic: {question!r}"

        for question in by_category["simple"]:
            decision = self.gate.classify(question)
            assert decision.agentic is False, f"simple seed not direct: {question!r}"

        # Refuse seeds classify without error (no throw); outcome per note.
        for question in by_category["refuse"]:
            decision = self.gate.classify(question)
            assert isinstance(decision.agentic, bool)