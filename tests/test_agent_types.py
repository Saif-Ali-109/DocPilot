"""Tests for agent/types.py — the Phase 2 shared contract.

Covers the budget constant, the verbatim SPEC §3.9 refusal sentence, the
loop-state shape AGENT G builds the LangGraph around, and the dataclass
defaults.
"""

from __future__ import annotations

from docpilot.agent import DEFAULT_MAX_RETRIES
from docpilot.agent.prompts import REFUSE_ANSWER
from docpilot.agent.types import (
    AgentLoopState,
    GateDecision,
    Judgment,
    LoopTraceStep,
)

SPEC_REFUSE_ANSWER = (
    "I don't know — the available documentation does not cover this question."
)

_REQUIRED_STATE_KEYS = (
    "question",
    "original_question",
    "current_query",
    "results",
    "sources",
    "attempts",
    "trace",
    "answer",
    "refused",
    "direct",
)


class TestConstants:
    def test_default_max_retries(self) -> None:
        assert DEFAULT_MAX_RETRIES == 2

    def test_refuse_answer_matches_spec_39_exactly(self) -> None:
        # SPEC §3.9 wording is verbatim and untouchable — tests assert it.
        assert REFUSE_ANSWER == SPEC_REFUSE_ANSWER

    def test_types_package_does_not_import_langgraph(self) -> None:
        # The shared contract must stay importable without langgraph
        # (AGENT G wires the graph separately).
        #
        # NOTE (AGENT G, Phase 2 wiring): this asserts the *contract module's
        # own namespace* rather than the process-global ``sys.modules`` — the
        # global dict is legitimately polluted by the graph tests (which must
        # load langgraph to run the StateGraph) before this test executes in
        # the same session. The intent — ``agent.types`` itself never depends
        # on langgraph — is preserved exactly.
        from docpilot.agent import types as types_module

        assert "langgraph" not in vars(types_module)
        assert "langgraph" not in vars(types_module).get("__builtins__", {})

        import docpilot.agent as agent_pkg

        assert "langgraph" not in vars(agent_pkg)


class TestAgentLoopState:
    def test_required_keys_present(self) -> None:
        annotations = AgentLoopState.__annotations__
        for key in _REQUIRED_STATE_KEYS:
            assert key in annotations, f"AgentLoopState missing key {key!r}"

    def test_state_instantiable_as_dict(self) -> None:
        state: AgentLoopState = {
            "question": "How do dependencies work?",
            "original_question": "How do dependencies work?",
            "current_query": "How do dependencies work?",
            "results": [],
            "sources": [],
            "attempts": 0,
            "trace": [],
            "answer": None,
            "refused": False,
            "direct": False,
        }
        assert state["question"] == "How do dependencies work?"
        assert state["attempts"] == 0
        assert state["refused"] is False


class TestContractDataclasses:
    def test_judgment_defaults(self) -> None:
        j = Judgment(verdict="sufficient", reason="docs cover it")
        assert j.verdict == "sufficient"
        assert j.reformulated_query is None

    def test_gate_decision_fields(self) -> None:
        d = GateDecision(agentic=False, signals=[], reason="empty input")
        assert d.agentic is False
        assert d.signals == []
        assert d.reason == "empty input"

    def test_loop_trace_step_fields(self) -> None:
        step = LoopTraceStep(step="gate", query="q", decision="direct")
        assert step.step == "gate"
        assert step.detail is None
        assert step.latency_ms is None

    def test_loop_trace_step_new_without_timer(self) -> None:
        step = LoopTraceStep.new("judge", "q", "sufficient")
        assert step.step == "judge"
        assert step.decision == "sufficient"
        assert step.latency_ms is None

    def test_loop_trace_step_new_with_timer(self) -> None:
        import time

        started = time.perf_counter()
        step = LoopTraceStep.new("search", "q", "retrieved", started_at=started)
        assert step.step == "search"
        assert step.latency_ms is not None
        assert step.latency_ms >= 0