"""Hermetic tests for the LangGraph wiring (PLAN §3.6, SPEC §4.5).

All tests use injected fakes — no network, no DB, no live LLM. They exercise
the compiled ``StateGraph`` directly with an ``AgentLoopState`` whose trace
already carries the pipeline's gate step (mirroring what
:func:`docpilot.agent.pipeline_agentic.agentic_ask` feeds the graph).

Covered routing / budget / node-level guarantees:
    * sufficient first pass → answer, exactly 1 judge call;
    * insufficient → reformulate → second search → sufficient;
    * insufficient twice → refuse (≤ AGENT_MAX_RETRIES judge calls);
    * trace latency_ms + turn detail (retrieved count / top scores);
    * a node is invoked with the *injected* component instance (no vendor).

``FakeRetriever`` and ``StubJudge`` are shared with the pipeline tests.
"""

from __future__ import annotations

from docpilot import config
from docpilot.agent.graph import build_graph, trace_step_to_dict
from docpilot.agent.prompts import REFUSE_ANSWER
from docpilot.agent.types import AgentLoopState, Judgment, LoopTraceStep
from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.models import Chunk, RetrieverResult

from test_pipeline_e2e import FakeGenerator


GATE_QUERY = "Combine path, query, and body parameters in one endpoint — what are the validation rules for each kind?"


# ---------------------------------------------------------------------------
# Shared fakes (also imported by tests/test_agent_pipeline.py)
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Canned retriever: returns a preset result list per query string."""

    def __init__(self, results_by_query: dict[str, list[RetrieverResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, top_k: int = 5, *, language: str | None = None):
        self.calls.append((query, top_k, language))
        return list(self.results_by_query.get(query, []))


class StubJudge:
    """Preset-judgment judge that pops from a queue and counts calls."""

    def __init__(self, judgments: list[Judgment]) -> None:
        self.judgments = list(judgments)
        self.calls = 0
        self.last_query_used: str | None = None

    def judge(self, question: str, results: list[RetrieverResult], query_used: str) -> Judgment:
        self.calls += 1
        self.last_query_used = query_used
        if not self.judgments:
            return Judgment(verdict="sufficient", reason="no preset remaining")
        return self.judgments.pop(0)


def make_result(
    content: str = "FastAPI supports path, query and body parameters.",
    source_file: str = "en/docs/tutorial/params.md",
    heading: str = "Parameters",
    score: float = 0.81,
    chunk_id: str = "c1",
) -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(
            id=chunk_id,
            content=content,
            heading_path=heading,
            source_file=source_file,
            chunk_index=0,
        ),
        score=score,
    )


RESULTS_BY_QUERY: dict[str, list[RetrieverResult]] = {
    GATE_QUERY: [
        make_result(
            content="[1] Path parameters, [2] query parameters, [3] body parameters validation.",
            source_file="en/docs/tutorial/params.md",
            heading="Parameters",
            score=0.91,
            chunk_id="c1",
        ),
        make_result(
            content="Validation rules differ per parameter kind.",
            source_file="en/docs/tutorial/validation.md",
            heading="Validation",
            score=0.72,
            chunk_id="c2",
        ),
    ],
    "reformulated validation rules": [
        make_result(
            content="Each parameter kind has its own validation rules.",
            source_file="en/docs/tutorial/validation.md",
            heading="Validation",
            score=0.88,
            chunk_id="c3",
        ),
    ],
}


def results_for(query: str) -> list[RetrieverResult]:
    """Per-query canned results (raises KeyError for unseen queries)."""
    return RESULTS_BY_QUERY[query]


def initial_state(question: str = GATE_QUERY, gate_decision: str = "agentic") -> AgentLoopState:
    """Build the same initial loop state the pipeline feeds the graph."""
    gate_step = LoopTraceStep.new("gate", question, gate_decision, detail="signals=[]; simple")
    return {
        "question": question,
        "original_question": question,
        "current_query": question,
        "results": [],
        "sources": [],
        "attempts": 0,
        "trace": [trace_step_to_dict(gate_step)],
        "answer": None,
        "refused": False,
        "direct": False,
    }


def _trace_steps(final: dict) -> list[str]:
    return [t.get("step") for t in final["trace"]]


def _build_app(retriever, judge, generator, max_retries: int = 2):
    return build_graph(
        retriever=retriever,
        judge=judge,
        generator=generator,
        citation_engine=StandardCitationEngine(),
        top_k=5,
        language=None,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_sufficient_first_pass_answers_with_one_judge_call() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge([Judgment(verdict="sufficient", reason="covers it")])
    gen = FakeGenerator("FastAPI supports all parameter kinds. [1]")
    app = _build_app(retriever, judge, gen)

    final = app.invoke(initial_state())

    assert judge.calls == 1
    assert final["attempts"] == 1
    assert final["refused"] is False
    assert "[1]" in final["answer"]
    assert "Sources:" in final["answer"]
    # gate (pipeline) → search → judge → answer
    assert _trace_steps(final) == ["gate", "search", "judge", "answer"]


def test_insufficient_reformulates_then_sufficient() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge(
        [
            Judgment(
                verdict="insufficient",
                reason="need more",
                reformulated_query="reformulated validation rules",
            ),
            Judgment(verdict="sufficient", reason="ok now"),
        ]
    )
    gen = FakeGenerator("Now covered. [1]")
    app = _build_app(retriever, judge, gen)

    final = app.invoke(initial_state())

    assert judge.calls == 2
    assert final["attempts"] == 2
    assert _trace_steps(final) == ["gate", "search", "judge", "search", "judge", "answer"]
    # The reformulated query was fed to the second retrieve round.
    assert final["current_query"] == "reformulated validation rules"
    assert retriever.calls[1][0] == "reformulated validation rules"
    assert final["refused"] is False


def test_insufficient_twice_refuses_within_budget() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge(
        [
            Judgment(verdict="insufficient", reason="need more", reformulated_query="reformulated validation rules"),
            Judgment(verdict="insufficient", reason="still lacking", reformulated_query="another query"),
        ]
    )
    gen = FakeGenerator("should not be reached")
    max_retries = 2
    app = _build_app(retriever, judge, gen, max_retries=max_retries)

    final = app.invoke(initial_state())

    assert judge.calls == max_retries
    assert final["attempts"] == max_retries
    assert final["refused"] is True
    assert final["answer"] == REFUSE_ANSWER
    assert gen.calls == 0  # answer node never ran
    assert _trace_steps(final) == ["gate", "search", "judge", "search", "judge", "refuse"]


def test_budget_scale_with_three_retries() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge(
        [
            Judgment(verdict="insufficient", reason="a", reformulated_query="q1"),
            Judgment(verdict="insufficient", reason="b", reformulated_query="q2"),
            Judgment(verdict="insufficient", reason="c", reformulated_query="q3"),
        ]
    )
    gen = FakeGenerator("unused")
    max_retries = 3
    app = _build_app(retriever, judge, gen, max_retries=max_retries)

    final = app.invoke(initial_state())

    assert judge.calls == 3
    assert final["refused"] is True


def test_build_graph_default_top_k_is_agent_loop_top_k() -> None:
    # PLAN §3.7 fix: build_graph with no explicit top_k uses AGENT_LOOP_TOP_K
    # (broader than the fast path's RETRIEVAL_TOP_K).
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge([Judgment(verdict="sufficient", reason="covers it")])
    gen = FakeGenerator("covered [1]")
    app = build_graph(
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
        language=None,
        max_retries=2,
    )

    final = app.invoke(initial_state())

    assert final["refused"] is False
    assert retriever.calls and retriever.calls[0][1] == config.AGENT_LOOP_TOP_K


# ---------------------------------------------------------------------------
# Trace content
# ---------------------------------------------------------------------------


def test_trace_steps_carry_latency_and_turn_detail() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge([Judgment(verdict="sufficient", reason="covers it")])
    gen = FakeGenerator("covered [1]")
    app = _build_app(retriever, judge, gen)

    final = app.invoke(initial_state())

    search = next(t for t in final["trace"] if t["step"] == "search")
    j = next(t for t in final["trace"] if t["step"] == "judge")
    answer = next(t for t in final["trace"] if t["step"] == "answer")

    assert search["latency_ms"] is not None and search["latency_ms"] >= 0
    assert j["latency_ms"] is not None and j["latency_ms"] >= 0
    assert answer["latency_ms"] is not None

    # Detail carries the turn number, retrieved count and top scores.
    assert "turn=1" in search["detail"]
    assert "retrieved 2 chunks" in search["detail"]
    assert "0.91" in search["detail"] and "0.72" in search["detail"]

    # Judge detail carries verdict + reason (and reformulated_query when set).
    assert "verdict=sufficient" in j["detail"]
    assert "reason=covers it" in j["detail"]


def test_judge_detail_includes_reformulated_query() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge(
        [
            Judgment(verdict="insufficient", reason="nope", reformulated_query="reformulated validation rules"),
            Judgment(verdict="sufficient", reason="ok"),
        ]
    )
    gen = FakeGenerator("ok [1]")
    app = _build_app(retriever, judge, gen)

    final = app.invoke(initial_state())
    judge_steps = [t for t in final["trace"] if t["step"] == "judge"]
    assert "reformulated_query=reformulated validation rules" in judge_steps[0]["detail"]


# ---------------------------------------------------------------------------
# Node-level: injected instance is the one invoked (no vendor calls)
# ---------------------------------------------------------------------------


def test_injected_components_are_the_ones_invoked() -> None:
    retriever = FakeRetriever(RESULTS_BY_QUERY)
    judge = StubJudge(
        [
            Judgment(verdict="insufficient", reason="nope", reformulated_query="reformulated validation rules"),
            Judgment(verdict="sufficient", reason="ok"),
        ]
    )
    gen = FakeGenerator("answer [1]")
    app = _build_app(retriever, judge, gen)

    final = app.invoke(initial_state())

    # The fake retriever was used (it recorded its call args).
    assert retriever.calls, "FakeRetriever never invoked"
    assert [c[0] for c in retriever.calls] == [GATE_QUERY, "reformulated validation rules"]
    # The fake judge was used.
    assert judge.calls == 2
    assert judge.last_query_used == "reformulated validation rules"
    # The fake generator produced the answer (answer node called it via
    # generate_answer, not a raw vendor path).
    assert gen.calls == 1
    assert gen.last_context is not None and gen.last_context.startswith("[1] ")
    assert gen.last_question == GATE_QUERY
    assert final["answer"].startswith("answer")
