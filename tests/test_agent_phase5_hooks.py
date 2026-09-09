"""Hermetic tests for the Phase 5 graph hooks (SPEC §7):

    * judge-skip routing (AGENT_JUDGE_SKIP_MIN_SCORE) — answer without the
      judge LLM call when the top retrieval score clears the threshold;
    * the emit hook — every node reports its step live, the retrieve node
      reports the debug-panel search payload, and the answer node streams
      tokens when the generator supports streaming.

No network / DB / live LLM — all fakes.
"""

from __future__ import annotations

from docpilot.agent.graph import build_graph
from docpilot.agent.pipeline_agentic import agentic_ask
from docpilot.citations.engine import StandardCitationEngine

from test_agent_graph import (
    GATE_QUERY,
    FakeRetriever,
    StubJudge,
    initial_state,
    results_for,
    _trace_steps,
)


class StreamingFakeGenerator:
    """Answer generator with streaming (word deltas) — mirrors Pipeline fakes."""

    def __init__(self, text: str = "Streamed answer [1]") -> None:
        self.text = text

    def generate_answer(self, context, sources, question) -> str:
        return self.text

    def generate_answer_stream(self, context, sources, question):
        for word in self.text.split(" "):
            yield word + " "


class PlainFakeGenerator(StreamingFakeGenerator):
    def generate_answer_stream(self, context, sources, question):  # type: ignore[override]
        raise NotImplementedError("generate_answer_stream not supported")


class ExplodingJudge(StubJudge):
    """Judge that fails loudly — proves it was never invoked."""

    def judge(self, question, results, query_used, *, tools_available=True):  # type: ignore[override]
        raise AssertionError("judge must not run when the skip threshold is cleared")


def _build_app(
    retriever,
    judge,
    generator,
    *,
    judge_skip_score: float = 0.0,
    emit=None,
):
    return build_graph(
        retriever=retriever,
        judge=judge,
        generator=generator,
        citation_engine=StandardCitationEngine(),
        top_k=5,
        language=None,
        max_retries=2,
        judge_skip_score=judge_skip_score,
        emit=emit,
    )


class TestJudgeSkip:
    def test_threshold_cleared_skips_judge_and_answers(self) -> None:
        retriever = FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)})  # top score 0.91
        gen = StreamingFakeGenerator()
        app = _build_app(
            retriever,
            ExplodingJudge([]),
            gen,
            judge_skip_score=0.8,
        )
        final = app.invoke(initial_state())
        assert _trace_steps(final) == ["gate", "search", "answer"]
        answer_step = next(t for t in final["trace"] if t["step"] == "answer")
        assert "judge skipped" in answer_step["detail"]
        assert final["refused"] is False
        assert final["answer"].startswith("Streamed answer")

    def test_threshold_disabled_runs_judge(self) -> None:
        retriever = FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)})
        judge = StubJudge([])
        app = _build_app(retriever, judge, StreamingFakeGenerator())
        final = app.invoke(initial_state())
        assert _trace_steps(final) == ["gate", "search", "judge", "answer"]
        assert judge.calls == 1

    def test_score_below_threshold_still_judges(self) -> None:
        retriever = FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)})
        judge = StubJudge([])
        app = _build_app(retriever, judge, StreamingFakeGenerator(), judge_skip_score=0.95)
        final = app.invoke(initial_state())
        assert _trace_steps(final) == ["gate", "search", "judge", "answer"]
        assert judge.calls == 1


class TestEmitHook:
    def test_step_events_and_search_payload_emitted(self) -> None:
        events = []
        app = _build_app(
            FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)}),
            StubJudge([]),
            StreamingFakeGenerator(),
            emit=events.append,
        )
        final = app.invoke(initial_state())
        kinds = [e["type"] for e in events]
        assert kinds.count("step") == 3  # search, judge, answer
        assert kinds.count("search") == 1
        step_names = [e["step"]["step"] for e in events if e["type"] == "step"]
        assert step_names == ["search", "judge", "answer"]
        payload = next(e for e in events if e["type"] == "search")
        assert payload["turn"] == 1
        assert payload["results"][0]["score"] == 0.91
        assert payload["results"][0]["kind"] == "tutorial"
        assert _trace_steps(final) == ["gate", "search", "judge", "answer"]

    def test_tokens_streamed_from_answer_node(self) -> None:
        events = []
        app = _build_app(
            FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)}),
            StubJudge([]),
            StreamingFakeGenerator("Streamed answer [1]"),
            emit=events.append,
        )
        final = app.invoke(initial_state())
        tokens = [e for e in events if e["type"] == "token"]
        assert tokens, "expected streamed token events"
        streamed = "".join(e["delta"] for e in tokens)
        assert "Streamed answer" in streamed
        assert final["answer"].startswith("Streamed answer")

    def test_no_tokens_when_generator_does_not_stream(self) -> None:
        events = []
        app = _build_app(
            FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)}),
            StubJudge([]),
            PlainFakeGenerator(),
            emit=events.append,
        )
        final = app.invoke(initial_state())
        assert not any(e["type"] == "token" for e in events)
        assert final["answer"].startswith("Streamed answer")


class TestPipelineEmit:
    def test_agentic_ask_emits_gate_step_first_on_direct_path(self) -> None:
        events: list[dict] = []
        retriever = FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)})
        result = agentic_ask(
            GATE_QUERY,
            strategy="direct",
            retriever=retriever,
            generator=StreamingFakeGenerator(),
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        assert events and events[0]["type"] == "step"
        assert events[0]["step"]["step"] == "gate"
        assert events[0]["step"]["decision"] == "forced-direct"
        assert result.direct is True

    def test_agentic_ask_emits_search_payload_with_kind(self) -> None:
        events: list[dict] = []
        agentic_ask(
            GATE_QUERY,
            strategy="agentic",
            retriever=FakeRetriever({GATE_QUERY: results_for(GATE_QUERY)}),
            generator=StreamingFakeGenerator(),
            judge=StubJudge([]),
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        payload = next(e for e in events if e["type"] == "search")
        assert payload["results"][0]["kind"] == "tutorial"
        assert payload["results"][0]["score"] == 0.91