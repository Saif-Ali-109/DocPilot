"""Tests for agent/judge.py — LLMSufficiencyJudge (stubbed Generator only).

Hermetic: never a real LLM, never a network call.  The ``StubGenerator``
returns whatever string the test sets and counts ``generate()`` calls so we
can assert the one-call-per-judge invariant (SPEC §4.1).
"""

from __future__ import annotations

from docpilot.agent.judge import LLMSufficiencyJudge
from docpilot.agent.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from docpilot.agent.types import Judgment
from docpilot.core.models import Chunk, RetrieverResult
from docpilot.generation.generator import Generator


class StubGenerator(Generator):
    """Minimal canned-response Generator that records calls and prompts."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response


def _result(
    content: str = "FastAPI supports path, query and body parameters.",
    score: float = 0.81,
    source_file: str = "en/docs/tutorial/params.md",
    heading: str = "Parameters",
) -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(
            id="c1",
            content=content,
            heading_path=heading,
            source_file=source_file,
            chunk_index=0,
        ),
        score=score,
    )


def _judge(
    stub: StubGenerator,
    question: str = "How do parameters work?",
    results: list[RetrieverResult] | None = None,
    query_used: str = "How do parameters work?",
) -> Judgment:
    """Run one judge() invocation against the stub and return the verdict."""
    return LLMSufficiencyJudge(stub).judge(
        question, results or [_result()], query_used
    )


class TestLLMSufficiencyJudge:
    def test_sufficient_verdict_parsed(self) -> None:
        stub = StubGenerator(
            '{"verdict": "sufficient", "reason": "the chunks cover parameters",'
            ' "reformulated_query": null}'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "sufficient"
        assert judgment.reason == "the chunks cover parameters"
        assert judgment.reformulated_query is None

    def test_insufficient_with_reformulated_query(self) -> None:
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "dependency lifecycle missing",'
            ' "reformulated_query": "When does dependency cleanup run in FastAPI?"}'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "insufficient"
        assert judgment.reformulated_query == (
            "When does dependency cleanup run in FastAPI?"
        )
        assert "dependency" in judgment.reason

    def test_json_wrapped_in_fences_parsed(self) -> None:
        stub = StubGenerator(
            '```json\n{"verdict": "sufficient", "reason": "ok",'
            ' "reformulated_query": null}\n```'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "sufficient"

    def test_trailing_prose_tolerated(self) -> None:
        stub = StubGenerator(
            'Here is my judgement: {"verdict": "insufficient",'
            ' "reason": "need more", "reformulated_query": "how to combine'
            ' dependencies?"} Hope that helps.'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "insufficient"
        assert judgment.reformulated_query == "how to combine dependencies?"

    def test_unparseable_output_defensive_sufficient(self) -> None:
        stub = StubGenerator("I cannot evaluate this document set right now.")
        judgment = _judge(stub)
        assert judgment.verdict == "sufficient"
        assert "unparseable" in judgment.reason

    def test_empty_output_defensive_sufficient(self) -> None:
        stub = StubGenerator("")
        judgment = _judge(stub)
        assert judgment.verdict == "sufficient"

    def test_unknown_verdict_defaults_to_sufficient(self) -> None:
        stub = StubGenerator(
            '{"verdict": "ambiguous", "reason": "uncertain",'
            ' "reformulated_query": "rephrase"}'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "sufficient"

    def test_exactly_one_generate_call_per_judge(self) -> None:
        """The judge must not loop / retry internally (SPEC §4.1)."""
        stub = StubGenerator('{"verdict": "sufficient", "reason": "ok",'
                             ' "reformulated_query": null}')
        _judge(stub)
        assert stub.calls == 1

    def test_prompt_contains_system_instructions_and_chunks(self) -> None:
        result = _result(content="Dependency cleanup runs after the response.")
        stub = StubGenerator('{"verdict": "sufficient", "reason": "ok",'
                             ' "reformulated_query": null}')
        _judge(stub, results=[result], query_used="dependency cleanup")
        assert stub.last_prompt is not None
        assert JUDGE_SYSTEM_PROMPT in stub.last_prompt
        assert "Dependency cleanup runs after the response." in stub.last_prompt
        assert "en/docs/tutorial/params.md" in stub.last_prompt
        assert "0.8100" in stub.last_prompt
        assert "QUERY USED FOR RETRIEVAL" in stub.last_prompt

    def test_judge_rejects_networks_by_construction(self) -> None:
        # Constructing with a stub means no network can be touched.
        stub = StubGenerator('{"verdict": "sufficient", "reason": "ok",'
                             ' "reformulated_query": null}')
        judge = LLMSufficiencyJudge(stub)
        assert judge._generator is stub


class TestBuildJudgeUserPrompt:
    def test_chunk_truncation_leaves_head_intact(self) -> None:
        long_content = "A" * 5000
        results = [_result(content=long_content)]
        prompt = build_judge_user_prompt("Q?", "Q?", results)
        # Content is truncated to a sane limit (module constant 1200).
        assert "A" * 1200 in prompt
        assert "A" * 1201 not in prompt

    def test_empty_results_gives_clean_prompt(self) -> None:
        prompt = build_judge_user_prompt("Q?", "Q?", [])
        assert "RETRIEVED CHUNKS:" in prompt
        assert "QUESTION:" in prompt