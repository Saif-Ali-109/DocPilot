"""Tests for agent/judge.py — LLMSufficiencyJudge (stubbed Generator only).

Hermetic: never a real LLM, never a network call.  The ``StubGenerator``
returns whatever string the test sets and counts ``generate()`` calls so we
can assert the one-call-per-judge invariant (SPEC §4.1).
"""

from __future__ import annotations

import logging

from docpilot.agent.judge import (
    LLMSufficiencyJudge,
    ScoreFloorBackstopJudge,
    judge_parse_fallback_counts,
    reset_judge_parse_fallback_counts,
)
from docpilot.agent.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_B,
    build_judge_user_prompt,
)
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

    def test_judge_prompt_softening_phrases_present(self) -> None:
        """PLAN §3.7 fix: the judge must not reject partial coverage — the
        softening wording must be in the system prompt verbatim."""
        assert (
            "even if a specific detail is only partially covered" in JUDGE_SYSTEM_PROMPT
        )
        assert (
            "Do not mark insufficient merely because a specific sentence is absent"
            in JUDGE_SYSTEM_PROMPT
        )
        assert "no usable evidence to begin answering" in JUDGE_SYSTEM_PROMPT
        assert "tutorial/xxx page" in JUDGE_SYSTEM_PROMPT

    def test_judge_rejects_networks_by_construction(self) -> None:
        # Constructing with a stub means no network can be touched.
        stub = StubGenerator('{"verdict": "sufficient", "reason": "ok",'
                             ' "reformulated_query": null}')
        judge = LLMSufficiencyJudge(stub)
        assert judge._generator is stub

    # -- Phase 3: needs_tool / tool_request parsing (SPEC §5) ---------------

    def test_needs_tool_and_tool_request_parsed(self) -> None:
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "live repo state needed",'
            ' "reformulated_query": null, "needs_tool": true,'
            ' "tool_request": {"name": "github.search_issues",'
            ' "params": {"query": "oauth token expiration"}}}'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "insufficient"
        assert judgment.needs_tool is True
        assert judgment.tool_request == {
            "name": "github.search_issues",
            "params": {"query": "oauth token expiration"},
        }

    def test_needs_tool_truthy_string_parsed(self) -> None:
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "x", "reformulated_query": null,'
            ' "needs_tool": "true",'
            ' "tool_request": {"name": "github.get_commits", "params": {}}}'
        )
        judgment = _judge(stub)
        assert judgment.needs_tool is True
        assert judgment.tool_request == {"name": "github.get_commits", "params": {}}

    def test_fenced_json_with_nested_tool_request_parsed(self) -> None:
        stub = StubGenerator(
            '```json\n{"verdict": "insufficient", "reason": "live state",'
            ' "reformulated_query": null, "needs_tool": true,'
            ' "tool_request": {"name": "github.list_issues",'
            ' "params": {"state": "open"}}}\n```'
        )
        judgment = _judge(stub)
        assert judgment.verdict == "insufficient"
        assert judgment.needs_tool is True
        assert judgment.tool_request == {
            "name": "github.list_issues",
            "params": {"state": "open"},
        }

    def test_malformed_needs_tool_and_tool_request_safe_defaults(self) -> None:
        # needs_tool="maybe" → False; tool_request missing its name → None.
        # The (still-valid) reformulated_query survives for the docs retry.
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "x",'
            ' "reformulated_query": "how to combine dependencies?",'
            ' "needs_tool": "maybe",'
            ' "tool_request": {"params": {"query": "x"}}}'
        )
        judgment = _judge(stub)
        assert judgment.needs_tool is False
        assert judgment.tool_request is None
        assert judgment.reformulated_query == "how to combine dependencies?"

    def test_missing_tool_keys_default_safe(self) -> None:
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "x",'
            ' "reformulated_query": "q"}'
        )
        judgment = _judge(stub)
        assert judgment.needs_tool is False
        assert judgment.tool_request is None

    def test_tool_request_requires_dict_params(self) -> None:
        # needs_tool true but malformed tool_request → request dropped (the
        # graph needs BOTH needs_tool and a well-formed request to fire).
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "x", "needs_tool": true,'
            ' "tool_request": {"name": "github.list_issues",'
            ' "params": "not-a-dict"}}'
        )
        judgment = _judge(stub)
        assert judgment.needs_tool is True
        assert judgment.tool_request is None

    def test_needs_tool_drops_reformulated_query(self) -> None:
        # Prompt contract: when needs_tool=true, reformulated_query must be
        # null (the tool replies with live evidence; it never loops to judge).
        stub = StubGenerator(
            '{"verdict": "insufficient", "reason": "x",'
            ' "reformulated_query": "stale retry", "needs_tool": true,'
            ' "tool_request": {"name": "github.search_issues",'
            ' "params": {"query": "x"}}}'
        )
        judgment = _judge(stub)
        assert judgment.needs_tool is True
        assert judgment.reformulated_query is None
        assert judgment.tool_request == {
            "name": "github.search_issues",
            "params": {"query": "x"},
        }

    def test_judge_prompt_contract_includes_tool_keys(self) -> None:
        assert '"needs_tool"' in JUDGE_SYSTEM_PROMPT
        assert '"tool_request"' in JUDGE_SYSTEM_PROMPT
        assert "github.search_issues" in JUDGE_SYSTEM_PROMPT
        assert "github.list_issues" in JUDGE_SYSTEM_PROMPT
        assert "github.get_commits" in JUDGE_SYSTEM_PROMPT
        # The tool is the last resort — docs retries are preferred.
        assert "prefer setting" in JUDGE_SYSTEM_PROMPT

    def test_judge_passes_tools_available_into_the_prompt(self) -> None:
        stub = StubGenerator(
            '{"verdict": "sufficient", "reason": "ok",'
            ' "reformulated_query": null}'
        )
        judge = LLMSufficiencyJudge(stub)
        judge.judge("Q?", [_result()], "Q?", tools_available=False)
        assert "TOOLS AVAILABLE: no" in stub.last_prompt
        assert "needs_tool" in stub.last_prompt and "MUST be false" in stub.last_prompt

        judge.judge("Q?", [_result()], "Q?", tools_available=True)
        assert "TOOLS AVAILABLE: yes" in stub.last_prompt

    # -- Phase 4: prompt B variant + system-prompt parameterization ---------

    def test_judge_accepts_custom_system_prompt(self) -> None:
        stub = StubGenerator(
            '{"verdict": "sufficient", "reason": "ok",'
            ' "reformulated_query": null}'
        )
        judge = LLMSufficiencyJudge(stub, system_prompt=JUDGE_SYSTEM_PROMPT_B)
        judge.judge("Q?", [_result()], "Q?")
        assert stub.last_prompt is not None
        assert "variant B" in stub.last_prompt

    def test_judge_defaults_to_production_prompt_a(self) -> None:
        stub = StubGenerator('{"verdict": "sufficient", "reason": "ok",'
                             ' "reformulated_query": null}')
        judge = LLMSufficiencyJudge(stub)
        judge.judge("Q?", [_result()], "Q?")
        assert "variant B" not in (stub.last_prompt or "")
        assert JUDGE_SYSTEM_PROMPT in (stub.last_prompt or "")

    def test_judge_prompt_b_is_distinct_but_contract_compatible(self) -> None:
        assert JUDGE_SYSTEM_PROMPT_B != JUDGE_SYSTEM_PROMPT
        # Same JSON contract keys.
        for key in ('"verdict"', '"reason"', '"reformulated_query"',
                    '"needs_tool"', '"tool_request"'):
            assert key in JUDGE_SYSTEM_PROMPT_B
        # Same live-validated tool rules (Phase 3 markers).
        for marker in (
            "github.search_issues",
            "github.list_issues",
            "github.get_commits",
            "is:issue",
            "prefer setting",
            "needs_tool",
        ):
            assert marker in JUDGE_SYSTEM_PROMPT_B
        # get_commits default-branch guidance carried over.
        assert "omit `ref`" in JUDGE_SYSTEM_PROMPT_B
        # JSON-only output directive (parse-failure contract parity).
        assert "No markdown fences" in JUDGE_SYSTEM_PROMPT_B


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

    def test_tools_available_true_emits_yes_line(self) -> None:
        prompt = build_judge_user_prompt("Q?", "Q?", [_result()], tools_available=True)
        assert "TOOLS AVAILABLE: yes" in prompt

    def test_tools_available_false_emits_no_and_instructs(self) -> None:
        prompt = build_judge_user_prompt("Q?", "Q?", [_result()], tools_available=False)
        assert "TOOLS AVAILABLE: no" in prompt
        assert '"needs_tool" MUST be false' in prompt
        assert '"tool_request" MUST be null' in prompt

    def test_tools_available_defaults_to_true(self) -> None:
        # Old 3-arg callers keep working — default keeps needs_tool enabled.
        prompt = build_judge_user_prompt("Q?", "Q?", [_result()])
        assert "TOOLS AVAILABLE: yes" in prompt


class TestJudgeParseFallbackCounter:
    """The parse-fallback counter records each defensive ``sufficient`` —
    one count per cause, plus a stable INFO log line carrying the running
    total (Phase 4 flip-condition signal, SPEC §6.3)."""

    def _run(self, raw: str) -> None:
        _judge(StubGenerator(raw))

    def test_empty_output_records_empty_cause(self) -> None:
        reset_judge_parse_fallback_counts()
        self._run("")
        assert judge_parse_fallback_counts() == {"empty": 1}

    def test_unparseable_output_records_unparseable_cause(self) -> None:
        reset_judge_parse_fallback_counts()
        self._run("I cannot evaluate this document set right now.")
        assert judge_parse_fallback_counts() == {"unparseable": 1}

    def test_unknown_verdict_records_bad_verdict_cause(self) -> None:
        reset_judge_parse_fallback_counts()
        self._run('{"verdict": "ambiguous", "reason": "uncertain",'
                  ' "reformulated_query": null}')
        assert judge_parse_fallback_counts() == {"bad_verdict": 1}

    def test_counts_accumulate_and_reset(self) -> None:
        reset_judge_parse_fallback_counts()
        self._run("")  # empty
        self._run("no json here")  # unparseable
        self._run('{"verdict": "maybe"}')  # bad_verdict
        assert judge_parse_fallback_counts() == {
            "empty": 1,
            "unparseable": 1,
            "bad_verdict": 1,
        }
        reset_judge_parse_fallback_counts()
        assert judge_parse_fallback_counts() == {}

    def test_successful_parse_does_not_increment(self) -> None:
        reset_judge_parse_fallback_counts()
        self._run('{"verdict": "sufficient", "reason": "ok",'
                  ' "reformulated_query": null}')
        self._run('{"verdict": "insufficient", "reason": "x",'
                  ' "reformulated_query": "q"}')
        assert judge_parse_fallback_counts() == {}

    def test_info_log_line_has_stable_format(self, caplog) -> None:
        reset_judge_parse_fallback_counts()
        with caplog.at_level(logging.INFO, logger="docpilot.agent.judge"):
            self._run("")  # first fallback: total=1
            self._run("still not json")  # second fallback: total=2
        lines = [r.getMessage() for r in caplog.records]
        assert (
            "Judge parse fallback: cause=empty -> sufficient, total=1" in lines
        )
        assert (
            "Judge parse fallback: cause=unparseable -> sufficient, total=2"
            in lines
        )


# ---------------------------------------------------------------------------
# ScoreFloorBackstopJudge
# ---------------------------------------------------------------------------


class TestScoreFloorBackstopJudge:
    """Hermetic tests for the score-floor backstop (PLAN §H finding 6)."""

    def _make_result(self, score: float = 0.9, text: str = "chunk") -> RetrieverResult:
        return RetrieverResult(
            chunk=Chunk(id="c1", content=text, heading_path="/h", source_file="s.md", chunk_index=0),
            score=score,
        )

    def _make_judge(self, verdict: str = "sufficient") -> LLMSufficiencyJudge:
        payload = (
            '{"verdict": "' + verdict + '", "reason": "stub", '
            '"reformulated_query": null, "needs_tool": false}'
        )
        return LLMSufficiencyJudge(StubGenerator(payload))

    def test_pass_through_when_floor_disabled(self) -> None:
        """floor=0.0 must never override the LLM verdict, even with a weak score."""
        judge = ScoreFloorBackstopJudge(self._make_judge("sufficient"), score_floor=0.0)
        j = judge.judge("q", [self._make_result(0.01)], "q")
        assert j.verdict == "sufficient"

    def test_pass_through_when_above_floor(self) -> None:
        judge = ScoreFloorBackstopJudge(self._make_judge("sufficient"), score_floor=0.5)
        j = judge.judge("q", [self._make_result(0.8)], "q")
        assert j.verdict == "sufficient"

    def test_forces_insufficient_when_below_floor(self) -> None:
        judge = ScoreFloorBackstopJudge(self._make_judge("sufficient"), score_floor=0.5)
        j = judge.judge("q", [self._make_result(0.3)], "q")
        assert j.verdict == "insufficient"
        assert "score floor" in j.reason

    def test_forces_insufficient_when_results_empty(self) -> None:
        judge = ScoreFloorBackstopJudge(self._make_judge("sufficient"), score_floor=0.5)
        j = judge.judge("q", [], "q")
        assert j.verdict == "insufficient"

    def test_preserves_needs_tool_when_floor_overrides(self) -> None:
        """Even when the backstop overrides verdict, needs_tool stays true so
        the graph can still route to the GitHub tool (Phase 3)."""
        payload = (
            '{"verdict": "sufficient", "reason": "stub", '
            '"reformulated_query": null, "needs_tool": true, '
            '"tool_request": {"name": "github.search_issues", "params": {"q": "x"}}}'
        )
        judge = ScoreFloorBackstopJudge(
            LLMSufficiencyJudge(StubGenerator(payload)), score_floor=0.5
        )
        j = judge.judge("q", [self._make_result(0.1)], "q")
        assert j.verdict == "insufficient"
        assert j.needs_tool is True
        assert j.tool_request is not None