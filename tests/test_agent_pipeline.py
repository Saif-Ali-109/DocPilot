"""Hermetic tests for the agentic pipeline + CLI strategy surface (PLAN §3.6).

Exercises :func:`docpilot.agent.pipeline_agentic.agentic_ask` and the CLI's
``ask --strategy`` surface with injected fakes only — no network, no DB, no
live LLM. Covers:

    (a) auto + simple → direct=True, answer identical to pipeline_ask.ask();
    (b) auto + complex → loop engaged, direct=False, citations + footer present;
    (c) strategy='direct' on a complex question → still Phase-1 output;
    (d) strategy='agentic' on a simple question → loop engaged;
    (e) invalid strategy → ValueError;
    (f) CLI main() with injected fakes: --strategy agentic + --json keys,
        invalid --strategy → argparse exit 2, no --agentic flag.
"""

from __future__ import annotations

import json

import pytest

from docpilot import cli, config
from docpilot.agent.pipeline_agentic import AgenticAgent, agentic_ask
from docpilot.agent.types import Judgment
from docpilot.citations.engine import StandardCitationEngine
from docpilot.pipeline_ask import ask as pipeline_ask
from docpilot.retrieval.retriever import SimpleRetriever
from docpilot.tools import ToolResult

from test_pipeline_e2e import (
    FIXTURE_DOCS,
    FakeEmbeddingProvider,
    FakeGenerator,
    InMemoryVectorStore,
)
from test_agent_graph import (
    FakeRetriever,
    StubJudge,
    StubTool,
    github_issue_result,
    make_result,
)

SIMPLE_Q = "How do I install FastAPI?"
COMPLEX_Q = "Combine path, query, and body parameters in one endpoint — what are the validation rules for each kind?"

CANNED = "To install FastAPI, run `pip install fastapi`. [1]"


def build_ask_wiring():
    """Phase 1 style fakes over the fixture corpus (retriever + generator)."""
    from docpilot.ingestion.chunker import MarkdownChunker

    chunker = MarkdownChunker()
    all_chunks = []
    for doc in FIXTURE_DOCS:
        all_chunks.extend(chunker.chunk(doc))
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.add(all_chunks, embedder.embed([c.content for c in all_chunks]))
    retriever = SimpleRetriever(embedder, store)
    generator = FakeGenerator(CANNED)
    return retriever, generator


def _agentic_fakes():
    """Components for the agentic path: canned retriever + stub judge."""
    retriever = FakeRetriever(
        {
            COMPLEX_Q: [
                make_result(content="Path, query and body parameter validation rules.", score=0.9),
                make_result(content="Each parameter kind validates differently.", score=0.7),
            ],
            "reformulated validation rules": [
                make_result(content="Now fully covered.", score=0.8),
            ],
        }
    )
    judge = StubJudge([Judgment(verdict="sufficient", reason="covers it")])
    gen = FakeGenerator("Each parameter kind has its own validation rules. [1]")
    return retriever, judge, gen


# ---------------------------------------------------------------------------
# (a) auto + simple → direct, identical to Phase 1
# ---------------------------------------------------------------------------


def test_auto_simple_is_direct_and_identical_to_phase1() -> None:
    retriever, gen = build_ask_wiring()
    engine = StandardCitationEngine()

    agent_res = agentic_ask(
        SIMPLE_Q,
        strategy="auto",
        retriever=retriever,
        generator=gen,
        citation_engine=engine,
    )
    phase1 = pipeline_ask(
        SIMPLE_Q, retriever=retriever, generator=gen, citation_engine=engine
    )

    assert agent_res.direct is True
    assert agent_res.answer == phase1.display
    assert [t.step for t in agent_res.trace] == ["gate"]
    assert agent_res.trace[0].decision == "direct"
    assert agent_res.refused is False


# ---------------------------------------------------------------------------
# top_k resolution: loop uses AGENT_LOOP_TOP_K, fast path keeps RETRIEVAL_TOP_K
# ---------------------------------------------------------------------------


def test_agentic_loop_retrieves_with_agent_loop_top_k_by_default() -> None:
    # PLAN §3.7 fix: the loop retrieve (and therefore the judge's evidence)
    # uses AGENT_LOOP_TOP_K (8) when the caller did not override top_k.
    retriever, judge, gen = _agentic_fakes()
    agent_res = agentic_ask(
        COMPLEX_Q,
        strategy="auto",
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
    )

    assert agent_res.direct is False
    assert retriever.calls, "retriever never invoked on the loop path"
    assert retriever.calls[0][1] == config.AGENT_LOOP_TOP_K


def test_direct_path_keeps_retrieval_top_k() -> None:
    # PLAN §3.7 guard: the fast path must NOT inherit the loop's broader
    # default — it stays at RETRIEVAL_TOP_K.
    retriever = FakeRetriever({SIMPLE_Q: [make_result(score=0.85)]})
    gen = FakeGenerator(CANNED)
    agent_res = agentic_ask(
        SIMPLE_Q,
        strategy="auto",
        retriever=retriever,
        generator=gen,
        citation_engine=StandardCitationEngine(),
    )

    assert agent_res.direct is True
    assert retriever.calls, "retriever never invoked on the fast path"
    assert retriever.calls[0][1] == config.RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# (b) auto + complex → loop engaged
# ---------------------------------------------------------------------------


def test_auto_complex_engages_loop_with_citations() -> None:
    retriever, judge, gen = _agentic_fakes()
    agent_res = agentic_ask(
        COMPLEX_Q,
        strategy="auto",
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
    )

    assert agent_res.direct is False
    assert judge.calls >= 1
    steps = [t.step for t in agent_res.trace]
    assert steps[0] == "gate"
    assert "search" in steps and "judge" in steps and "answer" in steps
    assert agent_res.trace[0].decision == "agentic"
    # citations + footer present in the display answer
    assert "[1]" in agent_res.answer
    assert "Sources:" in agent_res.answer
    assert agent_res.refused is False


# ---------------------------------------------------------------------------
# (c) strategy='direct' on a complex question → still Phase-1 output
# ---------------------------------------------------------------------------


def test_forced_direct_on_complex_stays_phase1() -> None:
    retriever, gen = build_ask_wiring()
    engine = StandardCitationEngine()

    agent_res = agentic_ask(
        COMPLEX_Q,
        strategy="direct",
        retriever=retriever,
        generator=gen,
        citation_engine=engine,
    )
    phase1 = pipeline_ask(
        COMPLEX_Q, retriever=retriever, generator=gen, citation_engine=engine
    )

    assert agent_res.direct is True
    assert agent_res.answer == phase1.display
    assert agent_res.trace[0].decision == "forced-direct"
    assert len(agent_res.trace) == 1


# ---------------------------------------------------------------------------
# (d) strategy='agentic' on a simple question → loop engaged
# ---------------------------------------------------------------------------


def test_forced_agentic_on_simple_engages_loop() -> None:
    retriever, judge, gen = _agentic_fakes()
    retriever.results_by_query.setdefault(SIMPLE_Q, [make_result(score=0.85)])
    agent_res = agentic_ask(
        SIMPLE_Q,
        strategy="agentic",
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
    )

    assert agent_res.direct is False
    assert agent_res.trace[0].decision == "forced-agentic"
    assert "judge" in [t.step for t in agent_res.trace]
    assert "answer" in [t.step for t in agent_res.trace]


# ---------------------------------------------------------------------------
# (e) invalid strategy → ValueError
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        agentic_ask(SIMPLE_Q, strategy="bogus")


# ---------------------------------------------------------------------------
# AgenticAgent wrapper
# ---------------------------------------------------------------------------


def test_agentic_agent_run_delegates_with_defaults_and_overrides() -> None:
    retriever, gen = build_ask_wiring()
    engine = StandardCitationEngine()
    agent = AgenticAgent(
        retriever=retriever,
        generator=gen,
        citation_engine=engine,
        strategy="auto",
    )
    res = agent.run(SIMPLE_Q)
    assert res.direct is True
    assert res.answer == pipeline_ask(
        SIMPLE_Q, retriever=retriever, generator=gen, citation_engine=engine
    ).display

    # Per-call override: force the agentic strategy on the agent.
    retriever2, judge, gen2 = _agentic_fakes()
    retriever2.results_by_query.setdefault(SIMPLE_Q, [make_result(score=0.85)])
    res2 = agent.run(SIMPLE_Q, strategy="agentic", retriever=retriever2, judge=judge, generator=gen2)
    assert res2.direct is False
    assert "judge" in [t.step for t in res2.trace]


# ---------------------------------------------------------------------------
# (f) CLI strategy surface
# ---------------------------------------------------------------------------


def _cli_agentic_fakes():
    retriever, judge, gen = _agentic_fakes()
    return {
        "retriever": retriever,
        "judge": judge,
        "generator": gen,
        "citation_engine": StandardCitationEngine(),
    }


def test_cli_strategy_agentic_json_includes_trace_and_strategy(capsys) -> None:
    code = cli.main(["ask", COMPLEX_Q, "--strategy", "agentic", "--json"], **_cli_agentic_fakes())
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["strategy"] == "agentic"
    assert payload["direct"] is False
    assert "trace" in payload
    trace = payload["trace"]
    assert trace and trace[0]["step"] == "gate"
    assert trace[0]["decision"] == "forced-agentic"
    # core Phase 1 keys remain present (additive, not replaced)
    assert {"question", "answer", "sources"} <= set(payload)
    assert "sources" in payload


def test_cli_auto_simple_json_is_direct(capsys) -> None:
    retriever, gen = build_ask_wiring()
    engine = StandardCitationEngine()
    code = cli.main(
        ["ask", SIMPLE_Q, "--json"],
        retriever=retriever,
        generator=gen,
        citation_engine=engine,
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["strategy"] == "auto"
    assert payload["direct"] is True
    assert payload["refused"] is False
    assert payload["trace"][0]["decision"] == "direct"


def test_cli_invalid_strategy_exits_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ask", SIMPLE_Q, "--strategy", "bogus"])
    assert excinfo.value.code == 2


def test_cli_rejects_agentic_alias() -> None:
    # There is NO --agentic flag (one way to do one thing, SPEC §4.1).
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ask", SIMPLE_Q, "--agentic"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# (g) Phase 3 — tool wiring through agentic_ask (SPEC §5)
# ---------------------------------------------------------------------------


def test_agentic_ask_with_tool_surfaces_github_evidence() -> None:
    """Hermetic end-to-end: judge needs_tool → tool call → cited answer with
    the tool's source in the footer."""
    judge = StubJudge(
        [
            Judgment(
                verdict="insufficient",
                reason="live issue state needed",
                needs_tool=True,
                tool_request={
                    "name": "github.search_issues",
                    "params": {"query": "oauth token expiration"},
                },
            )
        ]
    )
    tool = StubTool({"github.search_issues": github_issue_result()})
    # The generator cites the tool source ([3] — after the two doc chunks).
    gen = FakeGenerator("There is a live issue tracking this. [3]")

    agent_res = agentic_ask(
        COMPLEX_Q,
        strategy="auto",
        retriever=FakeRetriever(
            {
                COMPLEX_Q: [
                    make_result(content="Path, query and body parameter validation rules.", score=0.9),
                    make_result(content="Each parameter kind validates differently.", score=0.7),
                ],
            }
        ),
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
        tool=tool,
    )

    assert agent_res.direct is False
    assert agent_res.refused is False
    steps = [t.step for t in agent_res.trace]
    assert steps == ["gate", "search", "judge", "tool_call", "answer"]
    tool_step = next(t for t in agent_res.trace if t.step == "tool_call")
    assert tool_step.decision == "tool_call"
    assert tool_step.latency_ms is not None and tool_step.latency_ms >= 0

    # The tool's per-item source_label is a citeable SourceRef.
    github_refs = [s for s in agent_res.sources if s.file.startswith("github:")]
    assert len(github_refs) == 1
    assert github_refs[0].ref == 3
    assert github_refs[0].file == "github:acme/widget#42"
    assert github_refs[0].heading == "OAuth token expires"

    # A single tool call with the judge's params went through the interface.
    assert [c.name for c in tool.calls] == ["github.search_issues"]
    assert tool.calls[0].params == {"query": "oauth token expiration"}

    # The generator saw the live evidence section and the footer lines up.
    assert gen.last_context is not None
    assert "LIVE GITHUB EVIDENCE:" in gen.last_context
    # The prompt (format_sources) exposes the kind for the source-preference
    # rule; the user-facing footer (CitationEngine) keeps the clean format.
    assert "[3] github:acme/widget#42 (live) → OAuth token expires" in gen.last_sources
    assert "[3] github:acme/widget#42 → OAuth token expires" in agent_res.answer


def test_no_pat_means_no_tool_and_phase2_behavior(monkeypatch) -> None:
    """GITHUB_PAT="" → agentic_ask(..., tool=None) keeps Phase 2 behaviour:
    no tool node, no github sources, judge told tools are unavailable."""
    monkeypatch.setattr(config, "GITHUB_PAT", "")
    judge = StubJudge(
        [
            Judgment(
                verdict="insufficient",
                reason="would like live state but no tool is available",
                needs_tool=True,
                tool_request={"name": "github.search_issues", "params": {"query": "x"}},
            )
        ]
    )
    gen = FakeGenerator("Each parameter kind has its own validation rules. [1]")

    agent_res = agentic_ask(
        COMPLEX_Q,
        strategy="auto",
        retriever=FakeRetriever(
            {
                COMPLEX_Q: [
                    make_result(content="Path, query and body parameter validation rules.", score=0.9),
                    make_result(content="Each parameter kind validates differently.", score=0.7),
                ],
            }
        ),
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
    )

    assert agent_res.direct is False
    assert agent_res.refused is False
    steps = [t.step for t in agent_res.trace]
    # needs_tool degraded to plain insufficient → second search → sufficient.
    assert steps == ["gate", "search", "judge", "search", "judge", "answer"]
    assert "tool_call" not in steps
    assert not any(s.file.startswith("github:") for s in agent_res.sources)
    # The judge was told no tool is wired on every round.
    assert judge.tools_seen == [False, False]


def test_agentic_agent_accepts_tool_and_passes_it_through() -> None:
    agent = AgenticAgent()
    judge = StubJudge(
        [
            Judgment(
                verdict="insufficient",
                reason="live state needed",
                needs_tool=True,
                tool_request={"name": "github.search_issues", "params": {"query": "x"}},
            )
        ]
    )
    tool = StubTool({"github.search_issues": github_issue_result()})
    gen = FakeGenerator("Live evidence [3]")
    res = agent.run(
        COMPLEX_Q,
        strategy="agentic",
        retriever=FakeRetriever(
            {
                COMPLEX_Q: [
                    make_result(content="Docs chunk a.", score=0.9),
                    make_result(content="Docs chunk b.", score=0.7),
                ],
            }
        ),
        judge=judge,
        generator=gen,
        citation_engine=StandardCitationEngine(),
        tool=tool,
    )

    assert "tool_call" in [t.step for t in res.trace]
    assert any(s.file.startswith("github:") for s in res.sources)


# ---------------------------------------------------------------------------
# (h) Default-judge caching: _build_default_judge called at most once
# ---------------------------------------------------------------------------


def test_default_judge_is_cached_across_calls(monkeypatch) -> None:
    """Two agentic_ask calls with judge=None should build the judge only once.

    The second call must reuse the cached instance from ``_get_default_judge``
    rather than calling ``_build_default_judge`` again (WI-2).
    """
    from docpilot.agent.pipeline_agentic import _build_default_judge

    build_count = 0

    def _counting_build():
        nonlocal build_count
        build_count += 1
        return StubJudge([Judgment(verdict="sufficient", reason="cached judge")])

    monkeypatch.setattr(
        "docpilot.agent.pipeline_agentic._build_default_judge",
        _counting_build,
    )
    # Clear the cache so the first call actually exercises the builder path.
    monkeypatch.setattr(
        "docpilot.agent.pipeline_agentic._DEFAULT_JUDGE_CACHE",
        {},
    )

    retriever = FakeRetriever(
        {COMPLEX_Q: [make_result(content="chunk", score=0.9)]}
    )
    gen = FakeGenerator("Answer [1]")

    agentic_ask(
        COMPLEX_Q,
        strategy="agentic",
        retriever=retriever,
        generator=gen,
        citation_engine=StandardCitationEngine(),
        judge=None,
    )
    agentic_ask(
        COMPLEX_Q,
        strategy="agentic",
        retriever=retriever,
        generator=gen,
        citation_engine=StandardCitationEngine(),
        judge=None,
    )

    assert build_count == 1, (
        f"_build_default_judge was called {build_count} times; expected 1"
    )


# ---------------------------------------------------------------------------
# (i) Compiled-graph memoization: build_graph called at most once
# ---------------------------------------------------------------------------


def test_compiled_graph_is_reused_across_calls(monkeypatch) -> None:
    """Two agentic_ask calls with the same injected components compile once.

    The second call must reuse the cached compiled app from
    ``_get_compiled_graph`` rather than calling ``build_graph`` again (WI-3).
    """
    from collections import OrderedDict

    from docpilot.agent.graph import build_graph as real_build_graph

    build_count = 0

    def _counting_build(**kwargs):
        nonlocal build_count
        build_count += 1
        return real_build_graph(**kwargs)

    monkeypatch.setattr(
        "docpilot.agent.pipeline_agentic.build_graph",
        _counting_build,
    )
    # Clear the cache so the first call actually exercises the builder path.
    monkeypatch.setattr(
        "docpilot.agent.pipeline_agentic._COMPILED_GRAPH_CACHE",
        OrderedDict(),
    )
    # Keep the tool out of the loop (all-defaults tool creation would give a
    # fresh GitHubTool per call — different id → correctly different key).
    monkeypatch.setattr(config, "GITHUB_PAT", "")

    retriever, judge, gen = _agentic_fakes()
    citation_engine = StandardCitationEngine()  # single instance, reused

    res1 = agentic_ask(
        COMPLEX_Q,
        strategy="agentic",
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=citation_engine,
    )
    res2 = agentic_ask(
        COMPLEX_Q,
        strategy="agentic",
        retriever=retriever,
        judge=judge,
        generator=gen,
        citation_engine=citation_engine,
    )

    assert build_count == 1, (
        f"build_graph was called {build_count} times; expected 1"
    )
    # Behavioural parity: the reused graph answers identically.
    assert res1.refused is False and res2.refused is False
    assert res1.answer == res2.answer
    assert res1.sources == res2.sources
