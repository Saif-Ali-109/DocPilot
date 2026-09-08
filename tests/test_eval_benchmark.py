"""Hermetic tests for the Phase 4 classic-vs-agentic benchmark harness (SPEC §6.1).

No LLM, no network — a scripted runner drives every metric, and the committed
benchmark dataset itself is loaded and schema-checked (it is the comparison
contract — a bad row must fail loudly).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docpilot.agent.types import LoopTraceStep
from docpilot.eval import benchmark as bm
from docpilot.eval.benchmark import (
    BenchmarkQuestion,
    RunOutput,
    answer_body,
    body_markers,
    build_comparison,
    format_comparison,
    load_benchmark,
    main,
    report_to_dict,
    run_pipeline,
)

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "docpilot" / "eval" / "dataset" / "benchmark.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _question(
    qid: str = "q01",
    category: str = "docs-answerable",
    *,
    facts: tuple[str, ...] = ("alpha fact", "beta fact"),
    sources: tuple[str, ...] = ("docs/en/docs/tutorial/a.md",),
    gold_refusal: bool = False,
    expected_tool: str | None = None,
) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=qid,
        category=category,
        question=f"question {qid}?",
        gold_key_facts=facts,
        gold_sources=sources,
        gold_refusal=gold_refusal,
        expected_tool=expected_tool,
    )


def _scripted(
    *,
    answer: str = "",
    source_files: list[str] | None = None,
    refused: bool = False,
    trace_steps: list[LoopTraceStep] | None = None,
    latency_ms: float = 10.0,
):
    """Factory for a one-shot runner (same output for every question)."""
    files = source_files if source_files is not None else []
    steps = trace_steps if trace_steps is not None else []

    def run(question: str) -> RunOutput:
        return RunOutput(
            answer=answer,
            source_files=list(files),
            refused=refused,
            trace_steps=list(steps),
            latency_ms=latency_ms,
        )

    return run


def _search_step(n: int = 1) -> LoopTraceStep:
    return LoopTraceStep.new("search", f"query {n}", f"retrieve-{n}")


def _tool_step() -> LoopTraceStep:
    return LoopTraceStep.new("tool_call", "query", "tool_call")


# ---------------------------------------------------------------------------
# Committed dataset — the contract
# ---------------------------------------------------------------------------


class TestCommittedDataset:
    def test_loads_and_category_counts(self):
        q = load_benchmark(DATASET_PATH)
        assert len(q) == 15
        assert {c: sum(1 for x in q if x.category == c) for c in bm.VALID_CATEGORIES} == {
            "docs-answerable": 8,
            "live-state-answerable": 4,
            "neither": 3,
        }
        ids = [x.id for x in q]
        assert len(set(ids)) == len(ids), "question ids must be unique"

    @pytest.mark.parametrize("category", ["docs-answerable", "live-state-answerable", "neither"])
    def test_gold_pattern_per_category(self, category):
        q = load_benchmark(DATASET_PATH)
        rows = [x for x in q if x.category == category]
        assert rows
        if category == "docs-answerable":
            assert all(x.gold_key_facts and x.gold_sources for x in rows)
            assert all(x.gold_refusal is False for x in rows)
            assert all(x.expected_tool is None for x in rows)
        elif category == "live-state-answerable":
            assert all(not x.gold_key_facts and not x.gold_sources for x in rows)
            assert all(x.gold_refusal is False for x in rows)
            assert all(x.expected_tool in {"github.search_issues", "github.list_issues", "github.get_commits"} for x in rows)
        else:
            assert all(not x.gold_key_facts and not x.gold_sources for x in rows)
            assert all(x.gold_refusal is True for x in rows)
            assert all(x.expected_tool is None for x in rows)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    @pytest.fixture()
    def tmp_ds(self, tmp_path):
        def write(payload):
            p = tmp_path / "bench.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            return p

        return write

    def _base(self, **over):
        d = {
            "id": "b1",
            "category": "docs-answerable",
            "question": "q?",
            "gold_key_facts": ["f"],
            "gold_sources": ["docs/en/docs/a.md"],
            "gold_refusal": False,
        }
        d.update(over)
        return d

    def test_bad_category(self, tmp_ds):
        with pytest.raises(ValueError, match="bad category"):
            load_benchmark(tmp_ds({"questions": [self._base(category="wat")]}))

    def test_empty_question(self, tmp_ds):
        with pytest.raises(ValueError, match="'question' must be non-empty"):
            load_benchmark(tmp_ds({"questions": [self._base(question="  ")]}))

    def test_bad_facts(self, tmp_ds):
        with pytest.raises(ValueError, match="gold_key_facts"):
            load_benchmark(tmp_ds({"questions": [self._base(gold_key_facts=42)]}))

    def test_bad_sources(self, tmp_ds):
        with pytest.raises(ValueError, match="gold_sources"):
            load_benchmark(tmp_ds({"questions": [self._base(gold_sources=[""])]}))

    def test_bad_gold_refusal(self, tmp_ds):
        with pytest.raises(ValueError, match="gold_refusal.*bool"):
            load_benchmark(tmp_ds({"questions": [self._base(gold_refusal="yes")]}))

    def test_bare_list_also_accepted(self, tmp_ds):
        q = load_benchmark(tmp_ds([self._base()]))
        assert len(q) == 1 and q[0].id == "b1"


# ---------------------------------------------------------------------------
# Answer-body / citation helpers
# ---------------------------------------------------------------------------


class TestAnswerHelpers:
    def test_answer_body_strips_footer(self):
        display = "Answer with [1] marker.\n\nSources:\n[1] docs/en/docs/a.md"
        assert answer_body(display) == "Answer with [1] marker."

    def test_body_markers(self):
        assert body_markers("see [1] and [2], then [7]") == [1, 2, 7]
        assert body_markers("no markers") == []


# ---------------------------------------------------------------------------
# Scoring + aggregation (scripted runner)
# ---------------------------------------------------------------------------


class TestScoring:
    def test_docs_perfect(self):
        q = [_question("d1")]
        run = _scripted(
            answer="alpha fact and beta fact [1]",
            source_files=["docs/en/docs/tutorial/a.md"],
        )
        report = run_pipeline(q, "classic", run)
        assert len(report.rows) == 1
        r = report.rows[0]
        assert r.answer_correct == 1.0
        assert r.recall == 1.0
        assert r.marker_count == 1
        assert r.citation_validity == 1.0
        assert r.citation_gold_accuracy == 1.0
        assert r.retrieval_calls == 1  # empty trace → exactly one retrieval
        assert r.tool_calls == 0

    def test_docs_partial_facts(self):
        q = [_question("d1")]
        run = _scripted(answer="alpha fact only", source_files=[])
        report = run_pipeline(q, "classic", run)
        assert report.rows[0].answer_correct == 0.5

    def test_docs_fact_case_insensitive(self):
        q = [_question("d1")]
        run = _scripted(answer="ALPHA FACT and BETA FACT")
        report = run_pipeline(q, "classic", run)
        assert report.rows[0].answer_correct == 1.0

    def test_docs_recall_miss_and_bad_citation(self):
        q = [_question("d1")]
        run = _scripted(
            answer="alpha fact and beta fact [1] [9]",
            source_files=["docs/en/docs/other.md"],
        )
        report = run_pipeline(q, "classic", run)
        r = report.rows[0]
        assert r.recall == 0.0
        assert r.citation_validity == 0.5  # [1] valid, [9] out of range
        assert r.citation_gold_accuracy == 0.0  # none point to gold source

    def test_live_tool_fired(self):
        q = [_question("l1", "live-state-answerable", facts=(), sources=(), expected_tool="github.search_issues")]
        run = _scripted(
            answer="2 open issues [2]",
            refused=False,
            trace_steps=[_search_step(), _tool_step()],
            source_files=["docs/en/docs/tutorial/a.md", "github/fastapi/fastapi#123"],
        )
        report = run_pipeline(q, "agentic", run)
        r = report.rows[0]
        assert r.answer_correct == 1.0
        assert r.tool_calls == 1
        assert r.retrieval_calls == 1
        assert r.recall is None  # no gold sources for live state

    def test_live_answered_without_tool(self):
        q = [_question("l1", "live-state-answerable", facts=(), sources=())]
        report = run_pipeline(q, "agentic", _scripted(answer="answer", refused=False))
        assert report.rows[0].answer_correct == 0.5

    def test_live_refused(self):
        q = [_question("l1", "live-state-answerable", facts=(), sources=())]
        report = run_pipeline(q, "agentic", _scripted(answer="refusal", refused=True))
        assert report.rows[0].answer_correct == 0.0

    def test_neither_refused_is_correct(self):
        q = [_question("n1", "neither", facts=(), sources=(), gold_refusal=True)]
        report = run_pipeline(q, "classic", _scripted(answer="refusal", refused=True))
        assert report.rows[0].answer_correct == 1.0
        assert report.metrics.refusal_accuracy == 1.0

    def test_neither_answered_is_wrong(self):
        q = [_question("n1", "neither", facts=(), sources=(), gold_refusal=True)]
        report = run_pipeline(q, "classic", _scripted(answer="best is fastapi", refused=False))
        assert report.rows[0].answer_correct == 0.0
        assert report.metrics.refusal_accuracy == 0.0

    def test_retrieval_calls_counted_from_trace(self):
        q = [_question("d1")]
        run = _scripted(
            answer="alpha fact [1]",
            source_files=["docs/en/docs/tutorial/a.md"],
            trace_steps=[_search_step(1), _search_step(2)],
        )
        report = run_pipeline(q, "classic", run)
        assert report.rows[0].retrieval_calls == 2

    def test_groundedness_only_on_answered_rows(self):
        q = [_question("d1"), _question("n1", "neither", facts=(), sources=(), gold_refusal=True)]
        checked: list[str] = []
        checker = lambda answer, files: checked.append(answer) or True
        run = _scripted(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"])
        refused_run = _scripted(answer="I don't know", refused=True, source_files=[])
        # two separate runners per question via a map
        def runner(question: str) -> RunOutput:
            return refused_run(question) if question.startswith("question n1") else run(question)

        report = run_pipeline(q, "classic", runner, grounding_checker=checker)
        assert len(checked) == 1  # only the answered docs row
        assert report.rows[0].grounded is True
        assert report.rows[1].grounded is None
        assert report.metrics.groundedness_rate == 1.0
        assert report.metrics.groundedness_n == 1

    def test_ungrounded_row_counts_against_rate(self):
        q = [_question("d1")]
        report = run_pipeline(
            q, "classic", _scripted(answer="alpha fact [1]", source_files=["docs/en/docs/tutorial/a.md"]),
            grounding_checker=lambda a, f: False,
        )
        assert report.metrics.groundedness_rate == 0.0

    def test_aggregate_metrics_mix_categories(self):
        q = [
            _question("d1"),
            _question("l1", "live-state-answerable", facts=(), sources=()),
            _question("n1", "neither", facts=(), sources=(), gold_refusal=True),
        ]

        def runner(question: str) -> RunOutput:
            if question.startswith("question d1"):
                return RunOutput(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"], refused=False, latency_ms=5.0)
            if question.startswith("question l1"):
                return RunOutput(answer="2 open [2]", source_files=["a.md", "github/fastapi/fastapi#1"], refused=False, trace_steps=[_search_step(), _tool_step()], latency_ms=50.0)
            return RunOutput(answer="refusal", refused=True, source_files=[], latency_ms=5.0)

        report = run_pipeline(q, "agentic", runner)
        m = report.metrics
        assert m.answer_correctness == pytest.approx(1.0)  # 1.0 + 1.0 + 1.0
        assert m.retrieval_recall_at_k == 1.0
        assert m.recall_n == 1  # docs only
        assert m.refusal_accuracy == 1.0
        assert m.refusal_n == 1
        assert m.avg_latency_ms == pytest.approx(20.0)
        assert m.avg_retrieval_calls == pytest.approx((1 + 1 + 1) / 3)
        assert m.total_tool_calls == 1

    def test_recall_excluded_when_no_gold_sources(self):
        q = [_question("l1", "live-state-answerable", facts=(), sources=())]
        report = run_pipeline(q, "agentic", _scripted(answer="x", refused=False, trace_steps=[_tool_step()]))
        assert report.metrics.retrieval_recall_at_k is None
        assert report.metrics.recall_n == 0


# ---------------------------------------------------------------------------
# Comparison builder
# ---------------------------------------------------------------------------


def _metrics_report(pipeline: str, **over):
    from docpilot.eval.benchmark import PipelineMetrics, PipelineReport

    base = dict(
        answer_correctness=0.0,
        retrieval_recall_at_k=None,
        recall_n=0,
        citation_validity=None,
        citation_validity_n=0,
        citation_gold_accuracy=None,
        citation_gold_n=0,
        refusal_accuracy=None,
        refusal_n=0,
        groundedness_rate=None,
        groundedness_n=0,
        avg_latency_ms=0.0,
        avg_retrieval_calls=0.0,
        avg_tool_calls=0.0,
        total_tool_calls=0,
    )
    base.update(over)
    return PipelineReport(pipeline=pipeline, dataset_path="-", generated_at="t", rows=[], metrics=PipelineMetrics(**base))


class TestComparison:
    def test_higher_better_winner(self):
        classic = _metrics_report("classic", answer_correctness=0.9)
        agentic = _metrics_report("agentic", answer_correctness=0.95)
        cmp = build_comparison(classic, agentic)
        row = next(r for r in cmp.metrics if r.metric == "answer_correctness")
        assert row.winner == "agentic"

    def test_lower_better_latency(self):
        classic = _metrics_report("classic", avg_latency_ms=100.0)
        agentic = _metrics_report("agentic", avg_latency_ms=250.0)
        cmp = build_comparison(classic, agentic)
        row = next(r for r in cmp.metrics if r.metric == "avg_latency_ms")
        assert row.winner == "classic"

    def test_tie(self):
        classic = _metrics_report("classic", refusal_accuracy=1.0)
        agentic = _metrics_report("agentic", refusal_accuracy=1.0)
        cmp = build_comparison(classic, agentic)
        row = next(r for r in cmp.metrics if r.metric == "refusal_accuracy (I-don't-know)")
        assert row.winner == "tie"

    def test_n_a_when_both_none(self):
        classic = _metrics_report("classic")
        agentic = _metrics_report("agentic")
        cmp = build_comparison(classic, agentic)
        row = next(r for r in cmp.metrics if r.metric == "groundedness_rate")
        assert row.winner == "n/a"

    def test_summary_counts(self):
        classic = _metrics_report("classic", answer_correctness=0.5, avg_latency_ms=1.0)
        agentic = _metrics_report("agentic", answer_correctness=0.9, avg_latency_ms=5.0)
        cmp = build_comparison(classic, agentic)
        assert "agentic" in cmp.summary  # agentic wins answer correctness, classic wins latency

    def test_format_comparison_contains_winner_column(self):
        classic = _metrics_report("classic", answer_correctness=0.5)
        agentic = _metrics_report("agentic", answer_correctness=0.9)
        text = format_comparison(build_comparison(classic, agentic))
        assert "classic vs. agentic" in text.lower()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_report_to_dict_roundtrip(self):
        q = [_question("d1")]
        report = run_pipeline(
            q, "classic",
            _scripted(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"]),
            grounding_checker=lambda a, f: True,
        )
        d = report_to_dict(report)
        assert d["pipeline"] == "classic"
        assert d["metrics"]["answer_correctness"] == 1.0
        assert d["metrics"]["groundedness_rate"] == 1.0
        row = d["rows"][0]
        assert set(row) >= {
            "id", "category", "gold_refusal", "refused", "answer_correct", "recall",
            "citation_validity", "citation_gold_accuracy", "marker_count",
            "retrieval_calls", "tool_calls", "latency_ms", "grounded", "answer", "source_files",
        }


# ---------------------------------------------------------------------------
# Grounding checker (stub generator)
# ---------------------------------------------------------------------------


class TestGroundingChecker:
    class StubGenerator:
        def __init__(self, response: str):
            self.response = response

        def generate(self, prompt: str) -> str:
            return self.response

    def test_grounded_true(self):
        checker = bm.GroundingChecker(TestGroundingChecker.StubGenerator('{"grounded": true, "unsupported_claims": []}'))
        assert checker.check("answer", ["file.md"]) is True

    def test_grounded_false(self):
        checker = bm.GroundingChecker(TestGroundingChecker.StubGenerator('text {"grounded": false, "unsupported_claims": ["x"]} trailing'))
        assert checker.check("answer", ["file.md"]) is False

    def test_unparseable_returns_none(self):
        checker = bm.GroundingChecker(TestGroundingChecker.StubGenerator("not json at all"))
        assert checker.check("answer", ["file.md"]) is None

    def test_callable_form(self):
        checker = bm.GroundingChecker(
            TestGroundingChecker.StubGenerator('{"grounded": true, "unsupported_claims": []}')
        )
        assert checker("answer", ["file.md"]) is True


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCLI:
    def test_unknown_argument(self):
        assert main(["--bogus"]) == 2

    def test_missing_value(self):
        assert main(["--out"]) == 2

    def test_dispatches_live_run(self, monkeypatch):
        called = {}

        def fake(dataset_path=None, *, out_dir=None, run_groundedness=True, pipeline="both", stamp=None):
            called["path"] = dataset_path
            called["out"] = out_dir
            called["grounding"] = run_groundedness
            called["pipeline"] = pipeline
            return "fake"

        monkeypatch.setattr(bm, "run_benchmark_live", fake)
        assert main(["--dataset", "b.json", "--out", "/tmp/opencode/out", "--no-grounding"]) == 0
        assert called == {"path": "b.json", "out": "/tmp/opencode/out", "grounding": False, "pipeline": "both"}

    def test_main_module_subcommand_dispatch(self, monkeypatch):
        calls = []

        def fake_main(argv):
            calls.append(argv)
            return 0

        monkeypatch.setattr(bm, "main", fake_main)
        from docpilot.eval.__main__ import main as pkg_main

        assert pkg_main(["benchmark", "--no-grounding"]) == 0
        assert calls == [["--no-grounding"]]


# ---------------------------------------------------------------------------
# Checkpointing / resume (the quota-failure safety net)
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def _reports(self):
        q = [_question("d1")]
        classic = run_pipeline(
            q, "classic",
            _scripted(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"]),
            grounding_checker=lambda a, f: True,
        )
        agentic = run_pipeline(
            q, "agentic",
            _scripted(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"]),
            grounding_checker=lambda a, f: True,
        )
        return classic, agentic

    def test_sidecars_written_and_merged(self, tmp_path):
        classic, agentic = self._reports()
        bm._write_pipeline_file(tmp_path, "s1", classic)
        bm._write_pipeline_file(tmp_path, "s1", agentic)

        merged = bm.merge_benchmark_checkpoints(tmp_path, "s1")
        assert merged.exists()
        payload = json.loads(merged.read_text(encoding="utf-8"))
        assert set(payload["pipelines"]) == {"classic", "agentic"}
        assert payload["comparison"] is not None
        assert "PARTIAL" not in payload["note"]
        assert payload["pipelines"]["classic"]["metrics"]["answer_correctness"] == 1.0
        assert payload["pipelines"]["classic"]["rows"][0]["grounded"] is True

    def test_merge_requires_both_sidecars(self, tmp_path):
        classic, _ = self._reports()
        bm._write_pipeline_file(tmp_path, "s1", classic)
        with pytest.raises(FileNotFoundError, match="agentic"):
            bm.merge_benchmark_checkpoints(tmp_path, "s1")

    def test_run_benchmark_live_partial_writes_sidecar(self, tmp_path, monkeypatch):
        def scripted(question: str) -> RunOutput:
            return RunOutput(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"], refused=False)

        monkeypatch.setattr(bm, "_classic_run", scripted)
        monkeypatch.setattr(bm, "_agentic_run", scripted)

        out = bm.run_benchmark_live(out_dir=tmp_path, run_groundedness=False, pipeline="classic")
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["note"].startswith("PARTIAL")
        assert payload["comparison"] is None
        assert set(payload["pipelines"]) == {"classic"}
        sidecars = list(tmp_path.glob("benchmark_*_classic.json"))
        assert len(sidecars) == 1

    def test_run_benchmark_live_bad_pipeline_raises(self):
        with pytest.raises(ValueError, match="pipeline must be one of"):
            bm.run_benchmark_live(pipeline="wat")

    def test_cli_pipeline_flag(self, monkeypatch):
        called = {}

        def fake(dataset_path=None, *, out_dir=None, run_groundedness=True, pipeline="both", stamp=None):
            called["pipeline"] = pipeline
            return "fake"

        monkeypatch.setattr(bm, "run_benchmark_live", fake)
        assert main(["--pipeline", "agentic"]) == 0
        assert called["pipeline"] == "agentic"

    def test_cli_merge_flag(self, monkeypatch, tmp_path):
        got = {}

        def fake_merge(out_dir, stamp):
            got["dir"] = out_dir
            got["stamp"] = stamp
            return str(tmp_path / "merged.json")

        monkeypatch.setattr(bm, "merge_benchmark_checkpoints", fake_merge)
        assert main(["--merge", str(tmp_path), "s1"]) == 0
        assert got == {"dir": str(tmp_path), "stamp": "s1"}

    def test_resume_adopts_existing_sidecar(self, tmp_path, monkeypatch):
        """A crashed run's finished half is adopted when resuming under the same stamp."""
        classic, _ = self._reports()
        bm._write_pipeline_file(tmp_path, "r1", classic)

        def scripted(question: str) -> RunOutput:
            return RunOutput(answer="alpha fact and beta fact [1]", source_files=["docs/en/docs/tutorial/a.md"], refused=False)

        monkeypatch.setattr(bm, "_agentic_run", scripted)
        out = bm.run_benchmark_live(out_dir=tmp_path, run_groundedness=False, pipeline="agentic", stamp="r1")
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload["pipelines"]) == {"classic", "agentic"}
        assert payload["comparison"] is not None
        assert "PARTIAL" not in payload["note"]

    def test_cli_resume_flag(self, monkeypatch):
        called = {}

        def fake(dataset_path=None, *, out_dir=None, run_groundedness=True, pipeline="both", stamp=None):
            called["pipeline"] = pipeline
            called["stamp"] = stamp
            return "fake"

        monkeypatch.setattr(bm, "run_benchmark_live", fake)
        assert main(["--pipeline", "agentic", "--resume", "20260908_190539"]) == 0
        assert called == {"pipeline": "agentic", "stamp": "20260908_190539"}

    def test_cli_resume_requires_stamp_shape(self):
        assert main(["--resume", "not-a-stamp"]) == 2


# ---------------------------------------------------------------------------
# Quota headroom probe (SPEC §6.5 — gate live runs on --probe)
# ---------------------------------------------------------------------------


class TestProbe:
    """probe_quota(): 0 = functional OK, 1 = rate-limited / throttled, 2 = other.
    Functional gate only — TPD is not introspectable and never claimed.
    """

    def _patch_probe(self, monkeypatch, result_fn):
        class _FakeGen:
            def __init__(self, **kwargs):
                self._model = "probe-model"

            def probe(self):
                return result_fn()

        monkeypatch.setattr("docpilot.generation.generator.GroqGenerator", _FakeGen)

    def test_probe_ok_reports_per_minute_headroom(self, monkeypatch, capsys):
        from docpilot.generation.generator import ProbeResult

        self._patch_probe(
            monkeypatch,
            lambda: ProbeResult(
                ok=True,
                completion="OK",
                limit=8000,
                used=1000,
                remaining=7000,
                requests_remaining=950,
            ),
        )
        assert bm.probe_quota() == 0
        out = capsys.readouterr().out
        assert "probe: OK" in out
        assert "tokens remaining=7000/8000" in out
        assert "requests remaining=950/1000" in out
        # honest caveat: daily TPD is not exposed and must be operator-confirmed
        assert "NOT exposed" in out

    def test_probe_rate_limited_prints_limit_used(self, monkeypatch, capsys):
        from docpilot.generation.generator import ProbeResult

        self._patch_probe(
            monkeypatch,
            lambda: ProbeResult(
                ok=False,
                reason=(
                    "RateLimitError: Rate limit reached for model "
                    "`openai/gpt-oss-20b` on tokens per day (TPD): "
                    "Limit 200000, Used 199455, Requested 2700. Please try "
                    "again in 15m30.96s."
                ),
            ),
        )
        assert bm.probe_quota() == 1
        out = capsys.readouterr().out
        assert "RATE LIMITED" in out
        assert "Limit=200000 Used=199455" in out

    def test_probe_throttled_per_minute_returns_one(self, monkeypatch, capsys):
        from docpilot.generation.generator import ProbeResult

        self._patch_probe(
            monkeypatch,
            lambda: ProbeResult(
                ok=True,
                completion="OK",
                limit=8000,
                used=7990,
                remaining=10,
                requests_remaining=1,
            ),
        )
        assert bm.probe_quota() == 1
        out = capsys.readouterr().out
        assert "GATE FAIL" in out
        assert "per-minute tokens remaining 10 < 1000" in out

    def test_probe_missing_headers_default_ok(self, monkeypatch, capsys):
        from docpilot.generation.generator import ProbeResult

        # No x-ratelimit headers: nothing to gate on beyond a successful call.
        self._patch_probe(monkeypatch, lambda: ProbeResult(ok=True, completion="OK"))
        assert bm.probe_quota() == 0
        out = capsys.readouterr().out
        assert "tokens remaining=None/None" in out

    def test_probe_other_error_returns_two(self, monkeypatch, capsys):
        from docpilot.generation.generator import ProbeResult

        self._patch_probe(
            monkeypatch,
            lambda: ProbeResult(ok=False, reason="AuthenticationError: 401 invalid key"),
        )
        assert bm.probe_quota() == 2
        assert "FAILED (AuthenticationError: 401 invalid key)" in capsys.readouterr().out

    def test_cli_probe_flag_returns_probe_code(self, monkeypatch):
        monkeypatch.setattr(bm, "probe_quota", lambda: 1)
        assert main(["--probe"]) == 1