"""Hermetic tests for the Phase 4 tool-necessity tri-class harness (SPEC §6.2/§6.4).

No LLM, no network — scripted judges drive the FP/FN math, and the committed
dataset itself is loaded and schema-checked (it is the Phase 3 necessity
contract — a bad row must fail loudly).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docpilot import eval as eval_pkg
from docpilot.agent.types import Judgment
from docpilot.eval import tool_necessity as tn
from docpilot.eval.tool_necessity import (
    ToolNecessityTriple,
    TripleChunk,
    evaluate_tool_necessity,
    format_report,
    load_tool_necessity_triples,
    main,
    report_to_dict,
)

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "docpilot" / "eval" / "dataset" / "tool_necessity.json"
)
CLASSES = ("docs-answerable", "live-state-answerable", "neither")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triple(
    tid: str = "x01",
    cls: str = "docs-answerable",
    *,
    gold_verdict: str = "sufficient",
    gold_needs_tool: bool = False,
    gold_tool_action: str | None = None,
) -> ToolNecessityTriple:
    return ToolNecessityTriple(
        id=tid,
        cls=cls,
        question=f"question {tid}?",
        query_used=f"question {tid}?",
        gold_verdict=gold_verdict,
        gold_needs_tool=gold_needs_tool,
        gold_tool_action=gold_tool_action,
        chunks=(TripleChunk(content="Evidence text.", source_file="docs/en/docs/tutorial/x.md"),),
    )


def _triples_for_classes() -> list[ToolNecessityTriple]:
    """One triple per class, matching the gold pattern of the contract."""
    return [
        _triple("d01", "docs-answerable", gold_verdict="sufficient", gold_needs_tool=False),
        _triple(
            "l01",
            "live-state-answerable",
            gold_verdict="insufficient",
            gold_needs_tool=True,
            gold_tool_action="github.search_issues",
        ),
        _triple("n01", "neither", gold_verdict="insufficient", gold_needs_tool=False),
    ]


def _judgment(verdict: str, needs_tool: bool = False, tool_request=None) -> Judgment:
    return Judgment(verdict=verdict, reason="stub", needs_tool=needs_tool, tool_request=tool_request)


# ---------------------------------------------------------------------------
# Committed dataset — the contract
# ---------------------------------------------------------------------------


class TestCommittedDataset:
    def test_loads_and_class_counts(self):
        t = load_tool_necessity_triples(DATASET_PATH)
        assert len(t) == 15
        assert {c: sum(1 for x in t if x.cls == c) for c in CLASSES} == {
            "docs-answerable": 5,
            "live-state-answerable": 5,
            "neither": 5,
        }
        ids = [x.id for x in t]
        assert len(set(ids)) == len(ids), "triple ids must be unique"

    @pytest.mark.parametrize("cls", CLASSES)
    def test_gold_pattern_per_class(self, cls):
        t = load_tool_necessity_triples(DATASET_PATH)
        rows = [x for x in t if x.cls == cls]
        assert rows
        if cls == "docs-answerable":
            assert all(x.gold_verdict == "sufficient" for x in rows)
            assert all(x.gold_needs_tool is False for x in rows)
            assert all(x.gold_tool_action is None for x in rows)
        elif cls == "live-state-answerable":
            assert all(x.gold_verdict == "insufficient" for x in rows)
            assert all(x.gold_needs_tool is True for x in rows)
            assert all(x.gold_tool_action in tn.VALID_ACTIONS for x in rows)
        else:
            assert all(x.gold_verdict == "insufficient" for x in rows)
            assert all(x.gold_needs_tool is False for x in rows)
            assert all(x.gold_tool_action is None for x in rows)

    def test_every_chunk_has_real_shape(self):
        t = load_tool_necessity_triples(DATASET_PATH)
        for x in t:
            assert x.chunks, x.id
            for c in x.chunks:
                assert c.content.strip() and c.source_file.strip()


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    @pytest.fixture()
    def tmp_dataset(self, tmp_path):
        def write(payload):
            p = tmp_path / "ds.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            return p

        return write

    def _base(self, **over):
        d = {
            "id": "t1",
            "class": "docs-answerable",
            "question": "q?",
            "query_used": "q?",
            "gold_verdict": "sufficient",
            "gold_needs_tool": False,
            "gold_tool_action": None,
            "chunks": [{"content": "c", "source_file": "docs/en/docs/a.md"}],
        }
        d.update(over)
        return d

    def test_bad_class(self, tmp_dataset):
        with pytest.raises(ValueError, match="bad class"):
            load_tool_necessity_triples(tmp_dataset({"triples": [self._base(**{"class": "wat"})]}))

    def test_bad_gold_verdict(self, tmp_dataset):
        with pytest.raises(ValueError, match="bad gold_verdict"):
            load_tool_necessity_triples(tmp_dataset({"triples": [self._base(gold_verdict="maybe")]}))

    def test_non_bool_gold_needs_tool(self, tmp_dataset):
        with pytest.raises(ValueError, match="gold_needs_tool.*bool"):
            load_tool_necessity_triples(tmp_dataset({"triples": [self._base(gold_needs_tool="yes")]}))

    def test_bad_gold_tool_action(self, tmp_dataset):
        with pytest.raises(ValueError, match="bad gold_tool_action"):
            load_tool_necessity_triples(
                tmp_dataset(
                    {
                        "triples": [
                            self._base(
                                cls="live-state-answerable",
                                gold_verdict="insufficient",
                                gold_needs_tool=True,
                                gold_tool_action="github.poke",
                            )
                        ]
                    }
                )
            )

    def test_empty_chunks(self, tmp_dataset):
        with pytest.raises(ValueError, match="'chunks' must be non-empty"):
            load_tool_necessity_triples(tmp_dataset({"triples": [self._base(chunks=[])]}))

    def test_chunk_missing_content(self, tmp_dataset):
        with pytest.raises(ValueError, match="'content' must be non-empty"):
            load_tool_necessity_triples(
                tmp_dataset({"triples": [self._base(chunks=[{"source_file": "a.md"}])]})
            )

    def test_chunk_missing_source(self, tmp_dataset):
        with pytest.raises(ValueError, match="'source_file' must be non-empty"):
            load_tool_necessity_triples(
                tmp_dataset({"triples": [self._base(chunks=[{"content": "c"}])]})
            )

    def test_bare_list_also_accepted(self, tmp_dataset):
        triples = load_tool_necessity_triples(tmp_dataset([self._base()]))
        assert len(triples) == 1 and triples[0].id == "t1"


# ---------------------------------------------------------------------------
# Harness metrics (scripted judges)
# ---------------------------------------------------------------------------


class ScriptedJudge:
    """Returns scripted verdict/tool per triple id."""

    def __init__(self, table):
        self.table = table  # id -> (verdict, needs_tool, tool_request)
        self.calls: list[str] = []

    def judge(self, question, results, query_used, *, tools_available=True):
        tid = results[0].chunk.id.split("-")[0]
        self.calls.append(tid)
        verdict, needs_tool, tool_request = self.table[tid]
        return _judgment(verdict, needs_tool, tool_request)


class TestHarnessMetrics:
    def test_perfect_judge(self):
        table = {}
        for x in _triples_for_classes():
            req = {"name": x.gold_tool_action, "params": {}} if x.gold_tool_action else None
            table[x.id] = (x.gold_verdict, x.gold_needs_tool, req)
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

        assert report.n == 3
        assert report.verdict_accuracy == 1.0
        assert report.false_positive_rate == 0.0
        assert report.false_negative_rate == 0.0
        assert report.neither_fire_rate == 0.0
        assert report.tool_request_valid_rate == 1.0
        assert report.tool_action_match_rate == 1.0
        assert all(r.correct_verdict for r in report.rows)
        assert all(r.predicted_needs_tool == r.gold_needs_tool for r in report.rows)

    def test_always_fire_judge_exposes_false_positives(self):
        always = {"name": "github.search_issues", "params": {"query": "x"}}
        table = {
            "d01": ("insufficient", True, always),
            "l01": ("insufficient", True, always),
            "n01": ("insufficient", True, always),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

        assert report.false_positive_rate == 1.0  # docs-answerable fired
        assert report.neither_fire_rate == 1.0
        assert report.false_negative_rate == 0.0
        assert report.verdict_accuracy == pytest.approx(2 / 3)  # only 2-of-3 verdicts correct
        assert report.tool_request_valid_rate == 1.0
        assert report.tool_action_match_rate == pytest.approx(1 / 3)  # only the live gold is search_issues
        assert report.per_class["docs-answerable"]["needs_tool_accuracy"] == 0.0
        assert report.per_class["neither"]["needs_tool_accuracy"] == 0.0

    def test_never_fire_judge_exposes_false_negatives(self):
        table = {
            "d01": ("sufficient", False, None),
            "l01": ("insufficient", False, None),
            "n01": ("insufficient", False, None),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

        assert report.false_negative_rate == 1.0  # live-state-answerable did not fire
        assert report.false_positive_rate == 0.0
        assert report.tool_request_valid_rate == 1.0  # vacuously true, nothing fired
        assert report.tool_action_match_rate == 1.0
        assert report.per_class["live-state-answerable"]["needs_tool_accuracy"] == 0.0

    def test_fired_without_tool_request_is_invalid(self):
        table = {
            "d01": ("sufficient", False, None),
            "l01": ("insufficient", True, None),  # needs_tool true but malformed
            "n01": ("insufficient", False, None),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

        fired = [r for r in report.rows if r.predicted_needs_tool]
        assert [r.id for r in fired] == ["l01"]
        assert report.tool_request_valid_rate == 0.0
        assert report.tool_action_match_rate == 0.0
        row = report.rows[1]
        assert row.tool_request_valid is False
        assert row.tool_action_matches is False
        assert row.predicted_action is None

    def test_wrong_action_does_not_match(self):
        req = {"name": "github.get_commits", "params": {}}
        table = {
            "d01": ("sufficient", False, None),
            "l01": ("insufficient", True, req),  # gold action is search_issues
            "n01": ("insufficient", False, None),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

        assert report.tool_action_match_rate == 0.0
        assert report.rows[1].predicted_action == "github.get_commits"
        assert report.rows[1].tool_action_matches is False

    def test_unexpected_verdict_token_raises(self):
        table = {
            "d01": ("maybe", False, None),
            "l01": ("insufficient", True, {"name": "github.search_issues", "params": {}}),
            "n01": ("insufficient", False, None),
        }
        with pytest.raises(RuntimeError, match="unexpected verdict"):
            evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))

    def test_tools_available_flag_is_true(self):
        seen = []

        class FlagJudge:
            def judge(self, question, results, query_used, *, tools_available=True):
                seen.append(tools_available)
                return _judgment("sufficient")

        evaluate_tool_necessity(_triples_for_classes(), FlagJudge())
        assert seen and all(v is True for v in seen)

    def test_parse_fallback_attribution(self, monkeypatch):
        import docpilot.eval.tool_necessity as tn_mod

        store = {"total": 0}

        def fake_reset():
            store["total"] = 0

        def fake_counts():
            return {"empty": store["total"]}

        monkeypatch.setattr(tn_mod, "reset_judge_parse_fallback_counts", fake_reset)
        monkeypatch.setattr(tn_mod, "judge_parse_fallback_counts", fake_counts)

        class FallbackJudge:
            def judge(self, question, results, query_used, *, tools_available=True):
                tid = results[0].chunk.id.split("-")[0]
                if tid == "xa":
                    store["total"] += 1
                return _judgment("sufficient")

        t1 = _triple("xa")
        t2 = _triple("xb")
        report = evaluate_tool_necessity([t1, t2], FallbackJudge())

        row_by_id = {r.id: r for r in report.rows}
        assert row_by_id["xa"].parse_fallback is True
        assert row_by_id["xb"].parse_fallback is False


# ---------------------------------------------------------------------------
# Report serialization / formatting
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_report_to_dict_roundtrip(self):
        search = {"name": "github.search_issues", "params": {"query": "x"}}
        table = {
            "d01": ("sufficient", False, None),
            "l01": ("insufficient", True, search),
            "n01": ("insufficient", False, None),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))
        d = report_to_dict(report)

        assert d["n"] == 3
        assert d["verdict_accuracy"] == 1.0
        assert isinstance(d["false_positive_rate"], float)
        assert d["per_class"]["docs-answerable"]["fires"] == 0
        row = d["rows"][1]
        assert row["id"] == "l01"
        assert row["predicted_action"] == "github.search_issues"
        assert set(row) >= {
            "id", "class", "gold_verdict", "predicted_verdict", "correct_verdict",
            "gold_needs_tool", "predicted_needs_tool", "gold_tool_action",
            "predicted_action", "tool_request_valid", "tool_action_matches", "parse_fallback",
        }

    def test_format_report_highlights_wrong_rows(self):
        always = {"name": "github.search_issues", "params": {"query": "x"}}
        table = {
            "d01": ("insufficient", True, always),
            "l01": ("insufficient", True, always),
            "n01": ("insufficient", True, always),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))
        text = format_report(report)

        assert "false_positive_rate" in text
        assert "✗ d01" in text
        assert "1.000" in text

    def test_format_report_surfaces_action_only_mismatch(self):
        # Verdict and needs_tool are correct; only the action choice differs —
        # must still be flagged (regression: previously suppressed on
        # otherwise-correct rows, hiding the tl03 live-run finding).
        wrong = {"name": "github.get_commits", "params": {}}
        table = {
            "d01": ("sufficient", False, None),
            "l01": ("insufficient", True, wrong),  # gold action is search_issues
            "n01": ("insufficient", False, None),
        }
        report = evaluate_tool_necessity(_triples_for_classes(), ScriptedJudge(table))
        text = format_report(report)

        assert "✗ l01" in text
        assert "action github.get_commits≠github.search_issues" in text


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCLI:
    def test_unknown_argument(self):
        assert main(["--bogus"]) == 2

    def test_missing_value(self):
        assert main(["--triples"]) == 2

    def test_dispatches_live_run(self, monkeypatch):
        called = {}

        def fake(triples_path=None, *, out_dir=None):
            called["path"] = triples_path
            called["out"] = out_dir
            return "fake"

        monkeypatch.setattr(tn, "run_tool_necessity_live", fake)
        assert main(["--triples", "ds.json", "--out", "/tmp/opencode/out"]) == 0
        assert called == {"path": "ds.json", "out": "/tmp/opencode/out"}

    def test_main_module_subcommand_dispatch(self, monkeypatch):
        calls = []

        def fake_main(argv):
            calls.append(argv)
            return 0

        monkeypatch.setattr(tn, "main", fake_main)
        from docpilot.eval.__main__ import main as pkg_main

        assert pkg_main(["tool-necessity", "--out", "/tmp/opencode/x"]) == 0
        assert calls == [["--out", "/tmp/opencode/x"]]