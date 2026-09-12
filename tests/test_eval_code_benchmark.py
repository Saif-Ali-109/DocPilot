"""Hermetic tests for Phase 6 T5: the code benchmark harness (PLAN §7.2 T5).

No LLM, no network — scripted runners drive every metric, the committed
code dataset is schema-checked (bad rows must fail loudly), and the
checkpoint/resume + TPD-abort recipes mirror the Phase 4 benchmark's
crash-safety contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docpilot.eval import code_benchmark as cb
from docpilot.eval.benchmark import QuotaExhausted
from docpilot.validation import RetrieveThenValidate

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "docpilot" / "eval" / "dataset" / "code_benchmark.json"
)

VALIDATOR = RetrieveThenValidate()

_EVIDENCE = [
    "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n'
    "```"
]
_GOOD_BLOCK = (
    "from fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}'
)
_BAD_IMPORT_BLOCK = "from flask import Flask\n\napp = Flask(__name__)"
_BAD_SYNTAX_BLOCK = "def broken(:\n    pass"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _out(
    *,
    question: str = "req",
    answer: str = "Here is the app [1]:\n\n```python\napp = FastAPI()\n```",
    source_files: list[str] | None = None,
    blocks: list[str] | None = None,
    verdicts: list | None = None,
    refused: bool = False,
    validation_failed: bool = False,
    turns: int = 1,
):
    verdicts = [] if verdicts is None else verdicts
    verdict = verdicts[0] if len(verdicts) == 1 else None
    return cb.CodeRunOutput(
        question=question,
        answer=answer,
        source_files=source_files if source_files is not None else [],
        code_blocks=blocks if blocks is not None else [],
        block_verdicts=list(verdicts),
        verdict=verdict,
        validation_failed=validation_failed,
        refused=refused,
        validation_turns=turns,
    )


def _pass_output(question: str = "req") -> cb.CodeRunOutput:
    v = VALIDATOR.validate(_GOOD_BLOCK, _EVIDENCE)
    return _out(
        question=question,
        answer='Here is the app [1]:\n\n```python\nfrom fastapi import FastAPI\n'
        "\napp = FastAPI()\n```",
        source_files=["en/docs/tutorial/first-steps.md"],
        blocks=[_GOOD_BLOCK],
        verdicts=[v],
    )


def _q(
    qid: str,
    *,
    imports=("fastapi",),
    surface=("FastAPI", "app", "get"),
    sources=("en/docs/tutorial/first-steps.md",),
    refusal: bool = False,
) -> cb.CodeBenchmarkQuestion:
    return cb.CodeBenchmarkQuestion(
        id=qid,
        request=f"request {qid}",
        gold_imports=imports,
        gold_surface=surface,
        gold_sources=sources,
        refusal_expected=refusal,
    )


# ---------------------------------------------------------------------------
# Committed dataset — the contract
# ---------------------------------------------------------------------------


class TestCommittedDataset:
    def test_loads_and_row_count(self):
        rows = cb.load_code_benchmark(DATASET_PATH)
        assert len(rows) == 6
        ids = [r.id for r in rows]
        assert len(set(ids)) == len(ids)

    def test_refusal_row_has_empty_golds(self):
        rows = cb.load_code_benchmark(DATASET_PATH)
        refusal = [r for r in rows if r.refusal_expected]
        assert len(refusal) == 1
        assert refusal[0].gold_imports == ()
        assert refusal[0].gold_surface == ()
        assert refusal[0].gold_sources == ()

    def test_code_rows_have_gold_sources_and_imports(self):
        rows = cb.load_code_benchmark(DATASET_PATH)
        code_rows = [r for r in rows if not r.refusal_expected]
        assert len(code_rows) == 5
        assert all(r.gold_sources for r in code_rows)
        assert all(r.gold_imports for r in code_rows)
        assert all(r.gold_surface for r in code_rows)

    def test_schema_rejects_missing_keys(self):
        payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        bad = payload["questions"][0].copy()
        del bad["gold_sources"]
        tmp = DATASET_PATH.parent / "_code_bad_missing.json"
        try:
            tmp.write_text(json.dumps([bad]), encoding="utf-8")
            with pytest.raises(ValueError, match="missing key"):
                cb.load_code_benchmark(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_schema_rejects_refusal_row_with_golds(self):
        tmp = DATASET_PATH.parent / "_code_bad_refusal.json"
        bad = {
            "id": "x",
            "request": "write django",
            "gold_imports": ["django"],
            "gold_surface": [],
            "gold_sources": [],
            "refusal_expected": True,
        }
        try:
            tmp.write_text(json.dumps([bad]), encoding="utf-8")
            with pytest.raises(ValueError, match="refusal_expected"):
                cb.load_code_benchmark(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_schema_rejects_answerable_row_without_sources(self):
        tmp = DATASET_PATH.parent / "_code_bad_nosrc.json"
        bad = {
            "id": "x",
            "request": "write code",
            "gold_imports": ["fastapi"],
            "gold_surface": ["FastAPI"],
            "gold_sources": [],
            "refusal_expected": False,
        }
        try:
            tmp.write_text(json.dumps([bad]), encoding="utf-8")
            with pytest.raises(ValueError, match="gold_sources"):
                cb.load_code_benchmark(tmp)
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_passing_row_scores_full(self):
        row = cb._score_row(_q("c01"), _pass_output(), _GOOD_BLOCK)
        assert row.validation_pass is True
        assert row.compiles is True
        assert row.imports_grounded is True
        assert row.surface_grounded is True
        assert row.used_gold_surface == 1.0
        assert row.used_gold_imports is True
        assert row.citation_gold_accuracy == 1.0
        assert row.marker_count == 1
        assert row.refusal_ok is True
        assert row.validation_turns == 1

    def test_bad_import_fails_validation(self):
        v = VALIDATOR.validate(_BAD_IMPORT_BLOCK, _EVIDENCE)
        out = _out(
            answer="[1]\n\n```python\nfrom flask import Flask\n```",
            source_files=["en/docs/tutorial/first-steps.md"],
            blocks=[_BAD_IMPORT_BLOCK],
            verdicts=[v],
        )
        row = cb._score_row(_q("c01"), out, _BAD_IMPORT_BLOCK)
        assert row.validation_pass is False
        assert row.compiles is True  # syntax fine
        assert row.imports_grounded is False
        # The import is ungrounded, but Flask is *bound* by that import and no
        # other external symbol is referenced — symbol grounding stays PASS.
        assert row.surface_grounded is True

    def test_uncovered_row_refused_scores_ok(self):
        out = cb.CodeRunOutput(
            question="c06",
            answer="I don't know — the available documentation does not cover this question.",
            source_files=[],
            code_blocks=[],
            block_verdicts=[],
            verdict=None,
            validation_failed=False,
            refused=True,
        )
        row = cb._score_row(_q("c06", imports=(), surface=(), sources=(), refusal=True), out, "")
        assert row.refusal_ok is True
        assert row.validation_pass is None
        assert row.compiles is None
        assert row.citation_gold_accuracy is None

    def test_uncovered_row_not_refused_fails(self):
        out = cb.CodeRunOutput(
            question="c06", answer="from django.db import models", source_files=[],
            code_blocks=["from django.db import models"], block_verdicts=[],
            verdict=None, validation_failed=False, refused=False,
        )
        row = cb._score_row(_q("c06", imports=(), surface=(), sources=(), refusal=True), out, "from django.db import models")
        assert row.refusal_ok is False

    def test_citation_marker_to_non_gold_source_lowers_accuracy(self):
        out = _pass_output()
        out.source_files = ["en/docs/advanced/other.md"]  # cited page not in gold
        row = cb._score_row(_q("c01"), out, _GOOD_BLOCK)
        assert row.citation_gold_accuracy == 0.0

    def test_syntax_error_block_fails_compile_check(self):
        out = _out(blocks=[_BAD_SYNTAX_BLOCK])
        row = cb._score_row(_q("c01"), out, _BAD_SYNTAX_BLOCK)
        assert row.compiles is False


# ---------------------------------------------------------------------------
# Aggregation + run loop
# ---------------------------------------------------------------------------


def _scripted(responses: dict[str, cb.CodeRunOutput]):
    def run(question: str) -> cb.CodeRunOutput:
        out = responses.get(question)
        if out is None:
            return _out(question=question)
        return out

    return run


def test_aggregate_math():
    qs = [_q("a"), _q("b"), _q("c", refusal=True, imports=(), surface=(), sources=())]
    out_b = _pass_output(question="request b")
    out_b.validation_turns = 3
    rows = [
        cb._score_row(qs[0], _pass_output(question="request a"), _GOOD_BLOCK),
        cb._score_row(qs[1], out_b, _GOOD_BLOCK),
        cb._score_row(
            qs[2],
            cb.CodeRunOutput(
                question="request c", answer="refusal", source_files=[],
                code_blocks=[], block_verdicts=[], verdict=None,
                validation_failed=False, refused=True,
            ),
            "",
        ),
    ]
    report = cb._aggregate(qs, rows, generated_at=None)
    m = report.metrics
    assert m.validation_pass_rate == 1.0
    assert m.validation_n == 2
    assert m.citation_gold_accuracy == 1.0
    assert m.citation_n == 2
    assert m.refusal_accuracy == 1.0
    assert m.refusal_n == 3  # refusal_ok computed for every row
    assert m.avg_validation_turns == (1 + 3 + 1) / 3


def test_run_loop_scores_and_checkpoints(tmp_path):
    qs = [_q("a"), _q("b")]
    runner = _scripted(
        {"request a": _pass_output(question="request a"),
         "request b": _pass_output(question="request b")}
    )
    checkpoint = tmp_path / "code.rows.jsonl"
    report = cb.run_code_benchmark(qs, runner, checkpoint_path=checkpoint)
    assert len(report.rows) == 2
    assert checkpoint.exists()
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "a"

    # Resume: new question joins the checkpointed rows — full-set report.
    calls = {"n": 0}

    def counting_run(question: str):
        calls["n"] += 1
        return _pass_output(question=question)

    qs3 = [_q("a"), _q("b"), _q("c")]
    report2 = cb.run_code_benchmark(qs3, counting_run, checkpoint_path=checkpoint)
    assert calls["n"] == 1  # only c ran
    assert [r.id for r in report2.rows] == ["a", "b", "c"]


def test_tpd_abort_propagates():
    def boom(question: str):
        raise RuntimeError("rate limit ... tokens per day (TPD): used")

    with pytest.raises(QuotaExhausted):
        cb.run_code_benchmark([_q("a")], boom)


def test_runner_contract_enforced():
    def bad(question: str):
        return None

    with pytest.raises(TypeError, match="must return CodeRunOutput"):
        cb.run_code_benchmark([_q("a")], bad)


def test_report_round_trip(tmp_path):
    qs = [_q("a")]
    report = cb.run_code_benchmark(
        qs, _scripted({"request a": _pass_output(question="request a")})
    )
    payload = cb.report_to_dict(report)
    assert payload["pipeline"] == "code"
    assert payload["metrics"]["validation_pass_rate"] == 1.0
    assert payload["rows"][0]["id"] == "a"
    text = cb.format_metrics(report)
    assert "validation-pass" in text
    assert "citation gold acc" in text