"""Hermetic tests for the Phase 4 judge A/B harness (SPEC §6.2 first slice).

No LLM, no network — a scripted ``StubJudge`` drives the math, and the real
``LLMSufficiencyJudge`` + stub generator covers parse-fallback attribution.
The committed dataset itself is loaded and schema-checked (it is the
calibration contract — a bad row must fail loudly).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docpilot.agent.judge import LLMSufficiencyJudge, reset_judge_parse_fallback_counts
from docpilot.agent.types import Judgment
from docpilot.core.models import RetrieverResult
from docpilot.eval import judge_ab
from docpilot.generation.generator import Generator
from docpilot.eval.judge_ab import (
    build_ab_report,
    evaluate_prompt,
    triples_to_results,
)
from docpilot.eval.triples import (
    JudgeTriple,
    TripleChunk,
    load_judge_triples,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "src" / "docpilot" / "eval" / "dataset" / "judge_triples.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubJudge:
    """Duck-typed SufficiencyJudge returning scripted verdicts per triple id."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.calls: list[str] = []

    def judge(self, question, results, query_used, *, tools_available=True) -> Judgment:
        triple_id = results[0].chunk.id.split("-")[0]
        self.calls.append(triple_id)
        return Judgment(
            verdict=self.verdicts.get(triple_id, "sufficient"),
            reason="stub",
        )


def _triple(tid: str = "x01", gold: str = "sufficient", category: str = "adversarial") -> JudgeTriple:
    return JudgeTriple(
        id=tid,
        category=category,
        question=f"question {tid}?",
        query_used=f"question {tid}?",
        gold=gold,
        chunks=(
            TripleChunk(content="Evidence text.", source_file="docs/en/docs/tutorial/x.md"),
        ),
    )


class StubGenerator(Generator):
    """Duck-typed stub; now inherits from Generator to get the default
    ``generate_system_user`` (concatenation fallback)."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


class TestLoadJudgeTriples:
    def _write(self, tmp_path: Path, payload) -> Path:
        path = tmp_path / "triples.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_dataset_parses(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            {"triples": [_triple().as_dict()]},
        )
        triples = load_judge_triples(path)
        assert len(triples) == 1
        assert triples[0].id == "x01"
        assert triples[0].gold == "sufficient"
        assert triples[0].chunks[0].score == 0.8

    def test_flat_array_payload_parses(self, tmp_path) -> None:
        path = self._write(tmp_path, [_triple().as_dict()])
        assert len(load_judge_triples(path)) == 1

    def test_missing_required_key_raises(self, tmp_path) -> None:
        row = _triple().as_dict()
        del row["gold"]
        path = self._write(tmp_path, {"triples": [row]})
        with pytest.raises(ValueError, match="missing key"):
            load_judge_triples(path)

    def test_bad_gold_raises(self, tmp_path) -> None:
        row = _triple(gold="maybe").as_dict()
        path = self._write(tmp_path, {"triples": [row]})
        with pytest.raises(ValueError, match="bad gold"):
            load_judge_triples(path)

    def test_bad_category_raises(self, tmp_path) -> None:
        row = _triple(category="nonsense").as_dict()
        path = self._write(tmp_path, {"triples": [row]})
        with pytest.raises(ValueError, match="bad category"):
            load_judge_triples(path)

    def test_empty_chunks_raises(self, tmp_path) -> None:
        row = _triple().as_dict()
        row["chunks"] = []
        path = self._write(tmp_path, {"triples": [row]})
        with pytest.raises(ValueError, match="non-empty"):
            load_judge_triples(path)

    def test_committed_dataset_is_valid(self) -> None:
        """The shipped benchmark dataset must always load clean."""
        triples = load_judge_triples(DATASET_PATH)
        assert len(triples) >= 20
        golds = {t.gold for t in triples}
        assert golds == {"sufficient", "insufficient"}
        cats = {t.category for t in triples}
        assert "adversarial" in cats and "sufficient-partial" in cats
        assert "insufficient-missing" in cats
        # adversarial must contain both verdicts (paraphrase robustness both ways)
        adv_golds = {t.gold for t in triples if t.category == "adversarial"}
        assert adv_golds == {"sufficient", "insufficient"}
        # every paraphrase target must exist
        ids = {t.id for t in triples}
        for t in triples:
            if t.paraphrase_of:
                assert t.paraphrase_of in ids


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


class TestTriplesToResults:
    def test_builds_retriever_results(self) -> None:
        results = triples_to_results(_triple())
        assert isinstance(results[0], RetrieverResult)
        assert results[0].chunk.content == "Evidence text."
        assert results[0].score == 0.8


# ---------------------------------------------------------------------------
# evaluate_prompt math
# ---------------------------------------------------------------------------


class TestEvaluatePrompt:
    def test_perfect_accuracy(self) -> None:
        triples = [
            _triple("a", "sufficient", "sufficient-direct"),
            _triple("b", "insufficient", "insufficient-missing"),
            _triple("c", "sufficient", "adversarial"),
            _triple("d", "insufficient", "adversarial"),
        ]
        report = evaluate_prompt(triples, StubJudge({"a": "sufficient", "b": "insufficient", "c": "sufficient", "d": "insufficient"}), prompt_name="A")
        assert report.n == 4
        assert report.verdict_accuracy == 1.0
        assert report.adversarial_accuracy == 1.0
        assert report.parse_failures == []
        assert report.confusion["sufficient"] == {"sufficient": 2, "insufficient": 0}
        assert report.confusion["insufficient"] == {"sufficient": 0, "insufficient": 2}

    def test_lean_sufficient_stub_misses_refusals(self) -> None:
        triples = [
            _triple("a", "sufficient", "sufficient-direct"),
            _triple("b", "insufficient", "insufficient-missing"),
            _triple("c", "insufficient", "adversarial"),
        ]
        # Stub always says sufficient — matches the defensive default.
        report = evaluate_prompt(triples, StubJudge({}), prompt_name="A")
        assert report.verdict_accuracy == 1 / 3
        assert report.per_category["insufficient-missing"]["accuracy"] == 0.0
        assert report.per_category["sufficient-direct"]["accuracy"] == 1.0
        assert report.confusion["insufficient"]["sufficient"] == 2

    def test_unexpected_verdict_token_raises(self) -> None:
        triples = [_triple("a", "sufficient")]
        stub = StubJudge({"a": "hallucinate"})
        with pytest.raises(RuntimeError, match="unexpected verdict"):
            evaluate_prompt(triples, stub, prompt_name="A")

    def test_parse_fallback_attributed_per_triple(self) -> None:
        reset_judge_parse_fallback_counts()
        triples = [
            _triple("a", "sufficient"),
            _triple("b", "insufficient"),
        ]
        judge = LLMSufficiencyJudge(StubGenerator("this is not json at all"))
        report = evaluate_prompt(triples, judge, prompt_name="B")
        # garbage → defensive sufficient for both triples
        assert report.parse_failures == ["a", "b"]
        assert report.parse_failure_rate == 1.0
        assert report.verdict_accuracy == 0.5  # 'a' correct by luck


# ---------------------------------------------------------------------------
# build_ab_report winner logic
# ---------------------------------------------------------------------------


class TestBuildABReport:
    def _report(self, acc: float, parse_rate: float, adv: float | None = 0.5) -> judge_ab.PromptReport:
        rows = [
            judge_ab.TripleEvalRow(id="a", category="adversarial" if adv is not None else "sufficient-direct", gold="sufficient", predicted="sufficient", correct=True, parse_fallback=False)
        ]
        return judge_ab.PromptReport(
            prompt_name="X", results=rows, n=1, verdict_accuracy=acc,
            adversarial_accuracy=adv, parse_failure_rate=parse_rate,
        )

    def test_higher_verdict_accuracy_wins(self) -> None:
        ab = build_ab_report("d", self._report(0.9, 0.0), self._report(0.8, 0.0))
        assert ab.winner == "A"

    def test_lower_parse_rate_wins_on_tie(self) -> None:
        ab = build_ab_report("d", self._report(0.8, 0.2), self._report(0.8, 0.0))
        assert ab.winner == "B"

    def test_higher_adversarial_wins_on_full_tie(self) -> None:
        ab = build_ab_report("d", self._report(0.8, 0.0, 0.5), self._report(0.8, 0.0, 0.7))
        assert ab.winner == "B"

    def test_exact_tie_reports_tie(self) -> None:
        ab = build_ab_report("d", self._report(0.8, 0.0, 0.5), self._report(0.8, 0.0, 0.5))
        assert ab.winner == "tie"

    def test_serialization_round_trip(self) -> None:
        ab = build_ab_report("d", self._report(0.9, 0.0), self._report(0.8, 0.1))
        payload = judge_ab.ab_report_to_dict(ab)
        assert payload["winner"] == "A"
        assert payload["triple_count"] == 1
        assert payload["prompts"]["A"]["verdict_accuracy"] == 0.9
        json.dumps(payload)  # must be JSON-serializable