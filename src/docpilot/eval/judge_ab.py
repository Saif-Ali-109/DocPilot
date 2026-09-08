"""Judge two-prompt A/B calibration harness (SPEC §6.2 — first slice).

Runs two judge system prompts (variant A = production ``JUDGE_SYSTEM_PROMPT``,
variant B = ``JUDGE_SYSTEM_PROMPT_B``) over the same labeled verdict triples
and reports verdict accuracy, adversarial accuracy, and parse-failure rate.
Prompt changes are adopted on this calibration data alone — never on judgment.

The core (:func:`evaluate_prompt`, :func:`build_ab_report`) is hermetic and
LLM-free-agnostic: it drives any :class:`~docpilot.agent.judge.SufficiencyJudge`,
so tests stub the judge and the live runner uses a real Groq-backed judge.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from docpilot import config
from docpilot.agent.judge import (
    LLMSufficiencyJudge,
    SufficiencyJudge,  # noqa: F401  (re-export for tests/stubs)
    judge_parse_fallback_counts,
    reset_judge_parse_fallback_counts,
)
from docpilot.agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT_B
from docpilot.agent.types import Judgment  # noqa: F401  (re-export for tests/stubs)
from docpilot.core.models import Chunk, RetrieverResult
from docpilot.eval.triples import JudgeTriple, load_judge_triples
from docpilot.generation.generator import GroqGenerator

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "judge_triples.json"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Order in which per-category accuracy is reported.
VALID_CATEGORY_ORDER: tuple[str, ...] = (
    "sufficient-direct",
    "sufficient-partial",
    "insufficient-missing",
    "irrelevant",
    "adversarial",
)


# ---------------------------------------------------------------------------
# Core report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TripleEvalRow:
    """One triple's verdict under one prompt."""

    id: str
    category: str
    gold: str
    predicted: str
    correct: bool
    parse_fallback: bool


@dataclass
class PromptReport:
    """Calibration results for one judge prompt variant."""

    prompt_name: str
    results: list[TripleEvalRow]
    n: int = 0
    verdict_accuracy: float = 0.0
    adversarial_accuracy: float | None = None
    parse_failures: list[str] = field(default_factory=list)
    parse_failure_rate: float = 0.0
    per_category: dict[str, dict] = field(default_factory=dict)
    confusion: dict[str, dict] = field(default_factory=dict)


@dataclass
class ABReport:
    """Comparison of the two prompt variants over the same triples."""

    dataset_path: str
    triple_count: int
    generated_at: str
    prompts: dict[str, PromptReport]
    winner: str
    note: str


# ---------------------------------------------------------------------------
# Core evaluation logic (hermetic — no LLM calls here)
# ---------------------------------------------------------------------------


def triples_to_results(triple: JudgeTriple) -> list[RetrieverResult]:
    """Convert a triple's context chunks into judge-ready results."""
    results: list[RetrieverResult] = []
    for i, c in enumerate(triple.chunks):
        results.append(
            RetrieverResult(
                chunk=Chunk(
                    id=f"{triple.id}-{i}",
                    content=c.content,
                    heading_path=c.heading,
                    source_file=c.source_file,
                    chunk_index=i,
                ),
                score=c.score,
            )
        )
    return results


def evaluate_prompt(
    triples: list[JudgeTriple],
    judge,
    *,
    prompt_name: str,
    tools_available: bool = False,
) -> PromptReport:
    """Evaluate one judge over every triple.

    ``judge`` is any object exposing ``judge(question, results, query_used,
    *, tools_available) -> Judgment``.  Parse-fallback attribution is per
    triple: the module counter is snapshot before/after each call (the judge
    makes exactly one LLM call per invocation, SPEC §4.1).

    Raises:
        RuntimeError: If a triple's verdict is not in
            :data:`~docpilot.eval.triples.VALID_GOLDS` — the harness must
            never silently accept an unexpected verdict token.
    """
    reset_judge_parse_fallback_counts()
    rows: list[TripleEvalRow] = []
    parse_failures: list[str] = []

    for triple in triples:
        before = sum(judge_parse_fallback_counts().values())
        judgment = judge.judge(
            triple.question,
            triples_to_results(triple),
            triple.query_used,
            tools_available=tools_available,
        )
        after = sum(judge_parse_fallback_counts().values())
        predicted = judgment.verdict
        if predicted not in ("sufficient", "insufficient"):
            raise RuntimeError(
                f"triple {triple.id!r}: judge returned unexpected verdict "
                f"{predicted!r}"
            )
        fallback = after > before
        if fallback:
            parse_failures.append(triple.id)
        rows.append(
            TripleEvalRow(
                id=triple.id,
                category=triple.category,
                gold=triple.gold,
                predicted=predicted,
                correct=predicted == triple.gold,
                parse_fallback=fallback,
            )
        )

    n = len(rows)
    correct = sum(1 for r in rows if r.correct)
    per_category: dict[str, dict] = {}
    for cat in VALID_CATEGORY_ORDER:
        cat_rows = [r for r in rows if r.category == cat]
        if not cat_rows:
            continue
        cat_correct = sum(1 for r in cat_rows if r.correct)
        per_category[cat] = {
            "correct": cat_correct,
            "total": len(cat_rows),
            "accuracy": cat_correct / len(cat_rows),
        }
    adversarial_rows = [r for r in rows if r.category == "adversarial"]
    adversarial_accuracy = None
    if adversarial_rows:
        adv_correct = sum(1 for r in adversarial_rows if r.correct)
        adversarial_accuracy = adv_correct / len(adversarial_rows)

    confusion: dict[str, dict] = {}
    for gold in ("sufficient", "insufficient"):
        gold_rows = [r for r in rows if r.gold == gold]
        confusion[gold] = {
            "sufficient": sum(1 for r in gold_rows if r.predicted == "sufficient"),
            "insufficient": sum(1 for r in gold_rows if r.predicted == "insufficient"),
        }

    return PromptReport(
        prompt_name=prompt_name,
        results=rows,
        n=n,
        verdict_accuracy=correct / n if n else 0.0,
        adversarial_accuracy=adversarial_accuracy,
        parse_failures=parse_failures,
        parse_failure_rate=len(parse_failures) / n if n else 0.0,
        per_category=per_category,
        confusion=confusion,
    )


VALID_CATEGORY_ORDER: tuple[str, ...] = (
    "sufficient-direct",
    "sufficient-partial",
    "insufficient-missing",
    "irrelevant",
    "adversarial",
)


def build_ab_report(
    dataset_path: str | Path,
    report_a: PromptReport,
    report_b: PromptReport,
    *,
    note: str = "",
) -> ABReport:
    """Compare the two prompt reports and pick a *winner*.

    Tie-break order (all on calibration data alone):
      1. verdict accuracy (higher wins),
      2. parse-failure rate (lower wins),
      3. adversarial accuracy (higher wins).
    Any remaining tie → ``"tie"``.  The winner is a *reporting* outcome —
    whether to adopt prompt B is the client's call, informed by this data.
    """
    def _key(r: PromptReport) -> tuple[float, float, float]:
        adv = r.adversarial_accuracy if r.adversarial_accuracy is not None else -1.0
        # lower parse rate is better → negate for the max() comparison
        return (r.verdict_accuracy, -r.parse_failure_rate, adv)

    key_a = _key(report_a)
    key_b = _key(report_b)
    if key_a == key_b:
        winner = "tie"
    else:
        winner = "A" if key_a > key_b else "B"

    return ABReport(
        dataset_path=str(dataset_path),
        triple_count=report_a.n,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prompts={"A": report_a, "B": report_b},
        winner=winner,
        note=note,
    )


# ---------------------------------------------------------------------------
# Live runner (`python -m docpilot.eval` or `docpilot.eval.judge_ab`)
# ---------------------------------------------------------------------------


def ab_report_to_dict(report: ABReport) -> dict:
    """JSON-serializable view of an :class:`ABReport`."""
    return {
        "dataset_path": report.dataset_path,
        "triple_count": report.triple_count,
        "generated_at": report.generated_at,
        "winner": report.winner,
        "note": report.note,
        "prompts": {
            name: {
                "prompt_name": p.prompt_name,
                "n": p.n,
                "verdict_accuracy": round(p.verdict_accuracy, 4),
                "adversarial_accuracy": (
                    round(p.adversarial_accuracy, 4)
                    if p.adversarial_accuracy is not None
                    else None
                ),
                "parse_failure_rate": round(p.parse_failure_rate, 4),
                "parse_failures": p.parse_failures,
                "per_category": {
                    cat: {
                        "correct": d["correct"],
                        "total": d["total"],
                        "accuracy": round(d["accuracy"], 4),
                    }
                    for cat, d in p.per_category.items()
                },
                "confusion": p.confusion,
                "rows": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "gold": r.gold,
                        "predicted": r.predicted,
                        "correct": r.correct,
                        "parse_fallback": r.parse_fallback,
                    }
                    for r in p.results
                ],
            }
            for name, p in report.prompts.items()
        },
    }


def _load_judge_with_prompt(system_prompt: str) -> LLMSufficiencyJudge:
    """Production judge over Groq, parameterized with *system_prompt*."""
    model = config.AGENT_JUDGE_MODEL or config.GROQ_MODEL
    return LLMSufficiencyJudge(
        GroqGenerator(model=model),
        system_prompt=system_prompt,
    )


def run_judge_ab_live(
    triples_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """Run the full A/B against the Live LLM and write a JSON report.

    Returns the written report path.  Requires the Groq key visible to this
    process (e.g. export GITHUB… via an interactive bashrc source first).
    """
    triples_path = Path(triples_path) if triples_path else _DATASET_PATH
    triples = load_judge_triples(triples_path)
    logger.info(
        "Judge A/B: %d triples from %s", len(triples), triples_path
    )

    report_a = evaluate_prompt(
        triples,
        _load_judge_with_prompt(JUDGE_SYSTEM_PROMPT),
        prompt_name="A",
    )
    report_b = evaluate_prompt(
        triples,
        _load_judge_with_prompt(JUDGE_SYSTEM_PROMPT_B),
        prompt_name="B",
    )
    ab = build_ab_report(
        triples_path,
        report_a,
        report_b,
        note=(
            "First slice, SPEC §6.2: symmetric one-pass run (same model, "
            "temperature=0, same triples).  Adoption decision is the client's "
            "call on this data alone."
        ),
    )

    out_dir = Path(out_dir) if out_dir else _REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"judge_ab_{stamp}.json"
    out_path.write_text(
        json.dumps(ab_report_to_dict(ab), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(_format_ab(ab))
    logger.info("A/B report written to %s", out_path)
    return out_path


def _format_ab(ab: ABReport) -> str:
    """Human-readable console table for the A/B run."""
    lines = [
        f"Judge two-prompt A/B — dataset: {ab.dataset_path} "
        f"({ab.triple_count} triples)",
        f"winner: {ab.winner} (calibration-only outcome; adoption is the "
        "client's call)",
        "",
        f"{'metric':<22}{'A':>10}{'B':>10}",
        "-" * 42,
    ]
    a = ab.prompts["A"]
    b = ab.prompts["B"]
    rows = [
        ("verdict_accuracy", a.verdict_accuracy, b.verdict_accuracy),
        (
            "adversarial_accuracy",
            a.adversarial_accuracy if a.adversarial_accuracy is not None else float("nan"),
            b.adversarial_accuracy if b.adversarial_accuracy is not None else float("nan"),
        ),
        ("parse_failure_rate", a.parse_failure_rate, b.parse_failure_rate),
    ]
    for label, va, vb in rows:
        lines.append(f"{label:<22}{va:>10.3f}{vb:>10.3f}")
    lines.append("-" * 42)
    lines.append("Per-category accuracy (A / B):")
    for cat in VALID_CATEGORY_ORDER:
        ca = a.per_category.get(cat)
        cb = b.per_category.get(cat)
        if ca is None or cb is None:
            continue
        lines.append(
            f"  {cat:<22}{ca['accuracy']:>8.3f} / {cb['accuracy']:<8.3f}"
            f" (n={ca['total']})"
        )
    lines.append("")
    lines.append("Confusion (gold → predicted): A / B")
    for gold in ("sufficient", "insufficient"):
        ca = a.confusion[gold]
        cb = b.confusion[gold]
        lines.append(
            f"  {gold:<12} sufficient: {ca['sufficient']}/{cb['sufficient']}  "
            f"insufficient: {ca['insufficient']}/{cb['insufficient']}"
        )
    lines.append("")
    for name in ("A", "B"):
        p = ab.prompts[name]
        if p.parse_failures:
            lines.append(
                f"  prompt {name} parse failures ({len(p.parse_failures)}): "
                f"{','.join(p.parse_failures)}"
            )
        else:
            lines.append(f"  prompt {name} parse failures: none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m docpilot.eval [--triples PATH] [--out DIR]``."""
    args = list(argv) if argv is not None else sys.argv[1:]
    triples_path: str | Path | None = None
    out_dir: str | Path | None = None
    i = 0
    while i < len(args):
        if args[i] == "--triples":
            i += 1
            if i >= len(args):
                print("--triples requires a path", file=sys.stderr)
                return 2
            triples_path = args[i]
        elif args[i] == "--out":
            i += 1
            if i >= len(args):
                print("--out requires a directory", file=sys.stderr)
                return 2
            out_dir = args[i]
        else:
            print(f"unknown argument {args[i]!r}", file=sys.stderr)
            return 2
        i += 1
    run_judge_ab_live(triples_path, out_dir=out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())