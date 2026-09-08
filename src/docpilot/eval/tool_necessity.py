"""Tool-necessity tri-class evaluation (SPEC §6.2 / §6.4).

Phase 3's necessity measurement: does the judge's ``needs_tool`` signal fire
when the docs demonstrably cannot answer, and stay quiet when they can?

Gold classes:
  - ``docs-answerable``        docs alone answer → expect verdict sufficient,
                               needs_tool false (a fire here is a *false
                               positive*);
  - ``live-state-answerable``  only GitHub can answer (issue/repo/commit
                               state) → expect verdict insufficient,
                               needs_tool true (a non-fire here is a *false
                               negative*), with the expected tool action;
  - ``neither``                no source suffices → expect verdict
                               insufficient, needs_tool false (fire = wrong).

Reports the SPEC §6.4 tool-necessity rates: false-positive and
false-negative rate on the tri-class set, verdict accuracy,
tool-request validity, and expected-action match rate.
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
    judge_parse_fallback_counts,
    reset_judge_parse_fallback_counts,
)
from docpilot.eval.judge_ab import triples_to_results
from docpilot.eval.triples import TripleChunk
from docpilot.generation.generator import GroqGenerator

logger = logging.getLogger(__name__)

VALID_CLASSES: frozenset[str] = frozenset(
    {"docs-answerable", "live-state-answerable", "neither"}
)
VALID_ACTIONS: frozenset[str] = frozenset(
    {"github.search_issues", "github.list_issues", "github.get_commits"}
)

_DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "tool_necessity.json"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass(frozen=True)
class ToolNecessityTriple:
    """A gold-labeled tri-class question."""

    id: str
    cls: str
    question: str
    query_used: str
    gold_verdict: str
    gold_needs_tool: bool
    gold_tool_action: str | None
    chunks: tuple[TripleChunk, ...]
    note: str = ""


@dataclass(frozen=True)
class ToolNecessityRow:
    """One triple's judge decision under tools_available=True."""

    id: str
    cls: str
    gold_verdict: str
    predicted_verdict: str
    correct_verdict: bool
    gold_needs_tool: bool
    predicted_needs_tool: bool
    gold_tool_action: str | None
    predicted_action: str | None
    tool_request_valid: bool
    tool_action_matches: bool
    parse_fallback: bool


@dataclass
class ToolNecessityReport:
    """§6.4 tool-necessity metrics over the tri-class set."""

    dataset_path: str
    generated_at: str
    n: int
    verdict_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    neither_fire_rate: float
    tool_request_valid_rate: float
    tool_action_match_rate: float
    per_class: dict[str, dict] = field(default_factory=dict)
    rows: list[ToolNecessityRow] = field(default_factory=list)
    note: str = ""


def load_tool_necessity_triples(path: str | Path) -> list[ToolNecessityTriple]:
    """Load and schema-validate the tri-class dataset.

    Raises:
        ValueError: On any schema violation — the dataset is the contract.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("triples")
        if not isinstance(rows, list):
            raise ValueError("dataset must be a JSON array or {triples: [...]}")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("dataset must be a JSON array or {triples: [...]}")

    triples: list[ToolNecessityTriple] = []
    for i, raw in enumerate(rows):
        tid = raw.get("id")
        if not isinstance(tid, str) or not tid.strip():
            raise ValueError(f"tool_necessity[{i}]: 'id' must be non-empty")
        cls = raw.get("class")
        if cls not in VALID_CLASSES:
            raise ValueError(
                f"tool_necessity[{i}] ({tid!r}): bad class {cls!r}; "
                f"expected one of {sorted(VALID_CLASSES)}"
            )
        for key in ("question", "query_used"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(
                    f"tool_necessity[{i}] ({tid!r}): '{key}' must be non-empty"
                )
        gold_verdict = raw.get("gold_verdict")
        if gold_verdict not in ("sufficient", "insufficient"):
            raise ValueError(
                f"tool_necessity[{i}] ({tid!r}): bad gold_verdict {gold_verdict!r}"
            )
        gold_needs_tool = raw.get("gold_needs_tool")
        if not isinstance(gold_needs_tool, bool):
            raise ValueError(
                f"tool_necessity[{i}] ({tid!r}): 'gold_needs_tool' must be a bool"
            )
        action = raw.get("gold_tool_action")
        if action is not None and action not in VALID_ACTIONS:
            raise ValueError(
                f"tool_necessity[{i}] ({tid!r}): bad gold_tool_action {action!r}"
            )
        chunks_raw = raw.get("chunks")
        if not isinstance(chunks_raw, list) or not chunks_raw:
            raise ValueError(
                f"tool_necessity[{i}] ({tid!r}): 'chunks' must be non-empty"
            )
        chunks: list[TripleChunk] = []
        for c_i, c in enumerate(chunks_raw):
            content = c.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"tool_necessity[{i}] ({tid!r}).chunks[{c_i}]: "
                    "'content' must be non-empty"
                )
            source = c.get("source_file", "")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"tool_necessity[{i}] ({tid!r}).chunks[{c_i}]: "
                    "'source_file' must be non-empty"
                )
            chunks.append(
                TripleChunk(
                    content=content,
                    source_file=source,
                    heading=str(c.get("heading", "") or ""),
                    score=float(c.get("score", 0.8)),
                )
            )
        triples.append(
            ToolNecessityTriple(
                id=tid,
                cls=cls,
                question=raw["question"],
                query_used=raw["query_used"],
                gold_verdict=gold_verdict,
                gold_needs_tool=gold_needs_tool,
                gold_tool_action=action,
                chunks=tuple(chunks),
                note=str(raw.get("note", "") or ""),
            )
        )
    return triples


def _judge_for_triple(triple: ToolNecessityTriple):
    """Build the judge inputs for a tri-class triple (reuses chunk plumbing)."""
    return triples_to_results(triple)


def evaluate_tool_necessity(
    triples: list[ToolNecessityTriple],
    judge,
) -> ToolNecessityReport:
    """Score one judge over the full tri-class set (tools_available=True).

    ``judge`` is any object exposing ``judge(question, results, query_used,
    *, tools_available) -> Judgment`` (duck-typed — StubJudge in tests).

    Raises:
        RuntimeError: On an unexpected verdict token.
    """
    reset_judge_parse_fallback_counts()
    rows: list[ToolNecessityRow] = []

    for triple in triples:
        before = sum(judge_parse_fallback_counts().values())
        judgment = judge.judge(
            triple.question,
            _judge_for_triple(triple),
            triple.query_used,
            tools_available=True,
        )
        after = sum(judge_parse_fallback_counts().values())

        predicted_verdict = judgment.verdict
        if predicted_verdict not in ("sufficient", "insufficient"):
            raise RuntimeError(
                f"triple {triple.id!r}: judge returned unexpected verdict "
                f"{predicted_verdict!r}"
            )
        predicted_needs_tool = bool(judgment.needs_tool)
        tool_request = judgment.tool_request
        tool_request_valid = predicted_needs_tool and isinstance(tool_request, dict)
        action = None
        matches = False
        if tool_request_valid:
            action = tool_request.get("name")
            matches = action == triple.gold_tool_action

        rows.append(
            ToolNecessityRow(
                id=triple.id,
                cls=triple.cls,
                gold_verdict=triple.gold_verdict,
                predicted_verdict=predicted_verdict,
                correct_verdict=predicted_verdict == triple.gold_verdict,
                gold_needs_tool=triple.gold_needs_tool,
                predicted_needs_tool=predicted_needs_tool,
                gold_tool_action=triple.gold_tool_action,
                predicted_action=action if isinstance(action, str) else None,
                tool_request_valid=tool_request_valid,
                tool_action_matches=matches,
                parse_fallback=after > before,
            )
        )

    n = len(rows)
    docs_rows = [r for r in rows if r.cls == "docs-answerable"]
    live_rows = [r for r in rows if r.cls == "live-state-answerable"]
    neither_rows = [r for r in rows if r.cls == "neither"]

    false_positives = sum(1 for r in docs_rows if r.predicted_needs_tool)
    false_negatives = sum(1 for r in live_rows if not r.predicted_needs_tool)
    neither_fires = sum(1 for r in neither_rows if r.predicted_needs_tool)

    fired = [r for r in rows if r.predicted_needs_tool]
    valid_requests = sum(1 for r in fired if r.tool_request_valid)
    action_matches = sum(1 for r in fired if r.tool_action_matches)

    per_class: dict[str, dict] = {}
    for cls in ("docs-answerable", "live-state-answerable", "neither"):
        cls_rows = [r for r in rows if r.cls == cls]
        if not cls_rows:
            continue
        correct = sum(1 for r in cls_rows if r.correct_verdict)
        fires = sum(1 for r in cls_rows if r.predicted_needs_tool)
        correct_fires = sum(
            1
            for r in cls_rows
            if r.predicted_needs_tool == r.gold_needs_tool
        )
        per_class[cls] = {
            "total": len(cls_rows),
            "verdict_accuracy": correct / len(cls_rows),
            "needs_tool_accuracy": correct_fires / len(cls_rows),
            "fires": fires,
        }

    return ToolNecessityReport(
        dataset_path="-",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n=n,
        verdict_accuracy=(
            sum(1 for r in rows if r.correct_verdict) / n if n else 0.0
        ),
        false_positive_rate=(false_positives / len(docs_rows)) if docs_rows else 0.0,
        false_negative_rate=(false_negatives / len(live_rows)) if live_rows else 0.0,
        neither_fire_rate=(neither_fires / len(neither_rows)) if neither_rows else 0.0,
        tool_request_valid_rate=(valid_requests / len(fired)) if fired else 1.0,
        tool_action_match_rate=(action_matches / len(fired)) if fired else 1.0,
        per_class=per_class,
        rows=rows,
    )


def report_to_dict(report: ToolNecessityReport) -> dict:
    """JSON-serializable view."""
    return {
        "dataset_path": report.dataset_path,
        "generated_at": report.generated_at,
        "n": report.n,
        "verdict_accuracy": round(report.verdict_accuracy, 4),
        "false_positive_rate": round(report.false_positive_rate, 4),
        "false_negative_rate": round(report.false_negative_rate, 4),
        "neither_fire_rate": round(report.neither_fire_rate, 4),
        "tool_request_valid_rate": round(report.tool_request_valid_rate, 4),
        "tool_action_match_rate": round(report.tool_action_match_rate, 4),
        "per_class": report.per_class,
        "note": report.note,
        "rows": [
            {
                "id": r.id,
                "class": r.cls,
                "gold_verdict": r.gold_verdict,
                "predicted_verdict": r.predicted_verdict,
                "correct_verdict": r.correct_verdict,
                "gold_needs_tool": r.gold_needs_tool,
                "predicted_needs_tool": r.predicted_needs_tool,
                "gold_tool_action": r.gold_tool_action,
                "predicted_action": r.predicted_action,
                "tool_request_valid": r.tool_request_valid,
                "tool_action_matches": r.tool_action_matches,
                "parse_fallback": r.parse_fallback,
            }
            for r in report.rows
        ],
    }


def format_report(report: ToolNecessityReport) -> str:
    """Human-readable console table."""
    metrics: list[tuple[str, float]] = [
        ("verdict_accuracy", report.verdict_accuracy),
        ("false_positive_rate (docs-answerable → tool fired)", report.false_positive_rate),
        ("false_negative_rate (live-state → no tool)", report.false_negative_rate),
        ("neither_fire_rate", report.neither_fire_rate),
        ("tool_request_valid_rate", report.tool_request_valid_rate),
        ("tool_action_match_rate", report.tool_action_match_rate),
    ]
    lines = [
        f"Tool-necessity tri-class — {report.n} questions",
        "-" * 56,
        f"{'metric':<28}{'value':>10}",
        "-" * 56,
    ]
    lines += [f"{label:<28}{value:>10.3f}" for label, value in metrics]
    lines.append("-" * 56)
    lines.append("Per-class (verdict / needs_tool accuracy, fires):")
    for cls, d in report.per_class.items():
        lines.append(
            f"  {cls:<20} verdict={d['verdict_accuracy']:.3f} "
            f"needs_tool={d['needs_tool_accuracy']:.3f} fires={d['fires']}/{d['total']}"
        )
    lines.append("")
    for r in report.rows:
        if not (r.correct_verdict and r.predicted_needs_tool == r.gold_needs_tool):
            flags = []
            if not r.correct_verdict:
                flags.append(f"verdict {r.gold_verdict}≠{r.predicted_verdict}")
            if r.predicted_needs_tool != r.gold_needs_tool:
                flags.append(f"needs_tool {r.gold_needs_tool}≠{r.predicted_needs_tool}")
            if r.predicted_needs_tool and not r.tool_action_matches:
                flags.append(f"action {r.predicted_action}≠{r.gold_tool_action}")
            lines.append(f"  ✗ {r.id} [{r.cls}]: {'; '.join(flags)}")
        elif r.parse_fallback:
            lines.append(f"  ⚠ {r.id}: parse fallback")
    return "\n".join(lines)


def run_tool_necessity_live(
    triples_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """Run the tri-class evaluation against the production judge (Groq).

    Uses prompt A (the production prompt) with the judge's default model.
    """
    triples_path = Path(triples_path) if triples_path else _DATASET_PATH
    triples = load_tool_necessity_triples(triples_path)
    logger.info("Tool-necessity: %d tri-class questions from %s", len(triples), triples_path)

    model = config.AGENT_JUDGE_MODEL or config.GROQ_MODEL
    judge = LLMSufficiencyJudge(GroqGenerator(model=model))
    report = evaluate_tool_necessity(triples, judge)
    report.dataset_path = str(triples_path)
    report.note = (
        "Production judge (prompt A), Groq, temperature 0, tools_available=True. "
        "SPEC §6.2 tri-class necessity measurement — false positive = tool fired "
        "on docs-answerable; false negative = no tool on live-state-answerable."
    )

    out_dir = Path(out_dir) if out_dir else _REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tool_necessity_{stamp}.json"
    out_path.write_text(
        json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8"
    )
    print(format_report(report))
    logger.info("Tool-necessity report written to %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m docpilot.eval tool-necessity [--triples PATH] [--out DIR]``."""
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
    run_tool_necessity_live(triples_path, out_dir=out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())