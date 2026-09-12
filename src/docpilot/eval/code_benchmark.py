"""Phase 6 code benchmark — validation-pass + citation accuracy (PLAN §7.2 T5).

Runs the **same** duck-typed harness recipe as the Phase 4 benchmark but over
a code-focused dataset (``dataset/code_benchmark.json``): each row is a code
request with gold signals for the §7.2 T5 semantics —

  * **compiles / imports cleanly** — every emitted block passes a hermetic
    ``compile()`` and the T1 validator's imports check is ``PASS`` (imports
    grounded in the retrieved evidence);
  * **symbol usage matches retrieved API** — the validator's symbols check is
    ``PASS`` and the gold surface identifiers (the API symbols the request
    must exercise) actually appear in the generated code;
  * **citations resolve to the right doc pages** — body citation markers
    resolve to sources inside ``gold_sources``, using exactly the same
    marker→source logic as the main benchmark's ``citation_gold_accuracy``
    (so the code run's number is directly comparable to the classic/agentic
    baseline where overlap exists, §7.5).

Metric set (report): validation-pass rate (fully validated, every check PASS),
compile rate, import-grounding rate, symbol-grounding rate, gold-surface usage
(fraction of expected identifiers exercised), gold-import usage rate,
citation-gold accuracy, refusal accuracy ("I don't know" on uncovered
requests), and avg validation turns + latency (the T4 loop cost).

The runner is duck-typed — ``run(request: str) -> CodeRunOutput`` — so
hermetic tests script every metric without a database or LLM, exactly like
:mod:`docpilot.eval.benchmark`.  Row checkpoints (jsonl) and daily-TPD abort
handling reuse the Phase 4 machinery (:func:`_load_done_ids`,
:class:`QuotaExhausted`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from docpilot.eval.benchmark import (
    QuotaExhausted,
    answer_body,
    body_markers,
    is_tpd_exhaustion,
)
from docpilot.validation.verdict import (
    CheckStatus,
    ValidationVerdict,
)

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "code_benchmark.json"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"

_CONFIG_KEYS: tuple[str, ...] = (
    "id",
    "request",
    "gold_imports",
    "gold_surface",
    "gold_sources",
    "refusal_expected",
)


@dataclass(frozen=True)
class CodeBenchmarkQuestion:
    """One gold-labeled code request (PLAN §7.2 T5).

    Attributes:
        id: Unique row id.
        request: The user's code request (what the pipeline receives).
        gold_imports: Import names the generated code must use (and that the
            validator must ground in the retrieved evidence).
        gold_surface: API identifiers the code must exercise — the symbols
            from the retrieved docs (e.g. ``FastAPI``, ``Query``).
        gold_sources: Doc pages the citations must resolve to.
        refusal_expected: True for requests the corpus does *not* cover — the
            model must answer with the SPEC §3.9 refusal, not fabricate code.
        note: Human note (optional).
    """

    id: str
    request: str
    gold_imports: tuple[str, ...]
    gold_surface: tuple[str, ...]
    gold_sources: tuple[str, ...]
    refusal_expected: bool = False
    note: str = ""


@dataclass
class CodeRunOutput:
    """What the code route produced for one request (duck-typed runner).

    Mirrors :class:`~docpilot.eval.benchmark.RunOutput` but carries the code
    route's verification payload so scoring never has to guess.
    """

    question: str
    answer: str
    source_files: list[str]
    code_blocks: list[str]
    block_verdicts: list[ValidationVerdict]
    verdict: ValidationVerdict | None
    validation_failed: bool
    refused: bool
    validation_turns: int = 1
    latency_ms: float = 0.0


@dataclass
class CodeBenchmarkRow:
    """One code request scored."""

    id: str
    refused: bool
    answer: str
    source_files: list[str]
    validation_pass: bool | None
    compiles: bool | None
    imports_grounded: bool | None
    surface_grounded: bool | None
    used_gold_surface: float | None
    used_gold_imports: bool | None
    citation_gold_accuracy: float | None
    marker_count: int
    validation_turns: int
    latency_ms: float
    refusal_ok: bool | None


@dataclass
class CodePipelineMetrics:
    """Aggregated code metrics over the dataset."""

    validation_pass_rate: float | None
    validation_n: int
    compiles_rate: float | None
    imports_grounded_rate: float | None
    surface_grounded_rate: float | None
    gold_surface_usage: float | None
    gold_import_usage_rate: float | None
    citation_gold_accuracy: float | None
    citation_n: int
    refusal_accuracy: float | None
    refusal_n: int
    avg_validation_turns: float
    avg_latency_ms: float


@dataclass
class CodePipelineReport:
    """The full scored code run (serialized to the reports dir)."""

    dataset_path: str
    generated_at: str
    rows: list[CodeBenchmarkRow]
    metrics: CodePipelineMetrics
    note: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_code_benchmark(path: str | Path) -> list[CodeBenchmarkQuestion]:
    """Load and schema-validate the code benchmark dataset.

    Raises:
        ValueError: On any schema violation — the dataset is the contract for
            the code eval run, so bad rows fail loudly.
        FileNotFoundError: When *path* does not exist.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("questions")
        if not isinstance(rows, list):
            raise ValueError("code dataset must be a JSON array or {questions: [...]}")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("code dataset must be a JSON array or {questions: [...]}")

    questions: list[CodeBenchmarkQuestion] = []
    for i, raw in enumerate(rows):
        missing = [k for k in _CONFIG_KEYS if k not in raw]
        if missing:
            raise ValueError(f"code_benchmark[{i}]: missing key(s) {missing}")
        qid = raw["id"]
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"code_benchmark[{i}]: 'id' must be non-empty")
        if not isinstance(raw["request"], str) or not raw["request"].strip():
            raise ValueError(f"code_benchmark[{i}] ({qid!r}): 'request' must be non-empty")

        def _str_list(value, key: str) -> tuple[str, ...]:
            if not isinstance(value, list) or any(
                not isinstance(v, str) or not v.strip() for v in value
            ):
                raise ValueError(
                    f"code_benchmark[{i}] ({qid!r}): '{key}' must be a list "
                    "of non-empty strings"
                )
            return tuple(value)

        imports = _str_list(raw["gold_imports"], "gold_imports")
        surface = _str_list(raw["gold_surface"], "gold_surface")
        sources = _str_list(raw["gold_sources"], "gold_sources")
        refusal = raw["refusal_expected"]
        if not isinstance(refusal, bool):
            raise ValueError(
                f"code_benchmark[{i}] ({qid!r}): 'refusal_expected' must be a bool"
            )

        if refusal:
            # Uncovered requests carry no gold code signals — the model must
            # refuse, not fabricate (§7.5).
            if imports or surface or sources:
                raise ValueError(
                    f"code_benchmark[{i}] ({qid!r}): a refusal_expected row "
                    "must have empty gold_imports/gold_surface/gold_sources"
                )
        elif not sources:
            raise ValueError(
                f"code_benchmark[{i}] ({qid!r}): a code-answerable row must "
                "have non-empty gold_sources"
            )

        questions.append(
            CodeBenchmarkQuestion(
                id=qid,
                request=raw["request"],
                gold_imports=imports,
                gold_surface=surface,
                gold_sources=sources,
                refusal_expected=refusal,
                note=str(raw.get("note", "") or ""),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _fully_validated(verdict: ValidationVerdict | None) -> bool:
    """True when the verdict proves the code valid — every check PASS.

    Identical discipline to the T4 loop (agent/code_route): a passed verdict
    with SKIP checks is *not* validated.
    """
    if verdict is None:
        return False
    return verdict.passed and all(
        check.status is CheckStatus.PASS for check in verdict.checks
    )


def _compiles_cleanly(blocks: list[str]) -> bool:
    """Hermetic syntax check over every emitted block (no execution)."""
    try:
        for block in blocks:
            compile(block, "<codegen-eval>", "exec")
    except SyntaxError:
        return False
    return True


def _check_pass(verdicts: list[ValidationVerdict], check_name: str) -> bool:
    """True when *check_name* is PASS on every block verdict."""
    if not verdicts:
        return False
    return all(
        any(c.name == check_name and c.status is CheckStatus.PASS for c in v.checks)
        for v in verdicts
    )


def _word_identifiers(text: str) -> set[str]:
    """Word-boundary identifiers inside *text* (used for the gold signals)."""
    import re

    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))


def _score_row(
    q: CodeBenchmarkQuestion,
    out: CodeRunOutput,
    block_text: str,
) -> CodeBenchmarkRow:
    """Score one code request against its gold signals."""
    blocks = out.code_blocks
    verdicts = out.block_verdicts

    # Code-focused metrics only exist when the model emitted code.
    validation_pass: bool | None = None
    compiles: bool | None = None
    imports_grounded: bool | None = None
    surface_grounded: bool | None = None
    used_gold_surface: float | None = None
    used_gold_imports: bool | None = None
    if blocks:
        validation_pass = _fully_validated(out.verdict)
        compiles = _compiles_cleanly(blocks)
        imports_grounded = _check_pass(verdicts, "imports_evidence")
        surface_grounded = _check_pass(verdicts, "symbols_evidence")
        identifiers = _word_identifiers(block_text)
        if q.gold_surface:
            used_gold_surface = sum(1 for s in q.gold_surface if s in identifiers) / len(
                q.gold_surface
            )
        if q.gold_imports:
            used_gold_imports = all(g in identifiers for g in q.gold_imports)

    # Citations resolve to the right doc pages — same marker logic as the
    # main benchmark's citation_gold_accuracy (comparable over overlap, §7.5).
    markers = body_markers(answer_body(out.answer))
    citation_gold = None
    if markers and q.gold_sources:
        citation_gold = (
            sum(
                1
                for m in markers
                if 1 <= m <= len(out.source_files)
                and out.source_files[m - 1] in q.gold_sources
            )
            / len(markers)
        )

    refusal_ok = None
    if q.refusal_expected:
        refusal_ok = out.refused
    else:
        refusal_ok = not out.refused

    return CodeBenchmarkRow(
        id=q.id,
        refused=out.refused,
        answer=out.answer,
        source_files=list(out.source_files),
        validation_pass=validation_pass,
        compiles=compiles,
        imports_grounded=imports_grounded,
        surface_grounded=surface_grounded,
        used_gold_surface=used_gold_surface,
        used_gold_imports=used_gold_imports,
        citation_gold_accuracy=citation_gold,
        marker_count=len(markers),
        validation_turns=out.validation_turns,
        latency_ms=out.latency_ms,
        refusal_ok=refusal_ok,
    )


def run_code_benchmark(
    questions: list[CodeBenchmarkQuestion],
    run,
    *,
    generated_at: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> CodePipelineReport:
    """Score one code run over the dataset.

    ``run`` is any callable ``(request: str) -> CodeRunOutput``.  Rows are
    checkpointed to a jsonl file as they finish and ids already present are
    skipped on re-entry — a mid-run daily-TPD abort never re-burns quota on
    scored requests (same recipe as :func:`docpilot.eval.benchmark.run_pipeline`).
    """
    done: set[str] = set()
    rows: list[CodeBenchmarkRow] = []
    if checkpoint_path is not None:
        cpath = Path(checkpoint_path)
        if cpath.exists():
            for line in cpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            # Resumed runs report over the *full* dataset: reconstruct the
            # already-scored rows (new rows join, not replace).
            for line in cpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(CodeBenchmarkRow(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            if done:
                logger.warning(
                    "Code benchmark: %d row(s) already checkpointed — skipping %s",
                    len(done), sorted(done)[:8],
                )
        cpath.parent.mkdir(parents=True, exist_ok=True)
    for q in questions:
        if q.id in done:
            continue
        try:
            out = run(q.request)
        except Exception as exc:
            if is_tpd_exhaustion(exc):
                raise QuotaExhausted(str(exc)) from exc
            raise
        if not isinstance(out, CodeRunOutput):
            raise TypeError(
                f"code runner must return CodeRunOutput; got {type(out).__name__}"
            )
        block_text = "\n".join(out.code_blocks)
        row = _score_row(q, out, block_text)
        rows.append(row)
        if checkpoint_path is not None:
            with open(checkpoint_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_row_to_dict(row)) + "\n")

    return _aggregate(questions, rows, generated_at=generated_at)


def _aggregate(
    questions: list[CodeBenchmarkQuestion],
    rows: list[CodeBenchmarkRow],
    *,
    generated_at: str | None,
) -> CodePipelineReport:
    code_rows = [r for r in rows if r.compiles is not None]  # emitted code
    def _avg(attr: str, subset: list[CodeBenchmarkRow]) -> float | None:
        vals = [getattr(r, attr) for r in subset]
        if not vals or all(v is None for v in vals):
            return None
        present = [v for v in vals if v is not None]
        return sum(present) / len(present) if present else None

    import_grounded = [r for r in code_rows if r.imports_grounded is not None]
    surface_grounded = [r for r in code_rows if r.surface_grounded is not None]
    surface_usage = [r for r in code_rows if r.used_gold_surface is not None]
    import_usage = [r for r in code_rows if r.used_gold_imports is not None]
    citation = [r for r in rows if r.citation_gold_accuracy is not None]
    refusal = [r for r in rows if r.refusal_ok is not None]

    metrics = CodePipelineMetrics(
        validation_pass_rate=_avg("validation_pass", code_rows),
        validation_n=len(code_rows),
        compiles_rate=_avg("compiles", code_rows),
        imports_grounded_rate=_avg("imports_grounded", import_grounded),
        surface_grounded_rate=_avg("surface_grounded", surface_grounded),
        gold_surface_usage=_avg("used_gold_surface", surface_usage),
        gold_import_usage_rate=_avg("used_gold_imports", import_usage),
        citation_gold_accuracy=_avg("citation_gold_accuracy", citation),
        citation_n=len(citation),
        refusal_accuracy=(
            sum(1 for r in refusal if r.refusal_ok) / len(refusal) if refusal else None
        ),
        refusal_n=len(refusal),
        avg_validation_turns=(
            sum(r.validation_turns for r in rows) / len(rows) if rows else 0.0
        ),
        avg_latency_ms=sum(r.latency_ms for r in rows) / len(rows) if rows else 0.0,
    )

    return CodePipelineReport(
        dataset_path="-",
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        rows=rows,
        metrics=metrics,
        note=(
            f"code eval over {len(rows)} request(s); citation_gold_accuracy uses "
            "the main benchmark's marker→source logic (comparable over overlap)"
        ),
    )


# ---------------------------------------------------------------------------
# Serialization / formatting
# ---------------------------------------------------------------------------


def _row_to_dict(r: CodeBenchmarkRow) -> dict:
    return {
        "id": r.id,
        "refused": r.refused,
        "validation_pass": r.validation_pass,
        "compiles": r.compiles,
        "imports_grounded": r.imports_grounded,
        "surface_grounded": r.surface_grounded,
        "used_gold_surface": None if r.used_gold_surface is None else round(r.used_gold_surface, 4),
        "used_gold_imports": r.used_gold_imports,
        "citation_gold_accuracy": (
            None if r.citation_gold_accuracy is None else round(r.citation_gold_accuracy, 4)
        ),
        "marker_count": r.marker_count,
        "validation_turns": r.validation_turns,
        "latency_ms": round(r.latency_ms, 1),
        "refusal_ok": r.refusal_ok,
        "answer": r.answer,
        "source_files": r.source_files,
    }


def _metrics_to_dict(m: CodePipelineMetrics) -> dict:
    def _r(v: float | None) -> float | None:
        return None if v is None else round(v, 4)

    return {
        "validation_pass_rate": _r(m.validation_pass_rate),
        "validation_n": m.validation_n,
        "compiles_rate": _r(m.compiles_rate),
        "imports_grounded_rate": _r(m.imports_grounded_rate),
        "surface_grounded_rate": _r(m.surface_grounded_rate),
        "gold_surface_usage": _r(m.gold_surface_usage),
        "gold_import_usage_rate": _r(m.gold_import_usage_rate),
        "citation_gold_accuracy": _r(m.citation_gold_accuracy),
        "citation_n": m.citation_n,
        "refusal_accuracy": _r(m.refusal_accuracy),
        "refusal_n": m.refusal_n,
        "avg_validation_turns": round(m.avg_validation_turns, 3),
        "avg_latency_ms": round(m.avg_latency_ms, 1),
    }


def report_to_dict(report: CodePipelineReport) -> dict:
    return {
        "pipeline": "code",
        "dataset_path": report.dataset_path,
        "generated_at": report.generated_at,
        "note": report.note,
        "metrics": _metrics_to_dict(report.metrics),
        "rows": [_row_to_dict(r) for r in report.rows],
    }


def format_metrics(report: CodePipelineReport) -> str:
    m = report.metrics
    lines = [f"Code benchmark — {len(report.rows)} request(s)"]
    lines.append(
        f"  validation-pass      {_fmt(m.validation_pass_rate)} ({m.validation_n} code rows)"
    )
    lines.append(f"  compiles-cleanly     {_fmt(m.compiles_rate)}")
    lines.append(f"  imports grounded     {_fmt(m.imports_grounded_rate)}")
    lines.append(f"  symbols grounded     {_fmt(m.surface_grounded_rate)}")
    lines.append(f"  gold surface used    {_fmt(m.gold_surface_usage)}")
    lines.append(f"  gold imports used    {_fmt(m.gold_import_usage_rate)}")
    lines.append(
        f"  citation gold acc    {_fmt(m.citation_gold_accuracy)} ({m.citation_n} rows w/ markers)"
    )
    lines.append(f"  refusal accuracy     {_fmt(m.refusal_accuracy)} ({m.refusal_n} refusal rows)")
    lines.append(f"  avg validation turns {m.avg_validation_turns:.2f}")
    lines.append(f"  avg latency ms       {m.avg_latency_ms:.1f}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Live run entry (CLI)
# ---------------------------------------------------------------------------


def _code_run(question: str) -> CodeRunOutput:
    """Production runner: force the opt-in code route on the live stack."""
    from docpilot.agent.code_route import run_code_route

    result = run_code_route(question, explicit_code=True)
    request = result.code
    if request is None:
        # Should not happen (explicit opt-in always takes the code route),
        # but never fabricate a pass row.
        return CodeRunOutput(
            question=question,
            answer="",
            source_files=[],
            code_blocks=[],
            block_verdicts=[],
            verdict=None,
            validation_failed=True,
            refused=True,
        )
    return CodeRunOutput(
        question=question,
        answer=request.display,
        source_files=[s.file for s in request.sources],
        code_blocks=list(request.code_blocks),
        block_verdicts=list(request.block_verdicts),
        verdict=request.verdict,
        validation_failed=request.validation_failed,
        refused=request.refused,
        validation_turns=request.generation_attempts,
        latency_ms=request.latency_ms,
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m docpilot.eval code-benchmark [--stamp STAMP] [--out DIR]``."""
    import argparse

    parser = argparse.ArgumentParser(prog="docpilot.eval code-benchmark")
    parser.add_argument("--dataset", default=None, help="Code dataset path (default: committed).")
    parser.add_argument("--out", default=None, help="Reports directory (default: package reports).")
    parser.add_argument("--stamp", default=None, help="Run stamp (default: now).")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset) if args.dataset else _DATASET_PATH
    out_dir = Path(args.out) if args.out else _REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or time.strftime("%Y%m%d_%H%M%S")

    questions = load_code_benchmark(dataset_path)
    logger.info("Code benchmark: %d request(s) from %s", len(questions), dataset_path)

    checkpoint = out_dir / f"code_benchmark_{stamp}.rows.jsonl"
    try:
        report = run_code_benchmark(questions, _code_run, checkpoint_path=checkpoint)
    except QuotaExhausted as exc:
        print(
            "\n⛔ Daily token budget exhausted (TPD) — nothing fabricated; scored "
            "rows are checkpointed. Resume when the bucket refills with: "
            f"--stamp {stamp}"
        )
        logger.error("Code benchmark aborted on TPD: %s", exc)
        return 3
    report.dataset_path = str(dataset_path)
    out_path = out_dir / f"code_benchmark_{stamp}.json"
    out_path.write_text(
        json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8"
    )
    checkpoint.unlink(missing_ok=True)
    print(format_metrics(report))
    logger.info("Code benchmark report written to %s", out_path)
    return 0