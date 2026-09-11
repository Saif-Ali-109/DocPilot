"""Phase 4 benchmark — classic-vs-agentic comparison (SPEC §6.1).

Runs the **same** question set through both pipelines:

  * **classic** — Phase 1 :func:`docpilot.pipeline_ask.ask` fast path
    (retrieve → generate → cite, no agent loop, no tool);
  * **agentic** — Phase 2 :func:`docpilot.agent.pipeline_agentic.agentic_ask`
    with ``strategy="agentic"`` (forced loop, judge + reformulate + Phase 3
    GitHub tool when a PAT exists).

Metrics per question (categories: ``docs-answerable`` / ``live-state-answerable``
/ ``neither``):

  * answer correctness — docs: fraction of gold key facts (short
    paraphrase-robust fragments) contained in the answer body; live: answered
    **and** the tool fired (0.5 = answered without the tool, 0.0 = refused);
    neither: 1.0 iff refused (this is the "I don't know" accuracy);
  * retrieval recall@k — any gold source file present in the offered sources
    (docs-only, where gold sources exist);
  * citation validity — body citation markers that resolve inside the offered
    sources; citation gold accuracy — markers resolving to a gold source
    (docs-only);
  * groundedness — external NLI-style audit of the answer against its sources
    (injectable checker; no LLM call in hermetic tests);
  * latency, retrieval-call count, tool-call count.

The runner is duck-typed: ``run(question) -> RunOutput``, so hermetic tests
script every metric without a database or API.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from docpilot.agent.prompts import REFUSE_ANSWER
from docpilot.agent.types import LoopTraceStep

logger = logging.getLogger(__name__)

VALID_CATEGORIES: frozenset[str] = frozenset(
    {"docs-answerable", "live-state-answerable", "neither"}
)
BODY_FOOTER_SEP = "\nSources:"
_DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "benchmark.json"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_MARKER_RE = re.compile(r"\[(\d+)\]")

# Provider-agnostic detection of the daily-budget wall. Duck-typed on the
# error text (no hard dependency on the Groq SDK): the provider body reads
# "... on tokens per day (TPD): Limit 200000, Used 199562, Requested 4523 ...".
_TPD_MARKER = "tokens per day (TPD)"


class QuotaExhausted(RuntimeError):
    """Fatal, non-transient: the provider's *daily* token budget is spent.

    Per-minute throttling is transient and handled by the generator's own
    retry/backoff; this is the "come back tomorrow / fresh org" wall. Rows
    already checkpointed to disk are safe — resume with the same
    ``--resume STAMP --pipeline <name>`` after the bucket refills.
    """


def is_tpd_exhaustion(exc: BaseException) -> bool:
    """True when *exc* is a daily-token (TPD) exhaustion, not per-minute."""
    return isinstance(exc, BaseException) and _TPD_MARKER in str(exc)

GROUNDEDNESS_SYSTEM: str = (
    "You audit whether the factual claims in an ANSWER are supported by the "
    "provided SOURCES. If any claim goes beyond the sources or contradicts "
    "them, mark grounded=false and list the offending claims.\n"
    'Reply with ONLY a JSON object: {"grounded": true or false, '
    '"unsupported_claims": ["..."]}.\n'
)


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One gold-labeled benchmark question (SPEC §6.1)."""

    id: str
    category: str
    question: str
    gold_key_facts: tuple[str, ...]
    gold_sources: tuple[str, ...]
    gold_refusal: bool
    expected_tool: str | None
    note: str = ""


@dataclass
class RunOutput:
    """What a pipeline produced for one question (duck-typed runner result)."""

    answer: str
    source_files: list[str]
    refused: bool
    trace_steps: list[LoopTraceStep] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class BenchmarkRow:
    """One question scored under one pipeline."""

    id: str
    category: str
    gold_refusal: bool
    refused: bool
    answer: str
    source_files: list[str]
    retrieval_calls: int
    tool_calls: int
    latency_ms: float
    answer_correct: float
    recall: float | None
    citation_validity: float | None
    citation_gold_accuracy: float | None
    marker_count: int
    grounded: bool | None


@dataclass
class PipelineMetrics:
    """Aggregated §6.1 metrics for one pipeline over the benchmark."""

    answer_correctness: float
    retrieval_recall_at_k: float | None
    recall_n: int
    citation_validity: float | None
    citation_validity_n: int
    citation_gold_accuracy: float | None
    citation_gold_n: int
    refusal_accuracy: float | None
    refusal_n: int
    groundedness_rate: float | None
    groundedness_n: int
    avg_latency_ms: float
    avg_retrieval_calls: float
    avg_tool_calls: float
    total_tool_calls: int


@dataclass
class PipelineReport:
    """One pipeline's full scored run."""

    pipeline: str
    dataset_path: str
    generated_at: str
    rows: list[BenchmarkRow]
    metrics: PipelineMetrics
    note: str = ""


@dataclass
class ComparisonRow:
    """One §6.1 metric compared across the two pipelines."""

    metric: str
    classic: float | None
    agentic: float | None
    winner: str  # "classic" | "agentic" | "tie" | "n/a"
    direction: str  # "higher" | "lower"


@dataclass
class ComparisonReport:
    """The §6.1 one-table comparison."""

    metrics: list[ComparisonRow]
    summary: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_benchmark(path: str | Path) -> list[BenchmarkQuestion]:
    """Load and schema-validate the benchmark question set.

    Raises:
        ValueError: On any schema violation.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("questions")
        if not isinstance(rows, list):
            raise ValueError("benchmark dataset must be a JSON array or {questions: [...]}")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("benchmark dataset must be a JSON array or {questions: [...]}")

    questions: list[BenchmarkQuestion] = []
    for i, raw in enumerate(rows):
        qid = raw.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"benchmark[{i}]: 'id' must be non-empty")
        category = raw.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"benchmark[{i}] ({qid!r}): bad category {category!r}; "
                f"expected one of {sorted(VALID_CATEGORIES)}"
            )
        if not isinstance(raw.get("question"), str) or not raw["question"].strip():
            raise ValueError(f"benchmark[{i}] ({qid!r}): 'question' must be non-empty")
        facts = raw.get("gold_key_facts")
        if not isinstance(facts, list) or any(not isinstance(f, str) or not f.strip() for f in facts):
            raise ValueError(
                f"benchmark[{i}] ({qid!r}): 'gold_key_facts' must be a list of non-empty strings"
            )
        sources = raw.get("gold_sources")
        if not isinstance(sources, list) or any(not isinstance(s, str) or not s.strip() for s in sources):
            raise ValueError(
                f"benchmark[{i}] ({qid!r}): 'gold_sources' must be a list of non-empty strings"
            )
        gold_refusal = raw.get("gold_refusal")
        if not isinstance(gold_refusal, bool):
            raise ValueError(f"benchmark[{i}] ({qid!r}): 'gold_refusal' must be a bool")
        tool_expected = raw.get("expected_tool")
        if tool_expected is not None and not isinstance(tool_expected, str):
            raise ValueError(f"benchmark[{i}] ({qid!r}): 'expected_tool' must be a string")
        questions.append(
            BenchmarkQuestion(
                id=qid,
                category=category,
                question=raw["question"],
                gold_key_facts=tuple(facts),
                gold_sources=tuple(sources),
                gold_refusal=gold_refusal,
                expected_tool=tool_expected,
                note=str(raw.get("note", "") or ""),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Answer-body / citation helpers
# ---------------------------------------------------------------------------


def answer_body(display: str) -> str:
    """Strip the ``Sources:`` footer, leaving the answer body (markers only)."""
    return display.split(BODY_FOOTER_SEP, 1)[0].strip()


def body_markers(body: str) -> list[int]:
    """All ``[N]`` citation markers in the answer body (ref numbers, in order)."""
    return [int(m) for m in _MARKER_RE.findall(body)]


def _refused(display_or_answer: str) -> bool:
    """Structural refusal detection (classic path has no flag; check verbatim)."""
    return REFUSE_ANSWER in display_or_answer


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_question(
    q: BenchmarkQuestion,
    out: RunOutput,
    body: str,
    tool_calls: int,
) -> float:
    """Answer-correctness semantics per category (see module docstring)."""
    if q.category == "docs-answerable":
        facts = q.gold_key_facts
        if not facts:
            return 0.0
        matched = sum(1 for f in facts if f.lower() in body.lower())
        return matched / len(facts)
    if q.category == "live-state-answerable":
        if out.refused:
            return 0.0
        return 1.0 if tool_calls >= 1 else 0.5
    # neither
    return 1.0 if out.refused else 0.0


def _load_done_ids(checkpoint_path: str | Path | None) -> set[str]:
    """Ids already scored and persisted in the row-checkpoint file.

    Tolerates a torn trailing line (crash mid-append) by skipping unparsable
    records — a partial last line only costs re-scoring that one question.
    """
    if checkpoint_path is None:
        return set()
    path = Path(checkpoint_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def _load_checkpoint_rows(checkpoint_path: str | Path | None) -> list[BenchmarkRow]:
    """Reconstruct previously scored rows from the row-checkpoint file.

    Used on resume so the aggregated report always spans the *full* question
    set — new rows join (not replace) the checkpointed ones. Torn lines are
    dropped (their question is simply re-run).
    """
    if checkpoint_path is None:
        return []
    path = Path(checkpoint_path)
    if not path.exists():
        return []
    rows: list[BenchmarkRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            rows.append(BenchmarkRow(**payload))
        except (TypeError, ValueError):
            continue
    return rows


def _append_checkpoint(checkpoint_path: str | Path | None, row: BenchmarkRow) -> None:
    """Append one scored row to the row-checkpoint file (jsonl)."""
    if checkpoint_path is None:
        return
    with open(checkpoint_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row_to_dict(row)) + "\n")


def run_pipeline(
    benchmark: list[BenchmarkQuestion],
    pipeline: str,
    run,
    *,
    grounding_checker=None,
    generated_at: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> PipelineReport:
    """Score one pipeline over the benchmark.

    ``run`` is any callable ``(question: str) -> RunOutput``.  Retrieval-call
    count is ``max(1, search steps)`` (the direct path performs exactly one
    retrieval but records no search trace step); tool-call count is the number
    of ``tool_call`` steps.

    ``grounding_checker`` is any callable ``(answer, source_files) -> bool``
    applied to *answered* rows only (refusals carry no factual claims).

    ``checkpoint_path`` enables crash-safe recovery: each scored row is
    appended to a jsonl file immediately, and ids already present are skipped
    on re-entry — so a mid-run daily-quota (TPD) abort never re-burns quota on
    questions that already scored. When a run ends, the caller decides whether
    to keep the checkpoint (aborted) or delete it (completed).
    """
    done = _load_done_ids(checkpoint_path)
    if done:
        logger.warning(
            "Benchmark %s: %d row(s) already checkpointed — skipping %s",
            pipeline, len(done), sorted(done)[:8],
        )
    rows = _load_checkpoint_rows(checkpoint_path)
    for q in benchmark:
        if q.id in done:
            continue
        try:
            out = run(q.question)
        except Exception as exc:
            if is_tpd_exhaustion(exc):
                raise QuotaExhausted(str(exc)) from exc
            raise
        # Refusal contract is the verbatim §3.9 sentence: a pipeline that
        # emits it has refused even if its flag wasn't set (e.g. the generator
        # self-refused on the answer path). Effective flag = flag OR text.
        out = replace(out, refused=out.refused or _refused(out.answer))
        body = answer_body(out.answer)
        markers = body_markers(body)

        retrieval_calls = max(1, sum(1 for s in out.trace_steps if s.step == "search"))
        tool_calls = sum(1 for s in out.trace_steps if s.step == "tool_call")

        validity = None
        if markers:
            validity = sum(1 for m in markers if 1 <= m <= len(out.source_files)) / len(markers)
        gold_accuracy = None
        if markers and q.gold_sources:
            gold_accuracy = sum(
                1
                for m in markers
                if 1 <= m <= len(out.source_files)
                and out.source_files[m - 1] in q.gold_sources
            ) / len(markers)

        recall = None
        if q.gold_sources:
            recall = 1.0 if any(g in out.source_files for g in q.gold_sources) else 0.0

        answer_correct = _score_question(q, out, body, tool_calls)

        grounded = None
        if grounding_checker is not None and not out.refused and out.answer.strip():
            grounded = bool(grounding_checker(out.answer, out.source_files))

        row = BenchmarkRow(
            id=q.id,
            category=q.category,
            gold_refusal=q.gold_refusal,
            refused=out.refused,
            answer=out.answer,
            source_files=out.source_files,
            retrieval_calls=retrieval_calls,
            tool_calls=tool_calls,
            latency_ms=out.latency_ms,
            answer_correct=answer_correct,
            recall=recall,
            citation_validity=validity,
            citation_gold_accuracy=gold_accuracy,
            marker_count=len(markers),
            grounded=grounded,
        )
        _append_checkpoint(checkpoint_path, row)
        rows.append(row)

    return _aggregate(rows, pipeline=pipeline, generated_at=generated_at)


def _aggregate(
    rows: list[BenchmarkRow],
    *,
    pipeline: str,
    generated_at: str | None,
) -> PipelineReport:
    n = len(rows)
    docs = [r for r in rows if r.category == "docs-answerable"]
    live = [r for r in rows if r.category == "live-state-answerable"]
    neither = [r for r in rows if r.category == "neither"]

    recall_rows = [r for r in docs if r.recall is not None]
    validity_rows = [r for r in rows if r.citation_validity is not None]
    gold_rows = [r for r in docs if r.citation_gold_accuracy is not None]
    grounded_rows = [r for r in rows if r.grounded is not None]

    metrics = PipelineMetrics(
        answer_correctness=sum(r.answer_correct for r in rows) / n if n else 0.0,
        retrieval_recall_at_k=(
            sum(r.recall or 0.0 for r in recall_rows) / len(recall_rows) if recall_rows else None
        ),
        recall_n=len(recall_rows),
        citation_validity=(
            sum(r.citation_validity for r in validity_rows) / len(validity_rows)
            if validity_rows
            else None
        ),
        citation_validity_n=len(validity_rows),
        citation_gold_accuracy=(
            sum(r.citation_gold_accuracy for r in gold_rows) / len(gold_rows)
            if gold_rows
            else None
        ),
        citation_gold_n=len(gold_rows),
        refusal_accuracy=(
            sum(1 for r in neither if r.gold_refusal == r.refused) / len(neither)
            if neither
            else None
        ),
        refusal_n=len(neither),
        groundedness_rate=(
            sum(1 for r in grounded_rows if r.grounded) / len(grounded_rows)
            if grounded_rows
            else None
        ),
        groundedness_n=len(grounded_rows),
        avg_latency_ms=sum(r.latency_ms for r in rows) / n if n else 0.0,
        avg_retrieval_calls=sum(r.retrieval_calls for r in rows) / n if n else 0.0,
        avg_tool_calls=sum(r.tool_calls for r in rows) / n if n else 0.0,
        total_tool_calls=sum(r.tool_calls for r in rows),
    )

    return PipelineReport(
        pipeline=pipeline,
        dataset_path="-",
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        rows=rows,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _winner(classic: float | None, agentic: float | None, direction: str) -> str:
    if classic is None and agentic is None:
        return "n/a"
    if classic is None:
        return "agentic"
    if agentic is None:
        return "classic"
    eps = 1e-9
    if abs(classic - agentic) <= eps:
        return "tie"
    if direction == "lower":
        return "classic" if classic < agentic else "agentic"
    return "classic" if classic > agentic else "agentic"


def build_comparison(classic: PipelineReport, agentic: PipelineReport) -> ComparisonReport:
    """Build the §6.1 one-table comparison (higher-better save latency/calls)."""
    c, a = classic.metrics, agentic.metrics
    comparisons = [
        ComparisonRow("answer_correctness", c.answer_correctness, a.answer_correctness,
                      _winner(c.answer_correctness, a.answer_correctness, "higher"), "higher"),
        ComparisonRow("retrieval_recall@k (docs)", c.retrieval_recall_at_k, a.retrieval_recall_at_k,
                      _winner(c.retrieval_recall_at_k, a.retrieval_recall_at_k, "higher"), "higher"),
        ComparisonRow("citation_validity", c.citation_validity, a.citation_validity,
                      _winner(c.citation_validity, a.citation_validity, "higher"), "higher"),
        ComparisonRow("citation_gold_accuracy (docs)", c.citation_gold_accuracy, a.citation_gold_accuracy,
                      _winner(c.citation_gold_accuracy, a.citation_gold_accuracy, "higher"), "higher"),
        ComparisonRow("refusal_accuracy (I-don't-know)", c.refusal_accuracy, a.refusal_accuracy,
                      _winner(c.refusal_accuracy, a.refusal_accuracy, "higher"), "higher"),
        ComparisonRow("groundedness_rate", c.groundedness_rate, a.groundedness_rate,
                      _winner(c.groundedness_rate, a.groundedness_rate, "higher"), "higher"),
        ComparisonRow("avg_latency_ms", c.avg_latency_ms, a.avg_latency_ms,
                      _winner(c.avg_latency_ms, a.avg_latency_ms, "lower"), "lower"),
        ComparisonRow("avg_retrieval_calls", c.avg_retrieval_calls, a.avg_retrieval_calls,
                      _winner(c.avg_retrieval_calls, a.avg_retrieval_calls, "lower"), "lower"),
        ComparisonRow("avg_tool_calls", c.avg_tool_calls, a.avg_tool_calls,
                      _winner(c.avg_tool_calls, a.avg_tool_calls, "lower"), "lower"),
    ]
    decided = [r for r in comparisons if r.winner != "n/a"]
    tally: dict[str, int] = {}
    for r in decided:
        if r.winner != "tie":
            tally[r.winner] = tally.get(r.winner, 0) + 1
    summary = (
        f"{tally.get('classic', 0)} classic / {tally.get('agentic', 0)} agentic / "
        f"{sum(1 for r in decided if r.winner == 'tie')} ties "
        f"over {len(decided)} decided metrics"
    )
    return ComparisonReport(metrics=comparisons, summary=summary)


# ---------------------------------------------------------------------------
# Serialization / formatting
# ---------------------------------------------------------------------------


def _row_to_dict(r: BenchmarkRow) -> dict:
    return {
        "id": r.id,
        "category": r.category,
        "gold_refusal": r.gold_refusal,
        "refused": r.refused,
        "answer_correct": round(r.answer_correct, 4),
        "recall": None if r.recall is None else round(r.recall, 4),
        "citation_validity": None if r.citation_validity is None else round(r.citation_validity, 4),
        "citation_gold_accuracy": (
            None if r.citation_gold_accuracy is None else round(r.citation_gold_accuracy, 4)
        ),
        "marker_count": r.marker_count,
        "retrieval_calls": r.retrieval_calls,
        "tool_calls": r.tool_calls,
        "latency_ms": round(r.latency_ms, 1),
        "grounded": r.grounded,
        "answer": r.answer,
        "source_files": r.source_files,
    }


def _metrics_to_dict(m: PipelineMetrics) -> dict:
    return {
        "answer_correctness": round(m.answer_correctness, 4),
        "retrieval_recall_at_k": None if m.retrieval_recall_at_k is None else round(m.retrieval_recall_at_k, 4),
        "recall_n": m.recall_n,
        "citation_validity": None if m.citation_validity is None else round(m.citation_validity, 4),
        "citation_validity_n": m.citation_validity_n,
        "citation_gold_accuracy": None if m.citation_gold_accuracy is None else round(m.citation_gold_accuracy, 4),
        "citation_gold_n": m.citation_gold_n,
        "refusal_accuracy": None if m.refusal_accuracy is None else round(m.refusal_accuracy, 4),
        "refusal_n": m.refusal_n,
        "groundedness_rate": None if m.groundedness_rate is None else round(m.groundedness_rate, 4),
        "groundedness_n": m.groundedness_n,
        "avg_latency_ms": round(m.avg_latency_ms, 1),
        "avg_retrieval_calls": round(m.avg_retrieval_calls, 3),
        "avg_tool_calls": round(m.avg_tool_calls, 3),
        "total_tool_calls": m.total_tool_calls,
    }


def report_to_dict(report: PipelineReport) -> dict:
    return {
        "pipeline": report.pipeline,
        "dataset_path": report.dataset_path,
        "generated_at": report.generated_at,
        "note": report.note,
        "metrics": _metrics_to_dict(report.metrics),
        "rows": [_row_to_dict(r) for r in report.rows],
    }


def comparison_to_dict(cmp: ComparisonReport) -> dict:
    return {
        "summary": cmp.summary,
        "metrics": [
            {
                "metric": r.metric,
                "classic": None if r.classic is None else round(r.classic, 4),
                "agentic": None if r.agentic is None else round(r.agentic, 4),
                "winner": r.winner,
                "direction": r.direction,
            }
            for r in cmp.metrics
        ],
    }


def format_metrics(report: PipelineReport) -> str:
    m = report.metrics
    lines = [f"Pipeline: {report.pipeline}", "-" * 52, f"{'metric':<34}{'value':>10}", "-" * 52]
    entries = [
        ("answer_correctness", m.answer_correctness),
        ("retrieval_recall@k (docs)", m.retrieval_recall_at_k),
        ("citation_validity", m.citation_validity),
        ("citation_gold_accuracy (docs)", m.citation_gold_accuracy),
        ("refusal_accuracy (I-don't-know)", m.refusal_accuracy),
        ("groundedness_rate", m.groundedness_rate),
        ("avg_latency_ms", m.avg_latency_ms),
        ("avg_retrieval_calls", m.avg_retrieval_calls),
        ("avg_tool_calls", m.avg_tool_calls),
        ("total_tool_calls", float(m.total_tool_calls)),
    ]
    for label, value in entries:
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"{label:<34}{rendered:>10}")
    lines.append("-" * 52)
    return "\n".join(lines)


def format_comparison(cmp: ComparisonReport) -> str:
    lines = [
        "Classic vs. agentic — SPEC §6.1 one-table comparison",
        "-" * 66,
        f"{'metric':<34}{'classic':>10}{'agentic':>10}  winner",
        "-" * 66,
    ]
    for r in cmp.metrics:
        c = "–" if r.classic is None else f"{r.classic:.4f}"
        a = "–" if r.agentic is None else f"{r.agentic:.4f}"
        lines.append(f"{r.metric:<34}{c:>10}{a:>10}  {r.winner}")
    lines.append("-" * 66)
    lines.append(cmp.summary)
    return "\n".join(lines)


def format_report(report: PipelineReport) -> str:
    lines = [format_metrics(report), ""]
    for r in report.rows:
        flags = []
        if r.category == "neither" and r.refused != r.gold_refusal:
            flags.append("refused-when-should-answer" if not r.refused else "answered-when-should-refuse")
        if r.category == "docs-answerable" and r.answer_correct < 1.0:
            flags.append(f"facts={r.answer_correct:.2f}")
        if (
            report.pipeline == "agentic"
            and r.category == "live-state-answerable"
            and r.tool_calls == 0
        ):
            flags.append("no-tool")
        if r.recall == 0.0:
            flags.append("recall-miss")
        if r.grounded is False:
            flags.append("ungrounded")
        if flags:
            lines.append(f"  ✗ {r.id} [{r.category}]: {'; '.join(flags)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Groundedness audit (live path only — inject in tests)
# ---------------------------------------------------------------------------


class GroundingChecker:
    """NLI-style audit of an answer against its offered sources.

    Uses ``Generator.generate`` with :data:`GROUNDEDNESS_SYSTEM` and a tolerant
    JSON parse; a documented proxy for full entailment scoring. Returns
    ``True``/``False``, or ``None`` when the audit output is unparseable.
    """

    def __init__(self, generator=None):
        from docpilot.generation.generator import GroqGenerator

        self._generator = generator if generator is not None else GroqGenerator()

    def check(self, answer: str, source_files: list[str]) -> bool | None:
        sources_text = "\n".join(f"[{i + 1}] {f}" for i, f in enumerate(source_files)) or "(none)"
        prompt = (
            f"{GROUNDEDNESS_SYSTEM}\n\nSOURCES:\n{sources_text}\n\nANSWER:\n{answer}"
        )
        raw = self._generator.generate(prompt)
        obj = self._parse_json(raw)
        if not isinstance(obj, dict):
            return None
        return obj.get("grounded")

    def __call__(self, answer: str, source_files: list[str]) -> bool | None:
        """Duck-typed callable form used by :func:`run_pipeline`."""
        return self.check(answer, source_files)

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last <= first:
            return None
        try:
            obj = json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Live runners + CLI
# ---------------------------------------------------------------------------


def _classic_run(question: str) -> RunOutput:
    """Phase 1 fast path (retrieve → generate → cite) with wall-clock timing."""
    from docpilot.pipeline_ask import ask

    started = time.perf_counter()
    result = ask(question)
    latency = (time.perf_counter() - started) * 1000.0
    return RunOutput(
        answer=result.display,
        source_files=[s.file for s in result.sources],
        refused=_refused(result.display),
        trace_steps=[],
        latency_ms=latency,
    )


def _agentic_run(question: str) -> RunOutput:
    """Phase 2/3 agentic loop (forced) with wall-clock timing and trace counts."""
    from docpilot.agent.pipeline_agentic import agentic_ask

    started = time.perf_counter()
    result = agentic_ask(question, strategy="agentic")
    latency = (time.perf_counter() - started) * 1000.0
    steps = list(result.trace)
    return RunOutput(
        answer=result.answer,
        source_files=[s.file for s in result.sources],
        refused=result.refused,
        trace_steps=steps,
        latency_ms=latency,
    )


_PIPELINE_CHOICES = ("classic", "agentic", "both")


def _write_pipeline_file(out_dir: Path, stamp: str, report: PipelineReport) -> Path:
    """Persist one pipeline's scored run immediately (crash-safe sidecar)."""
    p = out_dir / f"benchmark_{stamp}_{report.pipeline}.json"
    p.write_text(
        json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8"
    )
    return p


def _load_pipeline_file(path: Path, pipeline: str) -> PipelineReport:
    """Rebuild a PipelineReport from a sidecar file (for resume/merge)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    m = payload["metrics"]
    metrics = PipelineMetrics(
        answer_correctness=m["answer_correctness"],
        retrieval_recall_at_k=m.get("retrieval_recall_at_k"),
        recall_n=m.get("recall_n", 0),
        citation_validity=m.get("citation_validity"),
        citation_validity_n=m.get("citation_validity_n", 0),
        citation_gold_accuracy=m.get("citation_gold_accuracy"),
        citation_gold_n=m.get("citation_gold_n", 0),
        refusal_accuracy=m.get("refusal_accuracy"),
        refusal_n=m.get("refusal_n", 0),
        groundedness_rate=m.get("groundedness_rate"),
        groundedness_n=m.get("groundedness_n", 0),
        avg_latency_ms=m["avg_latency_ms"],
        avg_retrieval_calls=m["avg_retrieval_calls"],
        avg_tool_calls=m["avg_tool_calls"],
        total_tool_calls=m.get("total_tool_calls", 0),
    )
    rows = [
        BenchmarkRow(
            id=r["id"],
            category=r["category"],
            gold_refusal=r["gold_refusal"],
            refused=r["refused"],
            answer=r["answer"],
            source_files=r["source_files"],
            retrieval_calls=r["retrieval_calls"],
            tool_calls=r["tool_calls"],
            latency_ms=r["latency_ms"],
            answer_correct=r["answer_correct"],
            recall=r.get("recall"),
            citation_validity=r.get("citation_validity"),
            citation_gold_accuracy=r.get("citation_gold_accuracy"),
            marker_count=r["marker_count"],
            grounded=r.get("grounded"),
        )
        for r in payload["rows"]
    ]
    return PipelineReport(
        pipeline=pipeline,
        dataset_path=payload.get("dataset_path", "-"),
        generated_at=payload.get("generated_at", "-"),
        rows=rows,
        metrics=metrics,
        note=payload.get("note", ""),
    )


def _combined_json(
    dataset_path: Path,
    reports: dict[str, PipelineReport],
    *,
    partial: bool,
) -> dict:
    cmp = None if partial else build_comparison(reports["classic"], reports["agentic"])
    note = (
        "SPEC §6.1 classic-vs-agentic comparison — same question set, both pipelines."
        if not partial
        else "PARTIAL — only "
        + ", ".join(sorted(reports))
        + " written (missing "
        + ", ".join(sorted(set(_PIPELINE_CHOICES) - {"both"} - set(reports)))
        + "); rerun the missing half with --pipeline and --resume STAMP to resume."
    )
    return {
        "dataset_path": str(dataset_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "pipelines": {name: report_to_dict(r) for name, r in reports.items()},
        "comparison": comparison_to_dict(cmp) if cmp else None,
    }


def run_benchmark_live(
    dataset_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    run_groundedness: bool = True,
    pipeline: str = "both",
    stamp: str | None = None,
) -> Path:
    """Run the selected pipeline(s) over the benchmark, checkpointing each half.

    Each pipeline's report is persisted to a sidecar file as soon as it
    completes, so a quota/failure mid-run never loses the finished half; the
    combined file + comparison is written only when both sides are present.

    Crash-safe row checkpoints (``<stamp>_<pipeline>.rows.jsonl``): every
    scored row is appended as it finishes, and a resumed run skips ids already
    in the file while still reporting over the full question set. A daily-TPD
    quota abort raises :class:`QuotaExhausted` (CLI exit code 3) *after*
    writing any completed sidecar + the partial combined file — nothing is
    fabricated and no scored row is re-burned on resume. Completed checkpoints
    are deleted when their half finishes.

    Resume flow: if a half crashed before completing, rerun it alone with
    ``pipeline="agentic"`` (or ``classic``) and ``stamp=<the crashed run's
    stamp>``; the finished sidecar from the crashed run is adopted and the
    combined report + comparison is produced for that stamp.
    """
    if pipeline not in _PIPELINE_CHOICES:
        raise ValueError(
            f"pipeline must be one of {_PIPELINE_CHOICES}; got {pipeline!r}"
        )
    dataset_path = Path(dataset_path) if dataset_path else _DATASET_PATH
    benchmark = load_benchmark(dataset_path)
    out_dir = Path(out_dir) if out_dir else _REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    logger.info("Benchmark: %d questions from %s", len(benchmark), dataset_path)

    checker = GroundingChecker() if run_groundedness else None
    reports: dict[str, PipelineReport] = {}
    quota_abort: QuotaExhausted | None = None
    for name in ("classic", "agentic"):
        if pipeline not in ("both", name):
            continue
        run = _classic_run if name == "classic" else _agentic_run
        checkpoint = out_dir / f"benchmark_{stamp}_{name}.rows.jsonl"
        try:
            report = run_pipeline(
                benchmark, name, run,
                grounding_checker=checker,
                checkpoint_path=checkpoint,
            )
        except QuotaExhausted as exc:
            quota_abort = exc
            print(
                "\n⛔ Daily token budget exhausted (TPD) — nothing fabricated; "
                "scored rows are checkpointed. Resume when the bucket refills "
                f"with: --pipeline {name} --resume {stamp}"
            )
            break
        report.dataset_path = str(dataset_path)
        report.note = (
            f"Live {report.pipeline} (Groq, temp 0); "
            + ("groundedness audit on" if run_groundedness else "groundedness audit off")
            + "; docs recall@k over offered sources"
        )
        sidecar = _write_pipeline_file(out_dir, stamp, report)
        checkpoint.unlink(missing_ok=True)
        reports[name] = report
        logger.info("Benchmark %s pipeline complete → %s", name, sidecar)
        print(format_metrics(report))
        print()

    partial = len(reports) < 2
    if partial:
        # Resume: adopt a previously-checkpointed sidecar for the missing half
        # when it already exists under this stamp (e.g. --pipeline agentic
        # --resume STAMP after the classic half completed under STAMP earlier).
        for name in ("classic", "agentic"):
            if name not in reports:
                side = out_dir / f"benchmark_{stamp}_{name}.json"
                if side.exists():
                    reports[name] = _load_pipeline_file(side, name)
    partial = len(reports) < 2
    out_path = out_dir / f"benchmark_{stamp}.json"
    out_path.write_text(
        json.dumps(_combined_json(dataset_path, reports, partial=partial), indent=2)
        + "\n",
        encoding="utf-8",
    )

    if partial:
        missing = sorted(set(_PIPELINE_CHOICES) - {"both"} - set(reports))
        print(
            f"⚠ PARTIAL — {sorted(reports)} done, {missing} missing; "
            f"rerun with --pipeline {missing[0]} to resume, then merge."
        )
        if reports:
            # No report to print when the very first pipeline aborted on quota.
            print(format_report(next(iter(reports.values()))))
    else:
        cmp = build_comparison(reports["classic"], reports["agentic"])
        print(format_comparison(cmp))
        print()
        print(format_report(reports["classic"]))
        print()
        print(format_report(reports["agentic"]))
    logger.info("Benchmark report written to %s", out_path)

    if quota_abort is not None:
        # Partial combined report is on disk; signal the abort so the CLI maps
        # it to a distinct exit code (see main).
        raise quota_abort
    return out_path


def merge_benchmark_checkpoints(out_dir: str | Path, stamp: str) -> Path:
    """Merge the two sidecar files of *stamp* into the combined report."""
    out_dir = Path(out_dir)
    files = {p: out_dir / f"benchmark_{stamp}_{p}.json" for p in ("classic", "agentic")}
    missing = [p for p, f in files.items() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing sidecar(s) {missing} for stamp {stamp!r} in {out_dir}"
        )
    reports = {
        p: _load_pipeline_file(f, p) for p, f in files.items()
    }
    payload = _combined_json(Path(reports["classic"].dataset_path), reports, partial=False)
    out_path = out_dir / f"benchmark_{stamp}.json"
    out_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Merged benchmark checkpoints → %s", out_path)
    return out_path


# Per-minute token headroom required for a real call to be serviable right now.
# The per-minute bucket refills in seconds, so this is only a functional gate
# ("would my first call 429 this minute"), not a run-viability gate.
_PROBE_MIN_RPM_TOKENS = 1000


def probe_quota() -> int:
    """Functional gate: auth + model reachable + a real completion works +
    per-minute token headroom for a call right now.

    Returns 0 (OK), 1 (actively rate-limited / throttled), or 2 (other
    failure). **TPD caveat:** Groq does not expose the daily token bucket via
    the API (verified live 2026-09-08 — no ratelimit headers and no
    admission-time gate on ``max_tokens``), so this probe makes no claim about
    daily headroom. Before a live run the operator must confirm the key's org
    has a fresh daily bucket; a mid-run TPD wall is handled by per-half
    checkpointing + resume-at-stamp + retry-after fail-fast, not by this gate
    (SPEC §6.5).
    """
    from docpilot.generation.generator import GroqGenerator

    gen = GroqGenerator(max_retries=1)
    print(f"probe: model={gen._model}")
    result = gen.probe()
    if not result.ok:
        reason = result.reason
        if "Rate limit" in reason or "429" in reason:
            m = re.search(r"Limit (\d+), Used (\d+)", reason)
            if m:
                print(f"probe: RATE LIMITED — Limit={m.group(1)} Used={m.group(2)}")
            else:
                print(f"probe: RATE LIMITED — {reason[:200]}")
            return 1
        print(f"probe: FAILED ({reason[:200]})")
        return 2
    print(f"probe: OK — completion={result.completion[:40]!r}")
    print(
        "probe: per-minute rate limits — "
        f"tokens remaining={result.remaining}/{result.limit}, "
        f"requests remaining={result.requests_remaining}/1000"
    )
    if result.remaining is not None and result.remaining < _PROBE_MIN_RPM_TOKENS:
        print(
            f"probe: GATE FAIL — per-minute tokens remaining {result.remaining} < "
            f"{_PROBE_MIN_RPM_TOKENS}; a real call would 429 right now."
        )
        return 1
    print(
        "probe: NOTE — daily TPD headroom is NOT exposed by Groq; you must "
        "confirm this key's org has a fresh daily bucket before a live run."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m docpilot.eval benchmark [--probe] [--dataset PATH] [--out DIR] [--no-grounding] [--pipeline {classic,agentic,both}] [--resume STAMP] [--merge DIR STAMP]``.

    ``--probe`` performs a one-call quota headroom check and exits 0 (OK) /
    1 (rate limited) / 2 (other failure) — gate live runs with ``--probe &&``.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    dataset_path: str | Path | None = None
    out_dir: str | Path | None = None
    run_groundedness = True
    pipeline: str = "both"
    stamp: str | None = None
    i = 0
    while i < len(args):
        if args[i] in ("--datasets", "--dataset", "--triples"):
            i += 1
            if i >= len(args):
                print(f"{args[i - 1]} requires a path", file=sys.stderr)
                return 2
            dataset_path = args[i]
        elif args[i] == "--out":
            i += 1
            if i >= len(args):
                print("--out requires a directory", file=sys.stderr)
                return 2
            out_dir = args[i]
        elif args[i] == "--no-grounding":
            run_groundedness = False
        elif args[i] == "--probe":
            return probe_quota()
        elif args[i] == "--pipeline":
            i += 1
            if i >= len(args) or args[i] not in _PIPELINE_CHOICES:
                print(f"--pipeline requires one of {_PIPELINE_CHOICES}", file=sys.stderr)
                return 2
            pipeline = args[i]
        elif args[i] == "--resume":
            i += 1
            if i >= len(args) or not (
                len(args[i]) == 15 and args[i][8] == "_"
                and args[i][:8].isdigit() and args[i][9:].isdigit()
            ):
                print("--resume requires the crashed run's STAMP (e.g. 20260908_190539)", file=sys.stderr)
                return 2
            stamp = args[i]
        elif args[i] == "--merge":
            i += 1
            if i + 1 >= len(args):
                print("--merge requires DIR and STAMP", file=sys.stderr)
                return 2
            merge_benchmark_checkpoints(args[i], args[i + 1])
            return 0
        else:
            print(f"unknown argument {args[i]!r}", file=sys.stderr)
            return 2
        i += 1
    try:
        run_benchmark_live(
            dataset_path,
            out_dir=out_dir,
            run_groundedness=run_groundedness,
            pipeline=pipeline,
            stamp=stamp,
        )
    except QuotaExhausted as exc:
        print(
            f"quota exhausted — daily token budget spent (TPD: {exc}); "
            f"rerun with --resume {stamp} --pipeline {pipeline} after the "
            "bucket refills",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())