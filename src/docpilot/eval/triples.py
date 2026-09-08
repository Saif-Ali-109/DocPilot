"""Labeled verdict-triple dataset for judge calibration (SPEC §6.2).

A *triple* is ``(question, context, gold_verdict)`` — judged by the
:class:`~docpilot.agent.judge.SufficiencyJudge` **independently of retrieval
and routing quality**.  Contexts are real passages cut from the FastAPI
corpus (``docs/en/docs/**``), so the judge sees the same shape of chunk it
meets in production.

Categories:
  - ``sufficient-direct``   — a chunk directly answers the question.
  - ``sufficient-partial``  — the concept is present but a detail is missing;
                              gold is still ``sufficient`` (anti-over-refusal).
  - ``insufficient-missing``— the subject is absent from context; honest
                              ``insufficient``.
  - ``irrelevant``          — context present but clearly unrelated.
  - ``adversarial``         — trickily worded question (negated premise,
                              near-miss vocabulary, precision trap); gold may
                              be either verdict.  These are the
                              *adversarial paraphrases* of SPEC §6.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_GOLDS: frozenset[str] = frozenset({"sufficient", "insufficient"})
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "sufficient-direct",
        "sufficient-partial",
        "insufficient-missing",
        "irrelevant",
        "adversarial",
    }
)
_CONFIG_KEYS: tuple[str, ...] = (
    "id",
    "category",
    "question",
    "query_used",
    "gold",
    "chunks",
)


@dataclass(frozen=True)
class TripleChunk:
    """One context chunk as the judge would receive it."""

    content: str
    source_file: str
    heading: str = ""
    score: float = 0.8


@dataclass(frozen=True)
class JudgeTriple:
    """A labeled ``(question, context, gold)`` instance for calibration."""

    id: str
    category: str
    question: str
    query_used: str
    gold: str
    chunks: tuple[TripleChunk, ...]
    paraphrase_of: str | None = None
    note: str = ""

    def as_dict(self) -> dict:
        """Round-trip serialization (used by the report writer)."""
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "query_used": self.query_used,
            "gold": self.gold,
            "chunks": [
                {
                    "content": c.content,
                    "source_file": c.source_file,
                    "heading": c.heading,
                    "score": c.score,
                }
                for c in self.chunks
            ],
            "paraphrase_of": self.paraphrase_of,
            "note": self.note,
        }


def _validate(raw: dict, index: int) -> JudgeTriple:
    missing = [k for k in _CONFIG_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"judge_triples[{index}]: missing key(s) {missing}"
        )
    tid = raw["id"]
    if not isinstance(tid, str) or not tid.strip():
        raise ValueError(f"judge_triples[{index}]: 'id' must be a non-empty str")
    category = raw["category"]
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"judge_triples[{index}] ({tid!r}): bad category {category!r}; "
            f"expected one of {sorted(VALID_CATEGORIES)}"
        )
    gold = raw["gold"]
    if gold not in VALID_GOLDS:
        raise ValueError(
            f"judge_triples[{index}] ({tid!r}): bad gold {gold!r}; "
            f"expected one of {sorted(VALID_GOLDS)}"
        )
    for key in ("question", "query_used"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(
                f"judge_triples[{index}] ({tid!r}): '{key}' must be non-empty"
            )
    chunks_raw = raw["chunks"]
    if not isinstance(chunks_raw, list) or not chunks_raw:
        raise ValueError(
            f"judge_triples[{index}] ({tid!r}): 'chunks' must be a non-empty list"
        )
    chunks: list[TripleChunk] = []
    for c_i, c in enumerate(chunks_raw):
        content = c.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"judge_triples[{index}] ({tid!r}).chunks[{c_i}]: "
                "'content' must be non-empty"
            )
        source = c.get("source_file", "")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"judge_triples[{index}] ({tid!r}).chunks[{c_i}]: "
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
    paraphrase_of = raw.get("paraphrase_of")
    if paraphrase_of is not None and not isinstance(paraphrase_of, str):
        raise ValueError(
            f"judge_triples[{index}] ({tid!r}): 'paraphrase_of' must be a str or null"
        )
    note = raw.get("note")
    if not isinstance(note, str):
        note = ""
    return JudgeTriple(
        id=tid,
        category=category,
        question=raw["question"],
        query_used=raw["query_used"],
        gold=gold,
        chunks=tuple(chunks),
        paraphrase_of=paraphrase_of,
        note=note,
    )


def load_judge_triples(path: str | Path) -> list[JudgeTriple]:
    """Load and schema-validate the labeled triple dataset.

    Raises:
        ValueError: On any schema violation — the dataset is the contract
            for the calibration run, so bad rows fail loudly, never silently.
        FileNotFoundError: When *path* does not exist.
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
    return [_validate(row, i) for i, row in enumerate(rows)]