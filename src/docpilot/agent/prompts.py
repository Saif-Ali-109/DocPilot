"""Judge and reformulation prompts (Phase 2, SPEC §4).

The :data:`REFUSE_ANSWER` constant is the verbatim SPEC §3.9 sentence and
**must never be altered** — tests assert its exact value.
"""

from __future__ import annotations

from docpilot.core.models import RetrieverResult

# ---------------------------------------------------------------------------
# REFUSE_ANSWER — SPEC §3.9 verbatim (tests assert exact value)
# ---------------------------------------------------------------------------

REFUSE_ANSWER: str = (
    "I don't know — the available documentation does not cover this question."
)
"""The exact refusal sentence from SPEC §3.9.

This value is *untouchable*.  Tests assert ``REFUSE_ANSWER`` matches the spec
word-for-word so a text-edit cannot silently degrade the "I don't know"
behaviour.
"""


# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT: str = """\
You are DocPilot's retrieval-sufficiency judge for technical documentation.

Your job is to evaluate whether the retrieved chunks provide enough evidence
to answer the user's question **fully and accurately** — without hallucinating
or guessing.

You MUST output ONLY a single JSON object with exactly these keys:
  "verdict"            — "sufficient" or "insufficient"
  "reason"             — a short explanation (one sentence is fine)
  "reformulated_query" — a rewritten retrieval query that would help find the
                         missing evidence, or null if the current query is fine
                         or the verdict is sufficient.

Rules:
- If the chunks clearly contain enough evidence to answer the question
  (even if partial), return "sufficient".
- If the chunks are missing key information needed to give a complete and
  accurate answer, return "insufficient" and propose a single reformulated
  query that focuses on the missing aspect.
- If uncertain, lean towards "sufficient" — the answering LLM can still
  refuse if its own judgement differs.
- The reformulated_query (when present) MUST be in the same language as the
  question.
- Output ONLY the JSON object.  No markdown fences, no explanation before
  or after the JSON.
"""

# Maximum characters kept per chunk in the judge prompt (truncation limit).
_CHUNK_TRUNCATE_LIMIT: int = 1200


# ---------------------------------------------------------------------------
# Judge user prompt builder
# ---------------------------------------------------------------------------


def build_judge_user_prompt(
    question: str,
    query_used: str,
    results: list[RetrieverResult],
) -> str:
    """Build the user-facing prompt for the sufficiency judge.

    Args:
        question: The original user question.
        query_used: The retrieval query that produced *results*.
        results: The retriever results to evaluate.

    Returns:
        A formatted prompt string ready to be passed to
        ``generator.generate()``.
    """
    lines: list[str] = []

    lines.append("QUESTION:")
    lines.append(question)
    lines.append("")
    lines.append("QUERY USED FOR RETRIEVAL:")
    lines.append(query_used)
    lines.append("")
    lines.append("RETRIEVED CHUNKS:")
    lines.append("")

    for i, r in enumerate(results):
        chunk_text = r.chunk.content[:_CHUNK_TRUNCATE_LIMIT]
        source_label = r.chunk.source_file
        heading = r.chunk.heading_path
        if heading:
            source_label = f"{source_label} → {heading}"
        lines.append(f"  [{i + 1}] (source: {source_label}, score: {r.score:.4f})")
        lines.append(f"  {chunk_text}")
        lines.append("")

    lines.append("Evaluate whether these chunks provide sufficient evidence "
                  "to answer the question. Output ONLY the JSON.")

    return "\n".join(lines)
