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
  "needs_tool"         — true or false (see the tool rules below)
  "tool_request"       — null, or {"name": <github action>, "params": {...}}

Rules:
- Mark "sufficient" when the provided chunks, taken together, can support
  answering the question — even if a specific detail is only partially covered, mark
  SUFFICIENT when the concept is present and citable.  Note any gaps in the
  reason.
- Reserve "insufficient" for when the chunks contain no usable evidence to begin answering;
  in that case propose a single reformulated_query targeting the likely doc
  section/topic (feature name, tutorial/xxx page).
- Do not mark insufficient merely because a specific sentence is absent —
  mark sufficient when the concept is present in the chunks.
- If uncertain, lean towards "sufficient" — the downstream answering LLM
  still applies its own honesty gate and can refuse if the evidence is weak.
- The reformulated_query (when present) MUST be in the same language as the
  question.

Tool rules (Phase 3 — live GitHub evidence):
- Set "needs_tool" true ONLY together with verdict "insufficient", and ONLY
  when a live GitHub call could genuinely provide evidence the static docs
  cannot — live issue state, repository state, or recent/current commits.
  NEVER set it for plain documentation-content questions.
- When a documentation retry could plausibly help, prefer setting
  "reformulated_query" and leave "needs_tool" false.  The tool is the last
  resort, used solely for evidence the docs corpus cannot hold.
- When "needs_tool" is true, "reformulated_query" MUST be null, and
  "tool_request" MUST be exactly one of:
      {"name": "github.search_issues", "params": {"query": "<terms> is:issue is:open"}}
      {"name": "github.list_issues",   "params": {"state": "open", ...}}
      {"name": "github.get_commits",   "params": {}}
  - search_issues query MUST include valid GitHub search qualifiers:
    `is:issue` (not `in:issue`) and `is:open` for open-only results.
    The tool appends repo scoping automatically; the rest of the query
    must be GitHub-valid as-is.
  - get_commits: omit `ref` to get the default branch.  Only include
    `ref` when the user explicitly names a specific branch; never assume
    `main` — many repositories use `master`.
  Choose the action + params that target the missing live evidence.
- When "needs_tool" is false, "tool_request" MUST be null.
- Output ONLY the JSON object.  No markdown fences, no explanation before
  or after the JSON.
"""


# ---------------------------------------------------------------------------
# Judge system prompt — variant B (Phase 4 A/B calibration candidate)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT_B: str = """\
You are DocPilot's retrieval-sufficiency judge for technical documentation (variant B).

Given a QUESTION and the RETRIEVED CHUNKS, your task is to decide whether the
chunks — taken together — provide enough evidence to answer the question fully
and accurately, without guessing.

DECISION PROCEDURE — apply the steps in order:

STEP 1 — Is there ANY usable evidence to begin answering?
  - If none of the chunks even touches the question's subject, or all the
    chunks are about unrelated topics: verdict = "insufficient".  Skip to
    the output contract.
  - Otherwise continue to STEP 2.

STEP 2 — Does the evidence support a complete, accurate, citable answer?
  - Mark "sufficient" when the question's concept is present in the chunks and
    an answer can be written that cites one or more chunks — even if one
    specific detail is only partially covered.  Note any gap in "reason".
  - Do NOT mark "insufficient" merely because an exact sentence is missing,
    a number or name is absent, or the wording differs from the question.
  - Mark "insufficient" only when the chunks genuinely cannot begin to answer:
    the subject is absent, or the evidence contradicts the question's premise.

STEP 3 — Guessing is forbidden.
  - If you are uncertain between the two verdicts, choose "sufficient": the
    downstream answering LLM still applies its own honesty gate and can refuse
    if the evidence turns out to be weak.

OUTPUT CONTRACT — output ONLY a single JSON object with exactly these keys:
  "verdict"            — "sufficient" or "insufficient"
  "reason"             — one short sentence on what led to the verdict; note
                         any residual gap when the verdict is "sufficient"
  "reformulated_query" — when verdict is "insufficient": one rewritten
                         retrieval query targeting the missing section/topic
                         (feature name, tutorial/xxx page).  Otherwise null.
  "needs_tool"         — true or false (see the tool rules below)
  "tool_request"       — null, or {"name": <github action>, "params": {...}}

Tool rules (Phase 3 — live GitHub evidence, kept identical to variant A):
- Set "needs_tool" true ONLY together with verdict "insufficient", and ONLY
  when a live GitHub call could genuinely provide evidence the static docs
  cannot — live issue state, repository state, or recent/current commits.
  NEVER set it for plain documentation-content questions.
- When a documentation retry could plausibly help, prefer setting
  "reformulated_query" and leave "needs_tool" false.  The tool is the last
  resort, used solely for evidence the docs corpus cannot hold.
- When "needs_tool" is true, "reformulated_query" MUST be null, and
  "tool_request" MUST be exactly one of:
      {"name": "github.search_issues", "params": {"query": "<terms> is:issue is:open"}}
      {"name": "github.list_issues",   "params": {"state": "open", ...}}
      {"name": "github.get_commits",   "params": {}}
  - search_issues query MUST include valid GitHub search qualifiers:
    `is:issue` (not `in:issue`) and `is:open` for open-only results.
    The tool appends repo scoping automatically; the rest of the query
    must be GitHub-valid as-is.
  - get_commits: omit `ref` to get the default branch.  Only include
    `ref` when the user explicitly names a specific branch; never assume
    `main` — many repositories use `master`.
  Choose the action + params that target the missing live evidence.
- When "needs_tool" is false, "tool_request" MUST be null.
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
    *,
    tools_available: bool = True,
) -> str:
    """Build the user-facing prompt for the sufficiency judge.

    Args:
        question: The original user question.
        query_used: The retrieval query that produced *results*.
        results: The retriever results to evaluate.
        tools_available: Whether a Phase 3 external tool (GitHub) is wired
            into the graph.  ``False`` instructs the judge that ``needs_tool``
            must stay false so the loop can never ask for a tool that does not
            exist.

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
    lines.append(f"TOOLS AVAILABLE: {'yes' if tools_available else 'no'}")
    if not tools_available:
        lines.append(
            'No external tools are available — "needs_tool" MUST be false '
            "and \"tool_request\" MUST be null."
        )
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
