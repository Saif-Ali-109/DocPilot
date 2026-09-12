"""SSE wire-format helpers (Phase 5, SPEC §7).

The chat endpoint streams Server-Sent Events — every event is a type-tagged
JSON dict on a ``data:`` line.  UI clients switch on ``event["type"]``:

    gate      routing decision {decision, detail, latency_ms}
    step      live mirror of one LoopTraceStep {step: <trace-step dict>}
    search    retrieval payload for the debug panel {turn, query, results,
              latency_ms}
    judge     (agentic path, via ``step`` with step=="judge")
    tool_call (agentic path, via ``step`` with step=="tool_call")
    code      (code route) one generation attempt's validation verdict
              {attempt, code_blocks, verdicts, passed, validation_failed,
              refused, reasons, attempt_latency_ms}
    token     one streamed answer delta {delta}
    answer    final display text + sources {text, sources, refused, direct};
              on the code route also carries ``code``, ``validation_failed``,
              ``validation_reasons``, ``validation_turns`` and the overall
              ``verdict`` (the per-attempt verdicts arrive via ``code`` events)
    error     {message}
    done      {trace, latency_ms, usage, validation_turns?,
              validation_failed?} — terminal event
"""

from __future__ import annotations

import json
from typing import Any


def event(kind: str, **payload: Any) -> dict:
    """Build one type-tagged SSE event dict."""
    return {"type": kind, **payload}


def sse_line(data: dict) -> str:
    """Render one SSE ``data:`` line (JSON, non-ASCII preserved)."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def parse_sse_block(block: str) -> dict | None:
    """Parse one SSE block back into a dict (test helper).

    Returns ``None`` for comments/heartbeats or malformed lines; the first
    ``data:`` line wins (our events are single-line by construction).
    """
    for line in block.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw:
            return None
        return json.loads(raw)
    return None