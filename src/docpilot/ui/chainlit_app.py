"""Chainlit UI for DocPilot (Phase 5, SPEC §7).

Runs the answer pipeline in-process (:func:`docpilot.api.service.ask_events`
— no HTTP hop, secrets stay server-side) and renders the event stream live:

    * ``token`` events stream into the assistant message as they arrive
      (perceived latency — first token in seconds on the direct fast path);
    * ``step`` events become `cl.Step` trace elements (gate/judge/tool_call/
      answer) for inspectability;
    * ``search`` events become a retrieval debug panel — chunks with scores
      and content excerpts;
    * the ``answer`` event replaces the streamed body with the authoritative
      citation-formatted text (answer + source footer) and renders the
      sources as their own element;
    * Phase 6 (PLAN §7.2 T6): a *Generate validated code* setting in the
      chat-settings panel is the explicit opt-in — with it on, each message
      runs :func:`docpilot.api.service.code_events` (the code route: gate →
      per-attempt retrieval + validation-verdict ``code`` events → answer,
      refused-with-sources when validation fails) instead of the standard
      ask path; the per-attempt ``code`` events render as validation-verdict
      trace steps;
    * Chainlit's built-in persistence keeps per-thread chat history (SPEC §7
      amendment: "built-in chat-history persistence … instead of hand-rolled
      session state").

Run from the repo root::

    chainlit run src/docpilot/ui/chainlit_app.py
"""

from __future__ import annotations

import asyncio
import logging

import chainlit as cl
from chainlit.input_widget import Switch

from docpilot.api.service import ask_events, code_events

logger = logging.getLogger(__name__)

_SENTINEL = object()

# step.event["step"] → (cl.Step label, cl.Step type)
_STEP_LABELS = {
    "gate": ("gate", "run"),
    "search": ("search", "retrieval"),
    "judge": ("judge", "llm"),
    "tool_call": ("tool_call", "tool"),
    "code_validation": ("code_validation", "run"),
    "answer": ("answer", "llm"),
    "refuse": ("refuse", "run"),
}


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "**DocPilot** — evidence-driven agentic RAG for FastAPI docs.\n\n"
            "Ask a documentation question. Simple questions take the fast path "
            "(retrieval → streamed answer); complex or live-state questions route "
            "through the agentic loop (judge + GitHub tool when needed). Retrieval "
            "chunks, scores, judge decisions and tool calls are shown as steps.\n\n"
            "**Want code?** Tick *Generate validated code* in the settings panel — "
            "the response then goes through the Phase 6 code route: every emitted "
            "block is validated against the retrieved docs and refused (with "
            "sources) if it can't be validated. The opt-in is per chat session."
        )
    ).send()
    settings = cl.ChatSettings(
        [
            Switch(
                id="ask_code",
                label="Generate validated code",
                tooltip="Explicit Phase 6 code route: generate → validate against "
                "retrieved docs → return or refuse with sources.",
                initial=False,
            )
        ]
    )
    await settings.send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("ask_code", bool(settings.get("ask_code", False)))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    use_code = bool(cl.user_session.get("ask_code", False))

    queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
    loop = asyncio.get_running_loop()

    def emit(data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    def run() -> dict:
        try:
            if use_code:
                # Explicit Phase 6 code route (§7.5 opt-in): the response is
                # validated against the retrieved docs, or refused + sources.
                return code_events(message.content, emit=emit)
            return ask_events(message.content, strategy="auto", emit=emit)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    task = asyncio.create_task(asyncio.to_thread(run))

    final_text: str = ""
    final_sources: list[dict] = []
    refused = False
    validation_failed = False

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        event_type = item.get("type")

        if event_type == "token":
            await answer_msg.stream_token(item.get("delta", ""))

        elif event_type == "step":
            step_info = item.get("step") or {}
            name, step_type = _STEP_LABELS.get(
                step_info.get("step"), (step_info.get("step", "step"), "run")
            )
            label = f"{name}"
            if step_info.get("decision"):
                label = f"{name} · {step_info['decision']}"
            step = cl.Step(name=label, type=step_type)
            await step.send()
            detail = step_info.get("detail") or ""
            latency = step_info.get("latency_ms")
            step.output = detail + (f"\n latency: {latency} ms" if latency is not None else "")
            await step.update()

        elif event_type == "search":
            await _show_retrieval_panel(item)

        elif event_type == "code":
            # A live validation-verdict trace event; on the code route the
            # message stays a "working" placeholder until the final answer
            # replaces it (the route re-generates up to N attempts).
            await _show_code_attempt(item)
            attempt = item.get("attempt", "?")
            answer_msg.content = f"_validating code — attempt {attempt}…_"
            await answer_msg.update()

        elif event_type == "answer":
            final_text = item.get("text", "")
            final_sources = item.get("sources") or []
            refused = bool(item.get("refused"))
            validation_failed = bool(item.get("validation_failed"))

        elif event_type == "done":
            pass  # finalize below

        elif event_type == "error":
            await answer_msg.update()
            answer_msg.content = f"⚠️ {item.get('message', 'unknown error')}"
            await answer_msg.update()
            break

    await task

    # Authoritative final render: the streamed body is replaced by the full
    # citation-formatted display (answer + source footer).  On the code route
    # the sources still render when the answer is a *validation* refusal —
    # they are the evidence for why no code was returned.
    if final_text:
        answer_msg.content = final_text
        await answer_msg.update()
        if final_sources and (not refused or validation_failed):
            await _show_sources(final_sources)


async def _show_retrieval_panel(payload: dict) -> None:
    """Render the debug panel: retrieved chunks with scores + excerpts."""
    results = payload.get("results") or []
    lines = [f"turn {payload.get('turn', 1)} · {payload.get('query', '')}", ""]
    for r in results:
        kind = r.get("kind") or ""
        kind_part = f" ({kind})" if kind else ""
        lines.append(
            f"- [{r.get('score', '?')}] `{r.get('file')}{kind_part}`"
            + (f" → {r['heading']}" if r.get("heading") else "")
        )
        excerpt = (r.get("excerpt") or "").replace("\n", " ")
        if excerpt:
            lines.append(f"    {excerpt[:160]}…")
    step = cl.Step(
        name=f"retrieved chunks · t{payload.get('turn', 1)}",
        type="retrieval",
    )
    await step.send()
    step.output = "\n".join(lines)
    await step.update()


async def _show_sources(sources: list[dict]) -> None:
    """Render the cited sources as their own step (like a reference card)."""
    lines = ["**Sources**", ""]
    for s in sources:
        kind = s.get("kind") or ""
        kind_part = f" ({kind})" if kind else ""
        heading = f" → {s['heading']}" if s.get("heading") else ""
        lines.append(f"[{s.get('ref')}] `{s.get('file')}{kind_part}`{heading}")
    step = cl.Step(name="cited sources", type="run")
    await step.send()
    step.output = "\n".join(lines)
    await step.update()


async def _show_code_attempt(payload: dict) -> None:
    """Render one generation attempt's validation verdict (the ``code`` event:
    per-block checks with PASS/FAIL/SKIP statuses, plus any failure reasons)."""
    attempt = payload.get("attempt", "?")
    blocks = payload.get("code_blocks") or []
    verdicts = payload.get("verdicts") or []
    reasons = payload.get("reasons") or []
    markers = {"pass": "✅", "fail": "❌", "skip": "⏭️"}

    lines = [f"attempt {attempt} · {len(blocks)} code block(s)", ""]
    if payload.get("refused"):
        lines.append("model refused — no code emitted")
    for verdict in verdicts:
        for check in verdict.get("checks", []):
            mark = markers.get(check.get("status"), "?")
            lines.append(f"{mark} `{check.get('name')}` — {check.get('detail', '')}")
    if payload.get("validation_failed"):
        lines.append("")
        lines.append("validation budget exhausted — refusing to return code")
    if reasons:
        lines.append("")
        lines.extend(f"- {reason}" for reason in reasons)
    step = cl.Step(name=f"validation verdict · attempt {attempt}", type="run")
    await step.send()
    step.output = "\n".join(lines)
    await step.update()