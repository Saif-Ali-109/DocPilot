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

from docpilot.api.service import ask_events

logger = logging.getLogger(__name__)

_SENTINEL = object()

# step.event["step"] → (cl.Step label, cl.Step type)
_STEP_LABELS = {
    "gate": ("gate", "run"),
    "search": ("search", "retrieval"),
    "judge": ("judge", "llm"),
    "tool_call": ("tool_call", "tool"),
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
            "chunks, scores, judge decisions and tool calls are shown as steps."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()

    def emit(data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    def run() -> dict:
        try:
            return ask_events(message.content, strategy="auto", emit=emit)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    task = asyncio.create_task(asyncio.to_thread(run))

    final_text: str = ""
    final_sources: list[dict] = []
    refused = False

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

        elif event_type == "answer":
            final_text = item.get("text", "")
            final_sources = item.get("sources") or []
            refused = bool(item.get("refused"))

        elif event_type == "done":
            pass  # finalize below

        elif event_type == "error":
            await answer_msg.update()
            answer_msg.content = f"⚠️ {item.get('message', 'unknown error')}"
            await answer_msg.update()
            break

    await task

    # Authoritative final render: the streamed body is replaced by the full
    # citation-formatted display (answer + source footer).
    if final_text:
        answer_msg.content = final_text
        await answer_msg.update()
        if not refused and final_sources:
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