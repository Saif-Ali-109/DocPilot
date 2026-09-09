"""LangGraph wiring for the agentic retrieval loop (SPEC §4/§5, PLAN §3.1/§3.2).

Builds a ``StateGraph`` over :class:`docpilot.agent.types.AgentLoopState` from
injected components. The graph is deliberately minimal and idiomatic:

    START → retrieve → judge →{ sufficient → answer → END
                             {
                             { insufficient & needs_tool & attempts < max
                             {     → tool_call →{ ok → answer → END
                             {                { error → refuse → END
                             {
                             { insufficient/retry & attempts < max → retrieve (loop back)
                             { insufficient & attempts ≥ max → refuse → END

The *gate* lives in :mod:`docpilot.agent.pipeline_agentic` (the pipeline routes
to the fast path directly and always records the gate trace step), so this
module only encodes the conditional agentic loop.

Phase 3 (SPEC §5): ``build_graph(..., tool=...)`` optionally wires a
:class:`~docpilot.tools.base.Tool` (e.g. ``GitHubTool``) into the loop. The
judge is told whether tools are available; when it judges ``insufficient``
*and* asks for a tool, the graph runs exactly one tool call and routes its
outcome to ``answer`` (evidence) or ``refuse`` (tool error) — the tool never
loops back to the judge, and judge-attempt budgeting is unchanged. With
``tool=None`` the Phase 2 behaviour is byte-identical: a defensive
``needs_tool`` request is treated as plain insufficient.

Hard rules honoured here:
    * **No vendor calls** — every LLM/DB/tool interaction goes through the
      injected ``retriever``, ``judge``, ``generator``, ``citation_engine``
      and ``tool``.
    * **Budget hard-enforced at the graph level** — the conditional edge after
      ``judge`` refuses once ``attempts >= max_retries`` (a needs_tool request
      on the last permitted round refuses without calling the tool).
    * **Our ``LoopTraceStep`` only** — no LangGraph/LangSmith tracing, no
      third-party trace hooks. Per-step latency is collected via
      ``LoopTraceStep.new(started_at=...)`` for inspectability only.

State serialisation: ``AgentLoopState`` stores ``results`` / ``sources`` /
``trace`` / ``tool_results`` as plain JSON-ish dicts (the contract forbids
custom classes in state so the graph can checkpoint without vendor-specific
serialisation).  Helper functions near the top of this module convert
``RetrieverResult`` / ``SourceRef`` / ``LoopTraceStep`` / ``ToolResult`` to
and from those dict forms; :mod:`docpilot.agent.pipeline_agentic` reuses them
to build the final :class:`AgentResult`.
"""

from __future__ import annotations

import logging
import time

from docpilot import config
from docpilot.agent.prompts import REFUSE_ANSWER
from docpilot.agent.types import (
    DEFAULT_MAX_RETRIES,
    AgentLoopState,
    LoopTraceStep,
    dict_to_tool_result,
    tool_result_to_dict,
)
from docpilot.core.models import Chunk, RetrieverResult, SourceRef, derive_source_kind
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources
from docpilot.pipeline_ask import _NO_CONTEXT_NOTE
from docpilot.tools import ToolRequest, ToolResult

logger = logging.getLogger(__name__)

# langgraph is imported lazily inside build_graph() — this module (and the
# whole docpilot.agent package) stays importable without it, matching the
# shared contract's design (agent/types.py declares no langgraph dependency).


# ---------------------------------------------------------------------------
# Serialisation helpers (shared with pipeline_agentic)
# ---------------------------------------------------------------------------


def result_to_dict(r: RetrieverResult) -> dict:
    """Serialise a :class:`RetrieverResult` to a plain dict for graph state."""
    return {
        "content": r.chunk.content,
        "score": r.score,
        "source_file": r.chunk.source_file,
        "heading": r.chunk.heading_path,
    }


def dict_to_result(d: dict) -> RetrieverResult:
    """Rebuild a :class:`RetrieverResult` from a graph-state dict."""
    chunk = Chunk(
        id="",
        content=d.get("content", ""),
        heading_path=d.get("heading"),
        source_file=d.get("source_file", ""),
    )
    return RetrieverResult(chunk=chunk, score=d.get("score", 0.0))


def source_to_dict(s: SourceRef) -> dict:
    """Serialise a :class:`SourceRef` to a plain dict for graph state."""
    return {"ref": s.ref, "file": s.file, "heading": s.heading, "kind": s.kind}


def dict_to_source(d: dict) -> SourceRef:
    """Rebuild a :class:`SourceRef` from a graph-state dict."""
    return SourceRef(
        ref=d["ref"],
        file=d["file"],
        heading=d.get("heading"),
        kind=d.get("kind"),
    )


def trace_step_to_dict(step: LoopTraceStep) -> dict:
    """Serialise a :class:`LoopTraceStep` to a plain dict for graph state / JSON."""
    return {
        "step": step.step,
        "query": step.query,
        "decision": step.decision,
        "detail": step.detail,
        "latency_ms": step.latency_ms,
    }


def dict_to_trace_step(d: dict) -> LoopTraceStep:
    """Rebuild a :class:`LoopTraceStep` from a graph-state dict."""
    return LoopTraceStep(
        step=d.get("step", ""),
        query=d.get("query", ""),
        decision=d.get("decision", ""),
        detail=d.get("detail"),
        latency_ms=d.get("latency_ms"),
    )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_nodes(
    *,
    retriever,
    judge,
    generator,
    citation_engine,
    top_k: int,
    language: str | None,
    tool=None,
    emit=None,
    judge_skip_score: float = 0.0,
) -> dict[str, object]:
    """Build the graph's node callables bound to the injected components.

    Each returned callable takes an :class:`AgentLoopState` dict and returns a
    partial state update. Nodes are closed over the injected component
    *instances* so the compiled graph never constructs vendor objects itself.

    Args:
        tool: An optional :class:`~docpilot.tools.base.Tool` (Phase 3). When
            ``None`` no ``tool_call`` node is produced and the judge is told
            tools are unavailable — the graph is Phase 2-identical.
        emit: An optional live-view callback ``emit(event: dict)`` (Phase 5,
            SPEC §7).  ``None`` → zero behaviour change (the CLI/eval never
            pass one).  When set, every node emits ``{"type": "step", "step":
            <trace-step dict>}`` as it completes, and the answer node streams
            answer deltas as ``{"type": "token", "delta": ...}`` when the
            generator supports streaming.
        judge_skip_score: Phase 5 hardening (SPEC §7).  ``0.0`` (default) →
            the judge always runs after retrieve (Phase 2/4 behaviour).  When
            ``> 0``, the retrieve node marks ``skip_judge`` when the top
            retrieval score clears the threshold; the router answers directly
            without the judge LLM call.
    """

    def _emit_step(step: LoopTraceStep) -> None:
        if emit is not None:
            emit({"type": "step", "step": trace_step_to_dict(step)})

    def retrieve_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        query: str = state["current_query"]
        turn = sum(1 for t in state["trace"] if t.get("step") == "search") + 1
        logger.debug(
            "Agent retrieve: turn=%s top_k=%s language=%s",
            turn,
            top_k,
            language,
        )
        results = retriever.retrieve(query, top_k=top_k, language=language)

        sources = [
            SourceRef(
                ref=i + 1,
                file=r.chunk.source_file,
                heading=r.chunk.heading_path or None,
                kind=derive_source_kind(r.chunk.source_file),
            )
            for i, r in enumerate(results)
        ]
        logger.debug("Agent retrieve (query=%r): %d result(s)", query, len(results))

        top_scores = [round(r.score, 4) for r in results[:3]]
        detail = (
            f"turn={turn}: retrieved {len(results)} chunks; "
            f"top scores: {top_scores}"
        )
        step = LoopTraceStep.new("search", query, "retrieved", detail=detail, started_at=started)
        _emit_step(step)
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        # Phase 5 judge skip: when a threshold is configured and the strongest
        # retrieved chunk clears it, skip the judge LLM call (evidence is
        # plainly sufficient).  Stored in state so the router + answer node
        # can act on it without an extra edge.
        skip_judge = bool(
            judge_skip_score > 0
            and results
            and results[0].score >= judge_skip_score
        )
        return {
            "results": [result_to_dict(r) for r in results],
            "sources": [source_to_dict(s) for s in sources],
            "trace": trace,
            "skip_judge": skip_judge,
        }

    def judge_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        attempts: int = state["attempts"] + 1
        query_used: str = state["current_query"]
        results = [dict_to_result(d) for d in state["results"]]

        tools_available = tool is not None
        judgment = judge.judge(
            state["original_question"],
            results,
            query_used,
            tools_available=tools_available,
        )
        logger.debug(
            "Agent judge (attempt %d, query=%r): verdict=%r needs_tool=%r",
            attempts,
            query_used,
            judgment.verdict,
            judgment.needs_tool,
        )

        # needs_tool fires only when a tool is actually wired AND the judge
        # delivered a well-formed request. A (defensive) needs_tool with
        # tool=None is treated as plain insufficient — never call a tool that
        # does not exist.
        needs_tool = bool(
            tools_available
            and judgment.verdict == "insufficient"
            and judgment.needs_tool
            and judgment.tool_request is not None
        )
        # Stash the validated request for the tool_call node. Always reset
        # every round so a stale request can never fire late.
        tool_request = judgment.tool_request if needs_tool else None

        detail = f"verdict={judgment.verdict}; reason={judgment.reason}"
        if needs_tool:
            decision = "insufficient/needs-tool"
            detail += f"; tool_request={tool_request}"
        else:
            if judgment.reformulated_query:
                detail += f"; reformulated_query={judgment.reformulated_query}"
            decision = (
                "sufficient"
                if judgment.verdict == "sufficient"
                else f"insufficient/retry-{attempts}"
            )
        step = LoopTraceStep.new("judge", query_used, decision, detail=detail, started_at=started)
        _emit_step(step)
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        update: dict = {
            "attempts": attempts,
            "trace": trace,
            "tool_request": tool_request,
        }
        if (
            judgment.verdict == "insufficient"
            and judgment.reformulated_query
            and not needs_tool
        ):
            # Feed the reformulation to the next retrieve round (in-contract
            # state field — no extra keys needed for the query itself).
            update["current_query"] = judgment.reformulated_query
        return update

    def tool_call_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        query: str = state["current_query"]
        request_dict: dict | None = state.get("tool_request")
        trace_so_far: list[dict] = list(state["trace"])

        if tool is None or not request_dict:
            # Defensive — the judge router guards this edge, but a stale or
            # malformed request must never crash the graph.
            message = "tool_call reached without a tool_request"
            step = LoopTraceStep.new(
                "tool_call", query, "tool_error", detail=message, started_at=started
            )
            _emit_step(step)
            return {
                "tool_error": message,
                "tool_request": None,
                "trace": trace_so_far + [trace_step_to_dict(step)],
            }

        request = ToolRequest(
            name=str(request_dict["name"]),
            params=dict(request_dict.get("params") or {}),
        )
        try:
            result = tool.execute(request)
        except Exception as exc:  # noqa: BLE001 — tool boundary: never raises
            logger.debug("Agent tool_call threw unexpectedly: %s", exc)
            result = ToolResult(
                ok=False,
                summary="",
                error=f"tool {request.name!r} raised: {exc}",
            )

        if result.ok:
            # Append each tool item's per-record source as a continuing
            # SourceRef. The last retrieve node replaced state["sources"], so
            # len(state["sources"]) == k — exactly the [1..k] chunks the
            # answer node will number — and the refs continue [k+1..].
            existing = len(state["sources"])
            sources: list[dict] = list(state["sources"])
            for idx, item in enumerate(result.items):
                if not isinstance(item, dict):
                    continue
                label = item.get("source_label")
                if not label:
                    continue
                sources.append(
                    source_to_dict(
                        SourceRef(
                            ref=existing + idx + 1,
                            file=str(label),
                            heading=item.get("title")
                            or item.get("message_first_line")
                            or None,
                            kind=derive_source_kind(str(label)),
                        )
                    )
                )
            detail = f"{request.name} -> ok: {len(result.items)} item(s)"
            step = LoopTraceStep.new(
                "tool_call", query, "tool_call", detail=detail, started_at=started
            )
            _emit_step(step)
            return {
                "sources": sources,
                "tool_results": list(state.get("tool_results", []))
                + [tool_result_to_dict(result)],
                "tool_error": None,
                "trace": trace_so_far + [trace_step_to_dict(step)],
            }

        # ok=False → tool_error; the conditional edge routes to refuse.
        message = result.error or "tool failed"
        detail = f"{request.name} -> error: {message}"
        step = LoopTraceStep.new(
            "tool_call", query, "tool_error", detail=detail, started_at=started
        )
        _emit_step(step)
        return {
            "tool_error": message,
            "trace": trace_so_far + [trace_step_to_dict(step)],
        }

    def answer_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        results = [dict_to_result(d) for d in state["results"]]
        sources = [dict_to_source(d) for d in state["sources"]]

        # Mirror pipeline_ask.ask's post-retrieval construction exactly
        # (SPEC §3.9): numbered context + SYSTEM_PROMPT + generate_answer +
        # StandardCitationEngine cite/format + footer composition.
        if results:
            context_text = "\n\n".join(
                f"[{i + 1}] {r.chunk.content}" for i, r in enumerate(results)
            )
        else:
            context_text = _NO_CONTEXT_NOTE

        # Phase 3: when the graph called a tool, append a clearly separated
        # LIVE GITHUB EVIDENCE section whose entries continue the numbering
        # after the retrieved chunks ([k+1]…) so the generator can cite
        # them directly. The tool SourceRefs were already appended by the
        # tool_call node, so the footer's [n] markers line up automatically.
        tool_results = [dict_to_tool_result(d) for d in state.get("tool_results", [])]
        if tool_results:
            offset = len(results)
            parts: list[str] = [context_text, "LIVE GITHUB EVIDENCE:"]
            parts += [
                f"[{offset + i + 1}] {tr.summary}"
                for i, tr in enumerate(tool_results)
            ]
            context_text = "\n\n".join(parts)

        sources_text = format_sources(sources)

        # Phase 5 streaming (SPEC §7): when the caller passed an emit hook and
        # the generator supports streaming, stream the answer deltas live
        # (token events) and use the buffered text as raw_response.  Without a
        # hook this is byte-identical Phase 2/4: one generate_answer call.
        raw_response = None
        if emit is not None and callable(
            getattr(generator, "generate_answer_stream", None)
        ):
            try:
                parts: list[str] = []
                for delta in generator.generate_answer_stream(
                    context_text, sources_text, state["original_question"]
                ):
                    parts.append(delta)
                    emit({"type": "token", "delta": delta})
                raw_response = "".join(parts)
            except NotImplementedError:
                raw_response = None
        if raw_response is None:
            raw_response = generator.generate_answer(
                context_text, sources_text, state["original_question"]
            )

        answer_text, footer = citation_engine.format_answer(raw_response, sources)
        display = f"{answer_text}\n\n{footer}" if footer else answer_text

        detail = "answer"
        if state.get("skip_judge"):
            detail = "answer (judge skipped: top score cleared the threshold)"
        step = LoopTraceStep.new(
            "answer", state["current_query"], "answer", detail=detail, started_at=started
        )
        _emit_step(step)
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        logger.debug("Agent answer node produced %d source(s)", len(sources))
        return {"answer": display, "trace": trace}

    def refuse_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        step = LoopTraceStep.new(
            "refuse", state["current_query"], "refuse", started_at=started
        )
        _emit_step(step)
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]
        logger.debug("Agent refuse node: budget exhausted (attempts=%d)", state["attempts"])
        return {"answer": REFUSE_ANSWER, "refused": True, "trace": trace}

    nodes: dict[str, object] = {
        "retrieve": retrieve_node,
        "judge": judge_node,
        "answer": answer_node,
        "refuse": refuse_node,
    }
    if tool is not None:
        nodes["tool_call"] = tool_call_node
    return nodes


# ---------------------------------------------------------------------------
# Conditional routing after judge
# ---------------------------------------------------------------------------


def _route_after_retrieve(state: AgentLoopState) -> str:
    """Route after retrieve: answer directly when the judge was skipped
    (Phase 5 hardening — top retrieval score cleared ``AGENT_JUDGE_SKIP_MIN_
    SCORE``), else run the judge.  With the skip disabled the retrieve node
    never sets ``skip_judge``, so this edge is behaviour-identical to the
    Phase 2 retrieve → judge edge."""
    return "answer" if state.get("skip_judge") else "judge"


def _route_after_judge(max_retries: int, tool=None):
    """Return the conditional-edge router bound to the budget + tool.

    The judge node stores its verdict in the *latest* trace step (the trace is
    a declared ``AgentLoopState`` field, so it survives LangGraph's schema
    merge — an undeclared verdict key would be silently dropped). The returned
    callable reads that last judge decision and returns the next node name:
    ``"answer"`` (sufficient), ``"tool_call"`` (a stashed within-budget tool
    request on an insufficient verdict), ``"refuse"`` (budget exhausted) or
    ``"retrieve"`` (reformulate / retry).

    Ordering matters: when ``needs_tool`` fires on the *last* permitted round
    (``attempts >= max_retries``) the budget check wins and the graph refuses
    without ever calling the tool.  When ``tool`` is ``None`` a defensive
    ``needs_tool`` is plain insufficient — the request was never stashed, so
    the router falls through to ``retrieve``/``refuse`` as in Phase 2.
    """

    def route(state: AgentLoopState) -> str:
        latest_decision = ""
        for t in reversed(state["trace"]):
            if t.get("step") == "judge":
                latest_decision = t.get("decision", "")
                break
        if latest_decision == "sufficient":
            return "answer"
        if (
            state.get("tool_request")
            and tool is not None
            and state["attempts"] < max_retries
        ):
            return "tool_call"
        if state["attempts"] >= max_retries:
            return "refuse"
        return "retrieve"

    return route


def _route_after_tool_call(state: AgentLoopState) -> str:
    """Route after the tool_call node: success → answer; failure → refuse.

    The tool never loops back to the judge — its evidence (or its error) is
    final for this question.
    """
    return "refuse" if state.get("tool_error") else "answer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    *,
    retriever,
    judge,
    generator,
    citation_engine,
    top_k: int | None = None,
    language: str | None = None,
    max_retries: int | None = None,
    tool=None,
    emit=None,
    judge_skip_score: float = 0.0,
):
    """Build and compile the agentic loop over the injected components.

    Args:
        retriever: A ``Retriever`` (reuses ``SimpleRetriever`` in production).
        judge: A ``SufficiencyJudge`` (one LLM call per invocation).
        generator: A ``Generator`` for final answer construction.
        citation_engine: A ``CitationEngine`` for inline markers + footer.
        top_k: Retrieval count for the loop; defaults to
            ``config.AGENT_LOOP_TOP_K`` (broader than the fast path).
        language: Optional retrieval language filter (``None`` = no filter).
        max_retries: Hard budget for judge calls; defaults to
            ``config.AGENT_MAX_RETRIES`` (falling back to
            ``DEFAULT_MAX_RETRIES``).
        tool: Optional Phase 3 ``Tool`` (e.g. ``GitHubTool``). When ``None``
            the graph is exactly Phase 2: the judge is told tools are
            unavailable and no ``tool_call`` node exists.
        emit: Optional live-view event callback (Phase 5, SPEC §7 — see
            :func:`make_nodes`).  ``None`` → zero behaviour change.
        judge_skip_score: Phase 5 hardening threshold (see
            :func:`make_nodes`); ``0.0`` (default) → judge always runs.

    Returns:
        A compiled LangGraph ``StateGraph`` app. Invoking it with an
        :class:`AgentLoopState` runs retrieve → judge → (tool_call | answer |
        refuse | retrieve…) and returns the final state dict.
    """
    top_k = top_k if top_k is not None else config.AGENT_LOOP_TOP_K
    max_retries = max_retries if max_retries is not None else (
        config.AGENT_MAX_RETRIES or DEFAULT_MAX_RETRIES
    )

    # Lazy langgraph import: the graph is the only module that needs it, so
    # the rest of the package stays importable without langgraph installed.
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(
        retriever=retriever,
        judge=judge,
        generator=generator,
        citation_engine=citation_engine,
        top_k=top_k,
        language=language,
        tool=tool,
        emit=emit,
        judge_skip_score=judge_skip_score,
    )

    builder = StateGraph(AgentLoopState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"answer": "answer", "judge": "judge"},
    )
    judge_paths: dict[str, str] = {
        "answer": "answer",
        "refuse": "refuse",
        "retrieve": "retrieve",
    }
    if tool is not None:
        # Only wire the judge → tool_call branch when the node exists; the
        # router never returns "tool_call" without a tool, and LangGraph
        # refuses a path map targeting a missing node.
        judge_paths["tool_call"] = "tool_call"
        # tool_call → answer (evidence obtained) | refuse (tool error). The
        # tool never loops back to the judge.
        builder.add_conditional_edges(
            "tool_call",
            _route_after_tool_call,
            {"answer": "answer", "refuse": "refuse"},
        )
    builder.add_conditional_edges("judge", _route_after_judge(max_retries, tool), judge_paths)
    builder.add_edge("answer", END)
    builder.add_edge("refuse", END)
    return builder.compile()
