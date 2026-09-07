"""LangGraph wiring for the agentic retrieval loop (SPEC §4, PLAN §3.1/§3.2).

Builds a ``StateGraph`` over :class:`docpilot.agent.types.AgentLoopState` from
injected components. The graph is deliberately minimal and idiomatic:

    START → retrieve → judge →{ sufficient → answer → END
                             { insufficient & attempts < max → retrieve (loop back)
                             { insufficient & attempts ≥ max → refuse → END

The *gate* lives in :mod:`docpilot.agent.pipeline_agentic` (the pipeline routes
to the fast path directly and always records the gate trace step), so this
module only encodes the conditional agentic loop.

Hard rules honoured here:
    * **No vendor calls** — every LLM/DB interaction goes through the injected
      ``retriever``, ``judge``, ``generator`` and ``citation_engine``.
    * **Budget hard-enforced at the graph level** — the conditional edge after
      ``judge`` refuses once ``attempts >= max_retries``.
    * **Our ``LoopTraceStep`` only** — no LangGraph/LangSmith tracing, no
      third-party trace hooks. Per-step latency is collected via
      ``LoopTraceStep.new(started_at=...)`` for inspectability only.

State serialisation: ``AgentLoopState`` stores ``results`` / ``sources`` /
``trace`` as plain JSON-ish dicts (the contract forbids custom classes in
state so the graph can checkpoint without vendor-specific serialisation).
Helper functions near the top of this module convert ``RetrieverResult`` /
``SourceRef`` / ``LoopTraceStep`` to and from those dict forms;
:mod:`docpilot.agent.pipeline_agentic` reuses them to build the final
:class:`AgentResult`.
"""

from __future__ import annotations

import logging
import time

from docpilot import config
from docpilot.agent.types import (
    DEFAULT_MAX_RETRIES,
    AgentLoopState,
    LoopTraceStep,
)
from docpilot.core.models import Chunk, RetrieverResult, SourceRef
from docpilot.generation.prompts import SYSTEM_PROMPT, format_sources
from docpilot.pipeline_ask import _NO_CONTEXT_NOTE
from docpilot.agent.prompts import REFUSE_ANSWER

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
    return {"ref": s.ref, "file": s.file, "heading": s.heading}


def dict_to_source(d: dict) -> SourceRef:
    """Rebuild a :class:`SourceRef` from a graph-state dict."""
    return SourceRef(ref=d["ref"], file=d["file"], heading=d.get("heading"))


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
) -> dict[str, object]:
    """Build the graph's node callables bound to the injected components.

    Each returned callable takes an :class:`AgentLoopState` dict and returns a
    partial state update. Nodes are closed over the injected component
    *instances* so the compiled graph never constructs vendor objects itself.
    """

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
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        return {
            "results": [result_to_dict(r) for r in results],
            "sources": [source_to_dict(s) for s in sources],
            "trace": trace,
        }

    def judge_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        attempts: int = state["attempts"] + 1
        query_used: str = state["current_query"]
        results = [dict_to_result(d) for d in state["results"]]

        judgment = judge.judge(state["original_question"], results, query_used)
        logger.debug(
            "Agent judge (attempt %d, query=%r): verdict=%r",
            attempts,
            query_used,
            judgment.verdict,
        )

        detail = f"verdict={judgment.verdict}; reason={judgment.reason}"
        if judgment.reformulated_query:
            detail += f"; reformulated_query={judgment.reformulated_query}"
        decision = (
            "sufficient"
            if judgment.verdict == "sufficient"
            else f"insufficient/retry-{attempts}"
        )
        step = LoopTraceStep.new("judge", query_used, decision, detail=detail, started_at=started)
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        update: dict = {
            "attempts": attempts,
            "trace": trace,
        }
        if judgment.verdict == "insufficient" and judgment.reformulated_query:
            # Feed the reformulation to the next retrieve round (in-contract
            # state field — no extra keys needed for the query itself).
            update["current_query"] = judgment.reformulated_query
        return update

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

        sources_text = format_sources(sources)
        raw_response = generator.generate_answer(
            context_text, sources_text, state["original_question"]
        )
        answer_text, footer = citation_engine.format_answer(raw_response, sources)
        display = f"{answer_text}\n\n{footer}" if footer else answer_text

        step = LoopTraceStep.new(
            "answer", state["current_query"], "answer", started_at=started
        )
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]

        logger.debug("Agent answer node produced %d source(s)", len(sources))
        return {"answer": display, "trace": trace}

    def refuse_node(state: AgentLoopState) -> dict:
        started = time.perf_counter()
        step = LoopTraceStep.new(
            "refuse", state["current_query"], "refuse", started_at=started
        )
        trace: list[dict] = list(state["trace"]) + [trace_step_to_dict(step)]
        logger.debug("Agent refuse node: budget exhausted (attempts=%d)", state["attempts"])
        return {"answer": REFUSE_ANSWER, "refused": True, "trace": trace}

    return {
        "retrieve": retrieve_node,
        "judge": judge_node,
        "answer": answer_node,
        "refuse": refuse_node,
    }


# ---------------------------------------------------------------------------
# Conditional routing after judge
# ---------------------------------------------------------------------------


def _route_after_judge(max_retries: int):
    """Return the conditional-edge router bound to the budget.

    The judge node stores its verdict in the *latest* trace step (the trace is
    a declared ``AgentLoopState`` field, so it survives LangGraph's schema
    merge — an undeclared verdict key would be silently dropped). The returned
    callable reads that last judge decision and returns the next node name:
    ``"answer"`` (sufficient), ``"refuse"`` (budget exhausted) or
    ``"retrieve"`` (reformulate / retry).
    """

    def route(state: AgentLoopState) -> str:
        latest_decision = ""
        for t in reversed(state["trace"]):
            if t.get("step") == "judge":
                latest_decision = t.get("decision", "")
                break
        if latest_decision == "sufficient":
            return "answer"
        if state["attempts"] >= max_retries:
            return "refuse"
        return "retrieve"

    return route


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

    Returns:
        A compiled LangGraph ``StateGraph`` app. Invoking it with an
        :class:`AgentLoopState` runs retrieve → judge → (answer | refuse |
        retrieve…) and returns the final state dict.
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
    )

    builder = StateGraph(AgentLoopState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "judge")
    builder.add_conditional_edges(
        "judge",
        _route_after_judge(max_retries),
        {"answer": "answer", "refuse": "refuse", "retrieve": "retrieve"},
    )
    builder.add_edge("answer", END)
    builder.add_edge("refuse", END)
    return builder.compile()
