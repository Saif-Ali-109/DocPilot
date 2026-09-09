"""Phase 5 FastAPI application (SPEC §7) — async backend, SSE streaming,
SQLite session/chat history.

Run::

    uvicorn docpilot.api.app:app --port 8000

Endpoints:

    GET  /api/v1/health                 {status, store, corpus}
    POST /api/v1/chat                   SSE stream (question → gate/search/
                                        token/answer/done events)
    POST /api/v1/sessions               create a chat session
    GET  /api/v1/sessions               list sessions (newest first)
    GET  /api/v1/sessions/{id}          session + messages
    DELETE /api/v1/sessions/{id}        delete session + messages (CASCADE)

Async boundary (SPEC §7): the RAG/agent core stays synchronous — every
blocking call (retrieval, LLM, SQLite) runs in a worker thread via
``asyncio.to_thread``; progressive events are bridged worker-thread → SSE
through an ``asyncio.Queue`` (``loop.call_soon_threadsafe``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docpilot.api.service import ask_events
from docpilot.api.sse import sse_line
from docpilot.api.store import SessionStore

logger = logging.getLogger(__name__)

_SENTINEL = object()


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    strategy: str | None = Field(default=None, pattern="^(auto|direct|agentic)$")
    top_k: int | None = Field(default=None, ge=1, le=20)
    language: str | None = None
    model: str | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


def create_app(
    *,
    store: SessionStore | None = None,
    engine: Callable[..., dict] | None = None,
) -> FastAPI:
    """Build the app with injectable store + ask engine (hermetic tests).

    Args:
        store: A :class:`SessionStore`; defaults to one over
            ``config.DOCPILOT_DB_PATH``.
        engine: The ask callable — must match :func:`ask_events`' signature
            (question + keyword overrides + ``emit``).  Defaults to the real
            production engine.
    """
    store = store or SessionStore()
    engine = engine or ask_events

    app = FastAPI(
        title="DocPilot API",
        version="0.1.0",
        description="Evidence-driven agentic RAG — API + UI (SPEC §7).",
    )

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    @app.get("/api/v1/health")
    async def health() -> dict:
        store_ok, corpus = await asyncio.gather(
            asyncio.to_thread(_store_probe, store),
            asyncio.to_thread(_corpus_probe),
        )
        return {
            "status": "ok",
            "api": "v1",
            "store": store_ok,
            "corpus": corpus,
        }

    # ------------------------------------------------------------------
    # chat (SSE)
    # ------------------------------------------------------------------

    @app.post("/api/v1/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            _chat_stream(body, store=store, engine=engine),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    @app.post("/api/v1/sessions")
    async def create_session(body: CreateSessionRequest | None = None) -> dict:
        title = (body or CreateSessionRequest()).title
        return await asyncio.to_thread(store.create_session, title)

    @app.get("/api/v1/sessions")
    async def list_sessions() -> dict:
        sessions = await asyncio.to_thread(store.list_sessions)
        return {"sessions": sessions}

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        session = await asyncio.to_thread(store.get_session, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = await asyncio.to_thread(store.list_messages, session_id)
        return {"session": session, "messages": messages}

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        deleted = await asyncio.to_thread(store.delete_session, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": True, "session_id": session_id}

    return app


# ---------------------------------------------------------------------------
# SSE streaming generator + persistence
# ---------------------------------------------------------------------------


async def _chat_stream(body: ChatRequest, *, store: SessionStore, engine: Callable[..., dict]):
    """Run the question in a worker thread, bridge emit events to the SSE
    stream, persist the exchange into the session store on completion."""
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=2048)
    loop = asyncio.get_running_loop()
    result: dict = {}

    def emit(data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data)

    async def worker() -> None:
        try:
            out = await asyncio.to_thread(
                engine,
                body.question,
                strategy=body.strategy,
                top_k=body.top_k,
                language=body.language,
                model=body.model,
                emit=emit,
            )
            result.update(out or {})
        except Exception as exc:  # noqa: BLE001 — surfaced to the client
            logger.exception("chat worker failed")
            try:
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001 — never raise from the error path
                pass
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    task = asyncio.create_task(worker())
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        yield sse_line(item)
        if item.get("type") == "error":
            # worker is finished after this; drain the sentinel
            continue

    await task  # propagate nothing; worker catches its own exceptions

    # Persist the exchange once the run finished (best-effort — session may
    # have been deleted mid-run; add_message is a no-op then).
    if body.session_id and result.get("answer") is not None:
        await asyncio.to_thread(store.add_message, body.session_id, "user", body.question)
        await asyncio.to_thread(
            store.add_message,
            body.session_id,
            "assistant",
            result["answer"],
            sources=result.get("sources"),
            trace=result.get("trace"),
            latency_ms=result.get("latency_ms"),
            refused=bool(result.get("refused", False)),
        )


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def _store_probe(store: SessionStore) -> bool:
    """True when the SQLite session store is reachable."""
    try:
        store.list_sessions()
        return True
    except Exception:  # noqa: BLE001 — health probe, never raises
        logger.warning("session store probe failed", exc_info=True)
        return False


def _corpus_probe() -> dict:
    """Best-effort pgvector corpus stats; never raises."""
    try:
        from docpilot.db.connection import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks")
                count = cur.fetchone()[0]
        finally:
            conn.close()
        return {"reachable": True, "chunks": count}
    except Exception:  # noqa: BLE001 — health probe, never raises
        return {"reachable": False, "chunks": None}


# Module-level app for ``uvicorn docpilot.api.app:app``.  Tests import
# ``create_app`` directly with injected store/engine and never touch this.
app = create_app()