"""Hermetic tests for Phase 6 T6: the code-route API/UI surface.

Covers :func:`docpilot.api.service.code_events` (SSE-shaped event stream:
gate → per-attempt search + validation-verdict ``code`` events → answer →
done), the ``on_event`` lifecycle hook on :func:`run_code_route`, and the
``POST /api/v1/code`` FastAPI route (streaming + session persistence).

No LLM / network / PostgreSQL: scripted generators drive the T4 loop.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docpilot.agent.code_route import run_code_route
from docpilot.api.app import create_app
from docpilot.api.service import code_events
from docpilot.api.sse import parse_sse_block
from docpilot.api.store import SessionStore
from docpilot.core.models import Chunk, RetrieverResult
from docpilot.validation import RetrieveThenValidate

QUESTION = "Write code for a minimal FastAPI app with a GET route."

_EVIDENCE = (
    "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n'
    "```"
)

_GOOD_CODE = (
    "Here is the minimal app [1]:\n\n```python\n"
    "from fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n'
    "```"
)

_BAD_CODE = (
    "[1]\n\n```python\nfrom flask import Flask\n\napp = Flask(__name__)\n```"
)

_REFUSAL = "I don't know — the available documentation does not cover this question."

VALIDATION_REFUSAL = (
    "I couldn't validate the generated code against the retrieved "
    "documentation, so I'm not returning code."
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class CodeFakeRetriever:
    def __init__(self, results: list[RetrieverResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query, top_k: int = 5, *, language: str | None = None):
        self.calls.append((query, top_k, language))
        return list(self.results)


class CodeFakeGenerator:
    """Scripted ``generate(prompt)`` — feeds the T4 loop attempt-by-attempt."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.last_usage: dict | None = {"total_tokens": 7}

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("generator exhausted — loop over-spent budget")
        return self.responses.pop(0)


def _evidence_result() -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(
            id="c1",
            content=_EVIDENCE,
            heading_path="First Steps",
            source_file="en/docs/tutorial/first-steps.md",
        ),
        score=0.91,
    )


def _event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _scripted(responses: list[str], *, turns: int | None = None) -> tuple[list[dict], dict, CodeFakeGenerator]:
    events: list[dict] = []
    generator = CodeFakeGenerator(responses)
    out = code_events(
        QUESTION,
        retriever=CodeFakeRetriever([_evidence_result()]),
        generator=generator,
        validator=RetrieveThenValidate(),
        max_validation_turns=turns,
        emit=events.append,
    )
    return events, out, generator


# ---------------------------------------------------------------------------
# code_events — event stream
# ---------------------------------------------------------------------------


def test_validated_first_attempt_streams_gate_search_code_answer_done():
    events, out, generator = _scripted([_GOOD_CODE])
    assert _event_types(events) == [
        "step",   # gate
        "search",
        "step",   # search trace
        "code",   # validation verdict
        "step",   # code_validation trace
        "answer",
        "done",
    ]

    # gate: explicit opt-in decision
    gate = events[0]["step"]
    assert gate["step"] == "gate"
    assert gate["decision"] == "code"
    assert "explicit" in gate["detail"]

    # search: debug-panel payload with the retrieved chunk
    search = events[1]
    assert search["turn"] == 1
    assert search["results"][0]["file"].endswith("first-steps.md")
    assert search["results"][0]["score"] == pytest.approx(0.91)

    # code: the validation verdict trace event (T6)
    code_event = events[3]
    assert code_event["attempt"] == 1
    assert len(code_event["code_blocks"]) == 1
    assert len(code_event["verdicts"]) == 1
    checks = {c["name"]: c["status"] for c in code_event["verdicts"][0]["checks"]}
    assert checks == {"parse": "pass", "imports_evidence": "pass", "symbols_evidence": "pass"}
    assert code_event["passed"] is True
    assert code_event["validation_failed"] is False
    assert code_event["refused"] is False
    assert code_event["reasons"] == []

    code_step = events[4]["step"]
    assert code_step["step"] == "code_validation"
    assert code_step["decision"] == "validated"

    # answer: citation-formatted display with code-route payloads
    answer = events[-2]
    assert answer["type"] == "answer"
    assert answer["code"] is True
    assert "first-steps.md" in answer["text"]
    assert answer["sources"][0]["kind"] == "tutorial"
    assert answer["validation_failed"] is False
    assert answer["validation_turns"] == 1
    assert answer["verdict"]["passed"] is True

    done = events[-1]
    assert {t["step"] for t in done["trace"]} == {"gate", "search", "code_validation"}
    assert done["validation_turns"] == 1
    assert done["usage"] == generator.last_usage

    assert out["code"] is True
    assert out["answer"] == answer["text"]
    assert not out["validation_failed"]


def test_validation_failure_refuses_with_sources_after_budget():
    events, out, _ = _scripted([_BAD_CODE], turns=0)
    answer = events[-2]
    assert answer["code"] is True
    assert answer["validation_failed"] is True
    assert answer["text"].startswith(VALIDATION_REFUSAL)
    # the retrieved sources travel with the refusal
    assert "Sources:" in answer["text"]
    assert "first-steps.md" in answer["text"]
    assert any("Flask" in r for r in answer["validation_reasons"])

    code_event = events[3]
    assert code_event["passed"] is False
    assert code_event["reasons"] == ["import(s) not grounded in retrieved evidence: Flask"]

    out_step = events[4]["step"]
    assert out_step["decision"] == "failed"

    done = events[-1]
    assert done["validation_failed"] is True
    assert done["validation_turns"] == 1
    assert out["validation_failed"] is True
    # the failed code is never the returned answer
    assert "flask" not in out["answer"].lower()


def test_model_refusal_emits_single_no_code_attempt():
    events, out, _ = _scripted([_REFUSAL])
    code_event = next(e for e in events if e["type"] == "code")
    assert code_event["refused"] is True
    assert code_event["code_blocks"] == []
    assert code_event["passed"] is None
    code_step = next(e["step"] for e in events if e["type"] == "step"
                     and e["step"]["step"] == "code_validation")
    assert code_step["decision"] == "no_code"
    answer = events[-2]
    assert answer["refused"] is True
    assert not answer["validation_failed"]
    assert answer["text"].startswith("I don't know")
    assert _event_types(events).count("code") == 1


def test_failed_attempt_reformulates_then_passes():
    events, out, generator = _scripted([_BAD_CODE, _GOOD_CODE], turns=1)
    code_events_seen = [e for e in events if e["type"] == "code"]
    assert len(code_events_seen) == 2
    assert [e["passed"] for e in code_events_seen] == [False, True]
    searches = [e for e in events if e["type"] == "search"]
    assert len(searches) == 2  # re-retrieval per attempt is visible
    answer = events[-2]
    assert answer["validation_turns"] == 2
    assert not answer["validation_failed"]
    # the reformulation carried the failure reasons back
    assert "VALIDATION FAILURES FROM THE PREVIOUS ATTEMPT" in generator.prompts[1]
    assert "Flask" in generator.prompts[1]


# ---------------------------------------------------------------------------
# run_code_route on_event lifecycle hook
# ---------------------------------------------------------------------------


def test_run_code_route_on_event_hook_kinds():
    seen: list[tuple[str, dict]] = []
    result = run_code_route(
        QUESTION,
        explicit_code=True,
        retriever=CodeFakeRetriever([_evidence_result()]),
        generator=CodeFakeGenerator([_GOOD_CODE]),
        validator=RetrieveThenValidate(),
        on_event=lambda kind, payload: seen.append((kind, payload)),
    )
    assert [k for k, _ in seen] == ["gated", "attempt"]
    assert seen[0][0] == "gated" and seen[0][1]["code"] is True
    assert seen[1][0] == "attempt" and seen[1][1]["turn"] == 1
    assert result.took_code_route
    assert result.code.generation_attempts == 1


def test_gated_event_reflects_fallback_when_not_armed_and_not_explicit():
    class _FallbackGenerator:
        """Supports both call styles: code route (``generate``) and the
        standard ask path (``generate_answer``) that the fallback runs."""

        def generate(self, prompt: str) -> str:
            return _GOOD_CODE

        def generate_answer(self, context, sources, question) -> str:
            return _GOOD_CODE

    seen: list[tuple[str, dict]] = []
    run_code_route(
        "not a code question",
        enabled=False,
        explicit_code=False,
        retriever=CodeFakeRetriever([_evidence_result()]),
        generator=_FallbackGenerator(),
        on_event=lambda kind, payload: seen.append((kind, payload)),
    )
    assert seen[0][0] == "gated"
    assert seen[0][1]["code"] is False
    assert "not armed" in seen[0][1]["reason"]
    assert len(seen) == 1  # no attempts on the fallback path


# ---------------------------------------------------------------------------
# FastAPI app — POST /api/v1/code
# ---------------------------------------------------------------------------


def _scripted_code_engine(question: str, *, top_k=None, language=None, model=None,
                          emit=None) -> dict:
    del top_k, language, model
    source = {"ref": 1, "file": "en/docs/tutorial/first-steps.md",
              "heading": "First Steps", "kind": "tutorial"}
    emit({"type": "step", "step": {"step": "gate", "query": question,
                                   "decision": "code", "detail": "explicit code opt-in",
                                   "latency_ms": 1}})
    emit({"type": "search", "turn": 1, "query": question,
          "results": [{"file": "en/docs/tutorial/first-steps.md", "heading": "First Steps",
                       "kind": "tutorial", "score": 0.9, "excerpt": "app = FastAPI()"}]})
    emit({"type": "code", "attempt": 1, "code_blocks": ["app = FastAPI()"],
          "verdicts": [{"passed": True, "checks": [
              {"name": "parse", "status": "pass", "detail": "parses cleanly"}]}],
          "passed": True, "validation_failed": False, "refused": False,
          "reasons": [], "attempt_latency_ms": 2.5})
    emit({"type": "answer", "text": "Here is the app [1]:\n\n```python\napp = FastAPI()\n```",
          "sources": [source], "refused": False, "code": True,
          "validation_failed": False, "validation_reasons": [],
          "validation_turns": 1, "verdict": {"passed": True, "checks": []}})
    emit({"type": "done", "trace": [], "latency_ms": 5.0, "usage": None,
          "validation_turns": 1, "validation_failed": False})
    return {
        "question": question,
        "answer": "Here is the app [1]:\n\n```python\napp = FastAPI()\n```",
        "sources": [source], "refused": False, "code": True, "trace": [],
        "latency_ms": 5.0, "usage": None, "validation_turns": 1,
        "validation_failed": False, "verdict": {"passed": True, "checks": []},
    }


def _client(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
    app = create_app(store=store, code_engine=_scripted_code_engine)
    return TestClient(app), store


class TestCodeEndpoint:
    def test_streams_events_and_persists(self, tmp_path) -> None:
        client, store = _client(tmp_path)
        session = store.create_session(title="code test")
        with client.stream("POST", "/api/v1/code", json={
            "question": "write code", "session_id": session["id"],
        }) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            blocks = [b for b in resp.iter_lines() if b]
            parsed = [parse_sse_block(b) for b in blocks if parse_sse_block(b)]
            assert [p["type"] for p in parsed] == ["step", "search", "code", "answer", "done"]
            assert parsed[2]["type"] == "code"
            assert parsed[2]["attempt"] == 1
            assert parsed[2]["passed"] is True
            assert parsed[-2]["code"] is True

        messages = store.list_messages(session["id"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert "app = FastAPI()" in messages[1]["content"]
        assert messages[1]["refused"] is False

    def test_without_session_id_no_persist(self, tmp_path) -> None:
        client, store = _client(tmp_path)
        with client.stream("POST", "/api/v1/code", json={"question": "write code"}) as resp:
            assert resp.status_code == 200
            for _b in resp.iter_lines():
                pass
        assert store.list_sessions() == []

    def test_empty_question_rejected(self, tmp_path) -> None:
        client, _ = _client(tmp_path)
        assert client.post("/api/v1/code", json={"question": ""}).status_code == 422

    def test_bad_top_k_rejected(self, tmp_path) -> None:
        client, _ = _client(tmp_path)
        resp = client.post("/api/v1/code", json={"question": "hi", "top_k": 0})
        assert resp.status_code == 422

    def test_engine_failure_emits_error_event(self, tmp_path) -> None:
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))

        def failing_engine(question: str, *, top_k=None, language=None, model=None,
                           emit=None) -> dict:
            del top_k, language, model, emit
            raise RuntimeError("code validator crashed")

        app = create_app(store=store, code_engine=failing_engine)
        client = TestClient(app)
        with client.stream("POST", "/api/v1/code", json={"question": "write code"}) as resp:
            assert resp.status_code == 200
            blocks = [b for b in resp.iter_lines() if b]
            parsed = [parse_sse_block(b) for b in blocks if parse_sse_block(b)]
            assert [p["type"] for p in parsed] == ["error"]
            assert "code validator crashed" in parsed[0]["message"]