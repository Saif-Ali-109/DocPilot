"""Hermetic tests for the Phase 5 API layer (SPEC §7): the event service
(ask_events) and the FastAPI app (SSE chat + SQLite sessions).

No network / PostgreSQL / live LLM: fakes are injected for retriever,
generator, judge and the ask engine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docpilot.api.app import create_app
from docpilot.api.service import ask_events
from docpilot.api.sse import parse_sse_block
from docpilot.api.store import SessionStore
from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.models import Chunk, RetrieverResult

# Reused fakes from the graph/pipeline tests.
from test_agent_graph import StubJudge

SIMPLE_QUESTION = "How do I install FastAPI?"
AGENTIC_QUESTION = (
    "Combine path, query, and body parameters in one endpoint — what are the "
    "validation rules for each kind?"
)
STREAM_TEXT = "Streamed answer [1]"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Fixed result list per query; mirrors test_agent_graph.FakeRetriever."""

    def __init__(self, results_by_query) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, top_k: int = 5, *, language: str | None = None):
        self.calls.append((query, top_k, language))
        return list(self.results_by_query.get(query, []))


class StreamingFakeGenerator:
    """Generator supporting answer streaming (word deltas) + usage records."""

    def __init__(self, text: str = STREAM_TEXT) -> None:
        self.text = text
        self.last_usage: dict | None = None
        self.calls: list[str] = []
        self.last_context = None
        self.last_sources = None
        self.last_question = None

    def _record(self, where: str, context, sources, question) -> None:
        self.calls.append(where)
        self.last_context = context
        self.last_sources = sources
        self.last_question = question

    def generate_answer(self, context, sources, question) -> str:
        self._record("generate_answer", context, sources, question)
        return self.text

    def generate_answer_stream(self, context, sources, question):
        self._record("generate_answer_stream", context, sources, question)
        words = self.text.split(" ")
        for i, word in enumerate(words):
            suffix = " " if i < len(words) - 1 else ""
            yield word + suffix


class PlainFakeGenerator(StreamingFakeGenerator):
    """Phase 4-style generator without streaming support."""

    def generate_answer_stream(self, context, sources, question):  # type: ignore[override]
        raise NotImplementedError("generate_answer_stream not supported")


def _event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _make_result(content: str = "Path parameters are declared as function args.",
                 source_file: str = "en/docs/tutorial/path-params.md",
                 heading: str = "Path Parameters",
                 score: float = 0.81) -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(id="c1", content=content, heading_path=heading, source_file=source_file),
        score=score,
    )


# ---------------------------------------------------------------------------
# ask_events — direct (fast) path
# ---------------------------------------------------------------------------


class TestDirectPathEvents:
    def _run(self, generator, question=SIMPLE_QUESTION):
        events: list[dict] = []
        retriever = FakeRetriever({question: [_make_result()]})
        out = ask_events(
            question,
            retriever=retriever,
            generator=generator,
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        return events, out, retriever

    def test_streams_tokens_then_answer_then_done(self) -> None:
        events, out, _ = self._run(StreamingFakeGenerator())
        types = _event_types(events)
        # gate step → search payload → search step → tokens → answer step →
        # answer → done
        assert types[0] == "step"
        assert events[0]["step"]["step"] == "gate"
        assert "search" in types
        token_idx = types.index("token")
        assert types[token_idx + 1] == "token"  # at least two deltas
        assert types[-2:] == ["answer", "done"]

        # tokens streamed are the generator's text split into words
        streamed = "".join(e["delta"] for e in events if e["type"] == "token")
        assert streamed == STREAM_TEXT

        # final answer is the citation-formatted display
        answer = events[-2]
        assert answer["type"] == "answer"
        assert "Streamed answer" in answer["text"]
        assert answer["direct"] is True
        assert answer["refused"] is False
        assert answer["sources"][0]["kind"] == "tutorial"

        # done carries trace + latency + usage
        done = events[-1]
        assert {t["step"] for t in done["trace"]} == {"gate", "search", "answer"}
        assert isinstance(done["latency_ms"], float)
        assert out["direct"] is True
        assert out["answer"] == answer["text"]

    def test_search_payload_has_chunks_with_scores(self) -> None:
        events, _, _ = self._run(StreamingFakeGenerator())
        search = next(e for e in events if e["type"] == "search")
        assert search["turn"] == 1
        assert search["query"] == SIMPLE_QUESTION
        assert search["results"][0]["file"].endswith("path-params.md")
        assert search["results"][0]["score"] == pytest.approx(0.81)
        assert isinstance(search["results"][0]["excerpt"], str)

    def test_non_streaming_generator_falls_back_to_single_call(self) -> None:
        events, _, _ = self._run(PlainFakeGenerator())
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) == 1
        assert tokens[0]["delta"].strip() == STREAM_TEXT.strip()

    def test_usage_forwarded_to_done(self) -> None:
        gen = StreamingFakeGenerator()
        gen.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        events, out, _ = self._run(gen)
        assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert events[-1]["usage"] == gen.last_usage

    def test_empty_retrieval_uses_no_context_note_and_still_answers(self) -> None:
        events, out, _ = self._run(
            StreamingFakeGenerator("I don't know — the available documentation does not cover this question."),
            question="Install FastAPI on Windows",
        )
        assert out["answer"].startswith("I don't know")
        assert any(e["type"] == "search" for e in events)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            ask_events("hi", strategy="bogus", retriever=FakeRetriever({}), generator=PlainFakeGenerator(), emit=lambda d: None)


class TestDirectPathRouting:
    def test_simple_question_stays_on_direct_path(self) -> None:
        events, out, _ = TestDirectPathEvents()._run(StreamingFakeGenerator(), question=SIMPLE_QUESTION)
        assert out["direct"] is True
        gate = events[0]["step"]
        assert gate["decision"] == "direct"

    def test_forced_direct(self) -> None:
        events = []
        retriever = FakeRetriever({AGENTIC_QUESTION: [_make_result()]})
        out = ask_events(
            AGENTIC_QUESTION,
            strategy="direct",
            retriever=retriever,
            generator=StreamingFakeGenerator(),
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        assert out["direct"] is True
        assert events[0]["step"]["decision"] == "forced-direct"

    def test_gate_decision_detail_includes_signals_when_present(self) -> None:
        events, _, _ = TestDirectPathEvents()._run(
            StreamingFakeGenerator(), question=SIMPLE_QUESTION
        )
        # simple question → no signals → detail is just the reason text
        assert "signals=" not in events[0]["step"].get("detail", "")


# ---------------------------------------------------------------------------
# ask_events — agentic path
# ---------------------------------------------------------------------------


class TestAgenticPathEvents:
    def _run(self, question=AGENTIC_QUESTION, generator=None, strategy="agentic"):
        events: list[dict] = []
        gen = generator or StreamingFakeGenerator(STREAM_TEXT)
        judge = StubJudge([])  # no preset → sufficient on first call
        out = ask_events(
            question,
            strategy=strategy,
            retriever=FakeRetriever({question: [_make_result()]}),
            generator=gen,
            judge=judge,
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        return events, out, gen, judge

    def test_agentic_events_and_delegation(self) -> None:
        events, out, gen, judge = self._run()
        types = _event_types(events)
        assert types[0] == "step" and events[0]["step"]["step"] == "gate"
        assert events[0]["step"]["decision"] == "forced-agentic"
        step_names = [e["step"]["step"] for e in events if e["type"] == "step"]
        assert step_names == ["gate", "search", "judge", "answer"]
        assert "search" in types  # debug payload
        assert "token" in types
        assert types[-2:] == ["answer", "done"]
        assert judge.calls == 1
        answer = events[-2]
        assert answer["direct"] is False
        assert out["trace"][1]["step"] == "search"

    def test_auto_routes_complex_question_to_agentic(self) -> None:
        events, out, _, judge = self._run(strategy="auto")
        gate = events[0]["step"]
        assert gate["decision"] == "agentic"
        assert out["direct"] is False
        assert judge.calls == 1

    def test_refusal_surfaces_verbatim(self) -> None:
        refusal = "I don't know — the available documentation does not cover this question."
        question = "What is the exact XRP exchange rate right now?"
        events: list[dict] = []
        out = ask_events(
            question,
            strategy="agentic",
            retriever=FakeRetriever({question: []}),
            generator=StreamingFakeGenerator(refusal),
            judge=StubJudge([]),  # sufficient on an empty retrieval
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        answer_ev = next(e for e in events if e["type"] == "answer")
        assert answer_ev["text"].startswith("I don't know")
        assert answer_ev["sources"] == []
        assert out["answer"].startswith("I don't know")


# ---------------------------------------------------------------------------
# model knob
# ---------------------------------------------------------------------------


class TestModelKnob:
    def test_model_forwarded_when_building_default_generator(self, monkeypatch) -> None:
        """Generators built lazily (generator=None) receive the model knob."""
        import docpilot.generation.generator as gen_mod

        built: list[str | None] = []

        class RecordingGroqGenerator(StreamingFakeGenerator):
            def __init__(self, model=None) -> None:
                super().__init__(text=STREAM_TEXT)
                built.append(model)

        monkeypatch.setattr(gen_mod, "GroqGenerator", RecordingGroqGenerator)
        events: list[dict] = []
        ask_events(
            SIMPLE_QUESTION,
            model="acme/faster-model",
            retriever=FakeRetriever({SIMPLE_QUESTION: [_make_result()]}),
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        assert built == ["acme/faster-model"]

    def test_injected_generator_wins_over_model(self) -> None:
        gen = StreamingFakeGenerator()
        events: list[dict] = []
        ask_events(
            SIMPLE_QUESTION,
            model="acme/ignored",
            retriever=FakeRetriever({SIMPLE_QUESTION: [_make_result()]}),
            generator=gen,
            citation_engine=StandardCitationEngine(),
            emit=events.append,
        )
        assert gen.calls == ["generate_answer_stream"]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def _scripted_engine(question: str, *, strategy=None, top_k=None, language=None,
                     model=None, emit=None) -> dict:
    """Deterministic fake ask engine for the app-level tests."""
    del model
    sources = [{"ref": 1, "file": "en/docs/tutorial/a.md", "heading": None, "kind": "tutorial"}]
    emit({"type": "step", "step": {"step": "gate", "step_type": "gate", "query": question,
                                   "decision": "direct", "detail": "simple", "latency_ms": 0}})
    emit({"type": "token", "delta": "Hello [1]"})
    emit({"type": "answer", "text": "Hello [1]\n\nSources:\n[1] en/docs/tutorial/a.md",
          "sources": sources, "refused": False, "direct": True})
    emit({"type": "done", "trace": [], "latency_ms": 12.3, "usage": None})
    return {
        "question": question, "answer": "Hello [1]\n\nSources:\n[1] en/docs/tutorial/a.md",
        "sources": sources, "refused": False, "direct": True, "trace": [],
        "latency_ms": 12.3, "usage": None,
    }


def _client(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
    app = create_app(store=store, engine=_scripted_engine)
    return TestClient(app), store


class TestHealth:
    def test_health_ok(self, tmp_path) -> None:
        client, _ = _client(tmp_path)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["store"] is True
        # corpus probe is best-effort: reachable false when PG is unavailable,
        # true when a local PG happens to be up — either is acceptable here.
        assert "corpus" in body


class TestChatSSE:
    def test_streams_events_and_persists(self, tmp_path) -> None:
        client, store = _client(tmp_path)
        session = store.create_session(title="chat test")

        with client.stream("POST", "/api/v1/chat", json={
            "question": "hello", "session_id": session["id"],
        }) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            blocks = [b for b in resp.iter_lines() if b]
            parsed = [parse_sse_block(b)["type"] for b in blocks if parse_sse_block(b)]
            assert parsed == ["step", "token", "answer", "done"]

        # the exchange was persisted (user + assistant rows)
        messages = store.list_messages(session["id"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"].startswith("Hello")
        assert messages[1]["sources"][0]["kind"] == "tutorial"

    def test_without_session_id_no_persist(self, tmp_path) -> None:
        client, store = _client(tmp_path)
        with client.stream("POST", "/api/v1/chat", json={"question": "hello"}) as resp:
            assert resp.status_code == 200
            for _b in resp.iter_lines():
                pass
        assert store.list_sessions() == []

    def test_empty_question_rejected(self, tmp_path) -> None:
        client, _ = _client(tmp_path)
        resp = client.post("/api/v1/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_bad_strategy_rejected(self, tmp_path) -> None:
        client, _ = _client(tmp_path)
        resp = client.post("/api/v1/chat", json={"question": "hi", "strategy": "bogus"})
        assert resp.status_code == 422


class TestSessionsAPI:
    def test_crud_flow(self, tmp_path) -> None:
        client, _ = _client(tmp_path)

        created = client.post("/api/v1/sessions", json={"title": "my chat"})
        assert created.status_code == 200
        sid = created.json()["id"]

        listed = client.get("/api/v1/sessions").json()["sessions"]
        assert len(listed) == 1 and listed[0]["id"] == sid

        got = client.get(f"/api/v1/sessions/{sid}")
        assert got.status_code == 200
        assert got.json()["session"]["title"] == "my chat"
        assert got.json()["messages"] == []

        deleted = client.delete(f"/api/v1/sessions/{sid}")
        assert deleted.status_code == 200

        missing = client.get(f"/api/v1/sessions/{sid}")
        assert missing.status_code == 404
        assert client.delete(f"/api/v1/sessions/{sid}").status_code == 404