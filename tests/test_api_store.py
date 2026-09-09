"""Hermetic tests for the Phase 5 SQLite session/chat store (SPEC §7).

Uses a tmp-path SQLite file — no network, no PostgreSQL, no LLM.
"""

from __future__ import annotations

from docpilot.api.store import SessionStore


def _fresh_store(tmp_path) -> SessionStore:
    return SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))


class TestSessions:
    def test_create_and_get(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session(title="My chat")
        assert s["title"] == "My chat"
        assert store.get_session(s["id"]) == s

    def test_default_title(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        assert s["title"] == "New chat"

    def test_list_newest_first(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        a = store.create_session(title="a")
        b = store.create_session(title="b")
        ids = [s["id"] for s in store.list_sessions()]
        assert ids == [b["id"], a["id"]]

    def test_get_missing_returns_none(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        assert store.get_session("nope") is None

    def test_delete_cascades_to_messages(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        store.add_message(s["id"], "user", "hi")
        store.add_message(s["id"], "assistant", "hello [1]")
        assert len(store.list_messages(s["id"])) == 2
        assert store.delete_session(s["id"]) is True
        assert store.get_session(s["id"]) is None
        assert store.list_messages(s["id"]) == []

    def test_delete_missing_returns_false(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        assert store.delete_session("nope") is False


class TestMessages:
    def test_add_and_list_round_trip(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        sources = [{"ref": 1, "file": "a.md", "heading": None, "kind": "tutorial"}]
        trace = [{"step": "gate", "decision": "direct"}]
        m = store.add_message(
            s["id"],
            "assistant",
            "Answer [1]\n\nSources:\n[1] a.md",
            sources=sources,
            trace=trace,
            latency_ms=1234.5,
            refused=False,
        )
        assert m is not None
        assert m["role"] == "assistant"
        assert m["sources"] == sources
        assert m["trace"] == trace
        assert m["latency_ms"] == 1234.5
        assert m["refused"] is False

        listed = store.list_messages(s["id"])
        assert len(listed) == 1
        assert listed[0]["content"].startswith("Answer")
        assert listed[0]["sources"] == sources
        assert listed[0]["trace"] == trace
        assert listed[0]["refused"] is False

    def test_refused_flag_round_trips(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        store.add_message(s["id"], "assistant", "I don't know — the available documentation does not cover this question.", refused=True)
        assert store.list_messages(s["id"])[0]["refused"] is True

    def test_add_to_missing_session_returns_none(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        assert store.add_message("nope", "user", "hi") is None

    def test_adding_a_message_touches_session_updated_at(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        before = store.get_session(s["id"])["updated_at"]
        store.add_message(s["id"], "user", "hi")
        after = store.get_session(s["id"])["updated_at"]
        assert after >= before

    def test_messages_are_chronological(self, tmp_path) -> None:
        store = _fresh_store(tmp_path)
        s = store.create_session()
        for i in range(3):
            store.add_message(s["id"], "user", f"q{i}")
        contents = [m["content"] for m in store.list_messages(s["id"])]
        assert contents == ["q0", "q1", "q2"]