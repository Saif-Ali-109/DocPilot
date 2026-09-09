"""SQLite session/chat-history store for the Phase 5 API (SPEC §7).

Tables:

    sessions(id TEXT PK, title, created_at, updated_at)
    messages(id INTEGER PK AUTOINCREMENT, session_id FK ON DELETE CASCADE,
             role, content, sources JSON, trace JSON, latency_ms, refused,
             created_at)

Design notes:

    * stdlib ``sqlite3`` only — no ORM (keep deps light; the corpus store is
      PostgreSQL backend-side, this is purely the session/chat record).
    * WAL mode + busy_timeout; every operation opens its own short-lived
      connection so concurrent async requests (each running in its own worker
      thread) never share a sqlite connection object.
    * ``sources`` / ``trace`` columns are JSON text (``None`` when absent);
      helpers accept/return Python lists.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from docpilot import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    sources    TEXT,
    trace      TEXT,
    latency_ms REAL,
    refused    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now_iso() -> str:
    """ISO-8601 local timestamp (second precision — fine for chat records)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class SessionStore:
    """SQLite-backed sessions + chat messages (thread-safe)."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or config.DOCPILOT_DB_PATH
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # connections
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            # WAL is file-based; :memory: databases ignore it safely.
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def create_session(self, title: str | None = None) -> dict:
        """Create a session row and return it as a dict."""
        session_id = uuid4().hex
        now = _now_iso()
        title = (title or "New chat")[:200]
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return {"id": session_id, "title": title, "created_at": now, "updated_at": now}

    def list_sessions(self) -> list[dict]:
        """Return all sessions, most recently updated first.

        Ties on the second-precision ``updated_at`` are broken by insertion
        order (``rowid``) so the ordering is deterministic.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions "
                "ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> dict | None:
        """Return one session (without messages), or ``None``."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages (CASCADE). Returns ``True`` if a
        row was deleted."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def _touch_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now_iso(), session_id),
        )

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        sources: list[dict] | None = None,
        trace: list[dict] | None = None,
        latency_ms: float | None = None,
        refused: bool = False,
    ) -> dict | None:
        """Append a message to *session_id*; touches the session's
        ``updated_at``. Returns the stored message dict, or ``None`` when the
        session does not exist (FK violation is caught)."""
        now = _now_iso()
        conn = self._connect()
        try:
            try:
                cur = conn.execute(
                    "INSERT INTO messages (session_id, role, content, sources, trace, "
                    "latency_ms, refused, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        role,
                        content,
                        json.dumps(sources) if sources is not None else None,
                        json.dumps(trace) if trace is not None else None,
                        latency_ms,
                        1 if refused else 0,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                # Session does not exist — FK violated.
                return None
            self._touch_session(conn, session_id)
            conn.commit()
            message_id = cur.lastrowid
        finally:
            conn.close()
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources,
            "trace": trace,
            "latency_ms": latency_ms,
            "refused": refused,
            "created_at": now,
        }

    def list_messages(self, session_id: str) -> list[dict]:
        """Return a session's messages in chronological order (empty when the
        session does not exist)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, session_id, role, content, sources, trace, latency_ms, "
                "refused, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                item = dict(r)
                item["sources"] = _loads(item.get("sources"))
                item["trace"] = _loads(item.get("trace"))
                item["refused"] = bool(item["refused"])
                out.append(item)
            return out
        finally:
            conn.close()


def _loads(value: Any) -> Any:
    """Parse a JSON column back to a Python object (``None`` passthrough)."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None