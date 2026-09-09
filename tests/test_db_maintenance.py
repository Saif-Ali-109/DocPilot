"""Tests for docpilot.db.maintenance — corpus dedupe helpers.

The COUNT/DELETE logic is SQL, so the behavioural tests are *integration*
tests against live PostgreSQL, skipped when unreachable (same pattern as
``TestPgVectorStoreIntegration``). They run inside a transaction that is
always ROLLED BACK, so the production corpus and schema are never mutated:

* the unique index is dropped *within the transaction* (DDL is
  transactional in PostgreSQL) so legacy twin rows can be simulated,
* helpers are exercised, then the rollback restores both the index and the
  table state.

Also covers the ``docpilot dedupe`` CLI (dry-run path — read-only).
"""

from __future__ import annotations

import pytest

from docpilot.db import maintenance


# ---------------------------------------------------------------------------
# Static structure (no DB)
# ---------------------------------------------------------------------------


class TestMaintenanceSQL:
    """The SQL constants must target the right table and keep the earliest row."""

    def test_count_targets_chunks(self) -> None:
        assert "FROM chunks" in maintenance._COUNT_DUPLICATES_SQL

    def test_count_uses_distinct_triple(self) -> None:
        assert "count(DISTINCT (source_file, heading_path, chunk_index))" in (
            maintenance._COUNT_DUPLICATES_SQL
        )

    def test_delete_keeps_earliest_row(self) -> None:
        assert "a.id > b.id" in maintenance._DELETE_DUPLICATES_SQL

    def test_delete_joins_on_the_triple(self) -> None:
        lower = maintenance._DELETE_DUPLICATES_SQL.lower()
        assert "source_file = b.source_file" in lower
        assert "chunk_index = b.chunk_index" in lower

    def test_delete_handles_nullable_heading(self) -> None:
        # heading_path is nullable; NULL must match NULL when grouping twins.
        assert "IS NOT DISTINCT FROM" in maintenance._DELETE_DUPLICATES_SQL


# ---------------------------------------------------------------------------
# Integration (live PostgreSQL, transaction + rollback)
# ---------------------------------------------------------------------------

# Embedding column is vector(384); this SQL builds a 384-dim vector.
_INSERT_TWINS_SQL = """
    INSERT INTO chunks (source_file, heading_path, chunk_index, content,
                        language, metadata, embedding)
    SELECT %s, 'Twin', 0, 'duplicate twin payload', 'en', '{}',
           ARRAY(SELECT 0.1 FROM generate_series(1, 384))::vector
    FROM generate_series(1, 2)
"""


@pytest.mark.integration
class TestMaintenanceIntegration:
    """Dedupe helpers against a real PostgreSQL (rolled back afterwards)."""

    @pytest.fixture(autouse=True)
    def _connect(self) -> None:
        try:
            from docpilot.db.connection import get_connection

            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._conn = conn
        except Exception as exc:  # noqa: BLE001 — skip when PG is unreachable
            pytest.skip(f"PostgreSQL unavailable: {exc}")

    @pytest.fixture(autouse=True)
    def _rollback(self) -> None:
        yield
        try:
            self._conn.rollback()
            self._conn.close()
        except Exception:
            pass

    def _wipe_marker(self, source_file: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_file = %s", (source_file,))

    def test_count_duplicate_rows_reports_twins(self) -> None:
        conn = self._conn
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_unique_triple")
        self._wipe_marker("_integration_dup.md")
        with conn.cursor() as cur:
            cur.execute(_INSERT_TWINS_SQL, ("_integration_dup.md",))
        assert maintenance.count_duplicate_rows(conn) >= 1

    def test_dedupe_chunks_removes_twin_rows(self) -> None:
        conn = self._conn
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_unique_triple")
        self._wipe_marker("_integration_dup.md")
        with conn.cursor() as cur:
            cur.execute(_INSERT_TWINS_SQL, ("_integration_dup.md",))
        deleted = maintenance.dedupe_chunks(conn)
        assert deleted >= 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM chunks WHERE source_file = %s",
                ("_integration_dup.md",),
            )
            (n,) = cur.fetchone()
        assert n == 1, "twin rows must collapse to a single kept row"
        # After dedupe within this transaction no twins remain.
        assert maintenance.count_duplicate_rows(conn) == 0

    def test_dedupe_noop_on_clean_table(self) -> None:
        conn = self._conn
        self._wipe_marker("_integration_dup.md")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            (baseline,) = cur.fetchone()
        deleted = maintenance.dedupe_chunks(conn)
        assert deleted == 0
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            (after,) = cur.fetchone()
        assert after == baseline


@pytest.mark.integration
class TestDedupeCli:
    """``python -m docpilot dedupe`` wiring (dry-run is read-only)."""

    @pytest.fixture(autouse=True)
    def _connect(self) -> None:
        try:
            from docpilot.db.connection import get_connection

            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._conn = conn
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"PostgreSQL unavailable: {exc}")
        yield
        try:
            self._conn.close()
        except Exception:
            pass

    def test_dry_run_reports_count_without_committing(self, capsys) -> None:
        from docpilot import cli

        code = cli.main(["dedupe", "--dry-run"], conn=self._conn)
        captured = capsys.readouterr()
        assert code == 0
        assert "Would delete" in captured.out
        assert captured.out.strip().endswith("duplicate chunk row(s)")

    def test_dry_run_is_exit_code_zero(self, capsys) -> None:
        from docpilot import cli

        assert cli.main(["dedupe", "--dry-run"], conn=self._conn) == 0