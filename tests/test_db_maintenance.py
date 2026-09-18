"""Tests for the ``docpilot dedupe`` CLI (dry-run path — read-only).

The dedupe *helpers* themselves (``TestMaintenanceSQL`` /
``TestMaintenanceIntegration``) live in ragkit's test suite next to
``ragkit.db.maintenance``; DocPilot only owns the CLI wiring here.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestDedupeCli:
    """``python -m docpilot dedupe`` wiring (dry-run is read-only)."""

    @pytest.fixture(autouse=True)
    def _connect(self) -> None:
        try:
            from ragkit.db.connection import get_connection

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