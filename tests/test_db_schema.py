"""Static tests for db/schema.sql — no live database required.

These tests read the schema.sql file and assert structural requirements
from SPEC.md §3.4 without ever connecting to PostgreSQL.
"""

from pathlib import Path

import pytest

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "src" / "docpilot" / "db" / "schema.sql"


@pytest.fixture(scope="module")
def schema_sql() -> str:
    """Read schema.sql once per test module."""
    text = _SCHEMA_PATH.read_text()
    assert len(text) > 0, "schema.sql is empty"
    return text


# ---- Required elements ----

class TestSchemaRequiredElements:
    """Schema must contain these DDL statements (SPEC §3.4)."""

    def test_create_vector_extension(self, schema_sql: str) -> None:
        assert "CREATE EXTENSION IF NOT EXISTS vector" in schema_sql

    def test_create_pgcrypto_extension(self, schema_sql: str) -> None:
        assert "CREATE EXTENSION IF NOT EXISTS pgcrypto" in schema_sql

    def test_chunks_table(self, schema_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS chunks" in schema_sql

    def test_embedding_column(self, schema_sql: str) -> None:
        assert "embedding" in schema_sql

    def test_vector_384(self, schema_sql: str) -> None:
        assert "vector(384)" in schema_sql


# ---- Forbidden elements: NO ANN index in Phase 1 ----

class TestSchemaNoAnnIndex:
    """Phase 1 schema must NOT contain any ANN index definitions.

    The schema's documentation comment mentions "HNSW"/"IVFFlat" as future
    work — that is fine. We strip ``--`` comment lines and assert that no
    actual ANN index DDL (IVFFlat/HNSW) exists in the executable SQL.
    """

    @pytest.fixture()
    def executable_sql(self, schema_sql: str) -> str:
        """schema.sql with comment lines stripped (only executable SQL)."""
        lines = []
        for line in schema_sql.splitlines():
            # A line whose first non-whitespace characters are `--` is a comment.
            if line.lstrip().startswith("--"):
                continue
            lines.append(line)
        return "\n".join(lines)

    def test_no_ivfflat(self, executable_sql: str) -> None:
        lower = executable_sql.lower()
        assert "ivfflat" not in lower, "IVFFlat index found — not allowed in Phase 1"

    def test_no_hnsw(self, executable_sql: str) -> None:
        lower = executable_sql.lower()
        assert "hnsw" not in lower, "HNSW index found — not allowed in Phase 1"

    def test_no_using_ivf(self, executable_sql: str) -> None:
        lower = executable_sql.lower()
        assert "using ivf" not in lower, "USING ivf clause found — not allowed in Phase 1"

    def test_no_using_hnsw(self, executable_sql: str) -> None:
        lower = executable_sql.lower()
        assert "using hnsw" not in lower, "USING hnsw clause found — not allowed in Phase 1"
