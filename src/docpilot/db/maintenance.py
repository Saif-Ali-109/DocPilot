"""Corpus maintenance helpers for the DocPilot PostgreSQL store.

Provides tools to detect and remove duplicate ``chunks`` rows — the
historical symptom of two overlapping/crashed ingest runs inserting the
same logical chunk twice (identical ``source_file`` + ``heading_path`` +
``chunk_index``). The ``chunks_unique_triple`` unique index (see
``db/schema.sql``) prevents new twins; these helpers clean up legacy ones.

Note on transactions: unlike the vector-store methods (which commit per
operation), :func:`dedupe_chunks` does **not** commit — the caller decides.
This keeps the operation safe to run inside a transaction for tests and
allows the CLI to commit once, then apply the schema/unique index.
"""

from __future__ import annotations

import psycopg

# Rows that are logical duplicates: count(*) - count(DISTINCT triple).
# PostgreSQL's DISTINCT treats NULLs as equal, matching the DELETE below
# (which uses IS NOT DISTINCT FROM for a nullable heading_path).
_COUNT_DUPLICATES_SQL = """
    SELECT count(*) - count(DISTINCT (source_file, heading_path, chunk_index))
    FROM chunks
"""

# Keep the earliest row (min id) per (source_file, heading_path, chunk_index)
# and delete the later twin(s). "Earliest" is arbitrary but deterministic —
# copies are byte-identical, so either is valid.
_DELETE_DUPLICATES_SQL = """
    DELETE FROM chunks a
    USING chunks b
    WHERE a.id > b.id
      AND a.source_file = b.source_file
      AND a.heading_path IS NOT DISTINCT FROM b.heading_path
      AND a.chunk_index = b.chunk_index
"""


def count_duplicate_rows(conn: psycopg.Connection) -> int:
    """Return how many rows in *conn* are exact duplicates (twin rows).

    Read-only. A clean corpus returns ``0``.
    """
    with conn.cursor() as cur:
        cur.execute(_COUNT_DUPLICATES_SQL)
        (n,) = cur.fetchone()
    return int(n)


def dedupe_chunks(conn: psycopg.Connection) -> int:
    """Delete duplicate ``chunks`` rows keeping one copy per logical triple.

    Args:
        conn: An open psycopg connection. Must NOT be in autocommit mode if
            the caller wants transaction control; nothing is committed here.

    Returns:
        The number of rows deleted.
    """
    with conn.cursor() as cur:
        cur.execute(_DELETE_DUPLICATES_SQL)
        deleted = cur.rowcount
    return int(deleted)