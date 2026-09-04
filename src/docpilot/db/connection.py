"""PostgreSQL connection management for DocPilot.

Provides ``get_connection()`` (a psycopg 3 Connection built from the
POSTGRES_* config keys) and ``ensure_schema()`` which executes
``db/schema.sql`` idempotently.

No pgvector-specific logic lives here — that belongs in ``retrieval.vector_store``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg

from docpilot import config

logger = logging.getLogger(__name__)

_SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> psycopg.Connection:
    """Return a new psycopg 3 Connection to the DocPilot PostgreSQL database.

    The connection uses the credentials from ``docpilot.config`` (POSTGRES_* keys).
    The caller is responsible for closing the connection.
    """
    conn = psycopg.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )
    logger.debug(
        "Connected to PostgreSQL %s:%s/%s",
        config.POSTGRES_HOST,
        config.POSTGRES_PORT,
        config.POSTGRES_DB,
    )
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Execute ``db/schema.sql`` idempotently on *conn*.

    The schema uses ``CREATE EXTENSION IF NOT EXISTS`` and
    ``CREATE TABLE IF NOT EXISTS`` so it is safe to call repeatedly.

    Args:
        conn: An open psycopg 3 connection.
    """
    sql = _SCHEMA_SQL_PATH.read_text()
    conn.execute(sql)
    conn.commit()
    logger.debug("Schema ensured (extensions + chunks table).")


@contextmanager
def connection_context() -> Generator[psycopg.Connection, None, None]:
    """Context-manager helper that yields a connection and ensures schema.

    Usage::

        with connection_context() as conn:
            # schema is ready, conn is open
            ...
        # conn is closed automatically

    The schema is ensured on every entry; because the DDL is idempotent
    (``IF NOT EXISTS``), repeated calls are harmless.
    """
    conn = get_connection()
    try:
        ensure_schema(conn)
        yield conn
    finally:
        conn.close()
