import json
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from docpilot.core.models import Chunk, RetrieverResult, source_language


class VectorStore(ABC):
    """Interface for vector similarity search backed by a database."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: Sequence[np.ndarray]) -> None:
        """Store chunks with their pre-computed embeddings."""
        ...

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Return the top_k nearest neighbors by cosine similarity.

        Args:
            query_embedding: The query vector.
            top_k: Maximum results to return.
            language: If set, restrict results to chunks whose source file
                starts with this language tag (e.g. ``"en"``). ``None``
                disables filtering.
        """
        ...

    @abstractmethod
    def delete_by_source(self, source_files: list[str]) -> int:
        """Delete all chunks whose source_file is in the given list. Return count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the store."""
        ...


class PgVectorStore(VectorStore):
    """VectorStore implementation backed by pgvector (PostgreSQL).

    **Design notes on psycopg 3 + pgvector:**

    We use ``pgvector.psycopg.register_vector(conn)`` (available in
    pgvector ≥ 0.3) to register the vector adapter/injector so that
    ``numpy.ndarray`` objects can be passed directly as query parameters
    for the ``embedding`` column.  For inserts, pgvector's register_vector
    also accepts ``list[float]`` or ``numpy.ndarray`` — so we pass the
    raw ``np.ndarray`` objects directly rather than serializing them to
    text strings.

    **Chunk id vs. schema id:**

    The ``Chunk`` dataclass has a string ``id`` attribute used as a
    deterministic identifier across ingestion runs.  The ``chunks`` table
    uses ``UUID PRIMARY KEY DEFAULT gen_random_uuid()`` for its ``id``
    column — the two are *not* the same.  The deterministic chunk id is
    stored in ``metadata`` JSONB under key ``"chunk_id"`` so that
    AGENT E's debug output can reference it.  (See class docstring.)
    """

    def __init__(self, conn) -> None:
        """Create a PgVectorStore over an open psycopg 3 connection.

        Args:
            conn: A psycopg 3 connection.  The caller must ensure the
                connection is open and the schema has been applied
                (``db.connection.ensure_schema``).
        """
        import pgvector.psycopg as pgvector_psycopg

        self._conn = conn
        # Register the pgvector type adapter on this connection so that
        # numpy arrays / list[float] can round-trip through the driver.
        pgvector_psycopg.register_vector(conn)

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: Sequence[np.ndarray]) -> None:
        """Insert *chunks* together with their pre-computed *embeddings*.

        Each chunk's deterministic ``chunk.id`` string is stored in the
        ``metadata`` JSONB column under key ``"chunk_id"``.

        Inserts are conflict-tolerated: a row whose ``(source_file,
        heading_path, chunk_index)`` triple already exists is skipped
        (``ON CONFLICT DO NOTHING``) rather than duplicating the chunk.
        Combined with the ``chunks_unique_triple`` unique index this makes
        ingest idempotent even under a delete/insert race (PLAN.md §6).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length"
            )
        sql = """
            INSERT INTO chunks (content, heading_path, source_file, chunk_index,
                                language, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT DO NOTHING
        """
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            # Merge chunk_id into the existing metadata dict
            meta = dict(chunk.metadata)
            meta["chunk_id"] = chunk.id
            rows.append((
                chunk.content,
                chunk.heading_path,
                chunk.source_file,
                chunk.chunk_index,
                source_language(chunk.source_file),
                _json_dumps(meta),
                emb.tolist(),
            ))
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
        self._conn.commit()

    def search(
        self, query_embedding: np.ndarray, top_k: int, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Return the *top_k* chunks most similar to *query_embedding*.

        Cosine distance is computed via pgvector's ``<=>`` operator.
        Similarity = 1 - distance.  Results are ordered DESC by similarity.

        A slightly wider window (``top_k * 2``) is fetched and rows whose
        ``(source_file, heading_path, chunk_index)`` triple was already seen
        are dropped, keeping the highest-scoring copy. This guarantees a
        duplicate chunk can never consume two top-k slots even if the table
        ever holds twin rows again (they cannot — see ``chunks_unique_triple``;
        this is a defensive safety net). With no duplicates the result set is
        identical to a plain ``LIMIT top_k`` query.

        Args:
            query_embedding: The query vector.
            top_k: Maximum results to return.
            language: If set, restrict results to chunks whose ``language``
                column matches this value. ``None`` disables filtering.
        """
        q = query_embedding.tolist()
        window = top_k * 2
        if language is not None:
            sql = """
                SELECT content, heading_path, source_file, chunk_index, metadata,
                       1.0 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE language = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (q, language, q, window)
        else:
            sql = """
                SELECT content, heading_path, source_file, chunk_index, metadata,
                       1.0 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (q, q, window)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[RetrieverResult] = []
        for content, heading_path, source_file, chunk_index, metadata, similarity in _first_distinct_rows(
            rows, top_k
        ):
            # Recover the deterministic chunk_id from metadata if present
            meta = _parse_metadata(metadata)
            chunk_id = meta.pop("chunk_id", "")
            results.append(
                RetrieverResult(
                    chunk=Chunk(
                        id=chunk_id,
                        content=content,
                        heading_path=heading_path,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata=meta,
                    ),
                    score=float(similarity),
                )
            )
        return results

    def delete_by_source(self, source_files: list[str]) -> int:
        """Delete all chunks whose ``source_file`` is in *source_files*.

        Returns the number of rows deleted.  This is the idempotent
        re-ingestion path (delete-then-insert).
        """
        sql = "DELETE FROM chunks WHERE source_file = ANY(%s)"
        with self._conn.cursor() as cur:
            cur.execute(sql, (source_files,))
            deleted = cur.rowcount
        self._conn.commit()
        return deleted

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        sql = "SELECT count(*) FROM chunks"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            (n,) = cur.fetchone()
        return int(n)


# ------------------------------------------------------------------
# Helpers (module-private)
# ------------------------------------------------------------------

def _json_dumps(obj) -> str:
    """Serialize *obj* to a JSON string for psycopg JSONB parameters."""
    return json.dumps(obj)


def _parse_metadata(raw) -> dict:
    """Parse a metadata value that may be a dict (from psycopg) or a JSON string."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    return {}


def _first_distinct_rows(rows: list[tuple], top_k: int) -> list[tuple]:
    """Keep the first occurrence of each ``(source_file, heading_path, chunk_index)``.

    ``rows`` is expected ordered by descending similarity, so the first
    occurrence of a triple carries its highest score. Used by
    :meth:`PgVectorStore.search` as a defensive dedupe net; with no duplicate
    triples it returns ``rows[:top_k]`` unchanged.
    """
    seen: set[tuple] = set()
    out: list[tuple] = []
    for row in rows:
        key = (row[2], row[1], row[3])  # source_file, heading_path, chunk_index
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= top_k:
            break
    return out
