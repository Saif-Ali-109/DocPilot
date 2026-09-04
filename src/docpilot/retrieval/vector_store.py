import json
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from docpilot.core.models import Chunk, RetrieverResult


class VectorStore(ABC):
    """Interface for vector similarity search backed by a database."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: Sequence[np.ndarray]) -> None:
        """Store chunks with their pre-computed embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrieverResult]:
        """Return the top_k nearest neighbors by cosine similarity."""
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
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length"
            )
        sql = """
            INSERT INTO chunks (content, heading_path, source_file, chunk_index, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
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
                _json_dumps(meta),
                emb.tolist(),
            ))
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
        self._conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrieverResult]:
        """Return the *top_k* chunks most similar to *query_embedding*.

        Cosine distance is computed via pgvector's ``<=>`` operator.
        Similarity = 1 - distance.  Results are ordered DESC by similarity.
        """
        sql = """
            SELECT content, heading_path, source_file, chunk_index, metadata,
                   1.0 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        q = query_embedding.tolist()
        with self._conn.cursor() as cur:
            cur.execute(sql, (q, q, top_k))
            rows = cur.fetchall()

        results: list[RetrieverResult] = []
        for content, heading_path, source_file, chunk_index, metadata, similarity in rows:
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
