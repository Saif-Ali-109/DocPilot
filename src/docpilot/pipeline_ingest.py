"""End-to-end ingestion pipeline: loader → parser → chunker → embed → store.

Orchestrates the Phase 1 interface implementations (SPEC.md §3.1, §3.4) into a
single :func:`ingest_corpus` callable. Every component is injectable so the
pipeline can be tested hermetically without a live PostgreSQL instance or the
BGE embedding model; production defaults are used when nothing is injected.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from docpilot.core.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


@dataclass
class IngestStats:
    """Aggregate statistics for a single ingestion run.

    Attributes:
        files_processed: Number of documents loaded from the corpus.
        total_chunks: Number of chunks stored across all documents.
        chunks_per_source: Mapping of ``source_file`` → chunk count.
        stage_timings: Seconds spent per stage
            (``load``, ``chunk``, ``delete``, ``embed``, ``store``).
        avg_chunk_size: Average chunk size (words, an approximate token size).
        min_chunk_size: Smallest chunk size (words).
        max_chunk_size: Largest chunk size (words).
    """

    files_processed: int = 0
    total_chunks: int = 0
    chunks_per_source: dict[str, int] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0


def _word_count(text: str) -> int:
    """Approximate token count via whitespace word splitting."""
    return len(text.split())


def _build_default_vector_store():
    """Open a DB connection, ensure the schema, wrap it in a PgVectorStore.

    Returns:
        A ``(store, conn)`` tuple. The caller owns *conn* and must close it.

    Raises:
        RuntimeError: If the database is unreachable or the pgvector schema
            cannot be applied. The psycopg/native error is chained as the
            cause so the message stays human-friendly.
    """
    from docpilot.db.connection import ensure_schema, get_connection
    from docpilot.retrieval.vector_store import PgVectorStore

    conn = None
    try:
        conn = get_connection()
        ensure_schema(conn)
        store = PgVectorStore(conn)
    except Exception as exc:
        if conn is not None:
            conn.close()
        raise RuntimeError(
            "Could not connect to the DocPilot PostgreSQL database / apply the "
            "pgvector schema — is the server running and is the 'vector' "
            f"extension available? ({exc})"
        ) from exc
    return store, conn


def ingest_corpus(
    corpus_dir: str | Path = "docs",
    *,
    loader=None,
    parser=None,
    chunker=None,
    embedding_provider=None,
    vector_store=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestStats:
    """Run the full ingestion pipeline and return per-run statistics.

    Args:
        corpus_dir: Directory containing the Markdown/MDX corpus. Only used
            when *loader* is not injected (default ``docs/``).
        loader: An optional ``DocumentLoader``
            (default: ``FastAPIDocumentLoader(corpus_dir)``).
        parser: An optional ``Parser`` (default: ``MarkdownParser``). Note the
            current ``MarkdownChunker`` performs its own internal parsing, so
            this parameter is part of the interface seam (for future chunker
            implementations) but is not consumed by the Phase 1 chunker.
        chunker: An optional ``Chunker`` (default: ``MarkdownChunker``).
        embedding_provider: An optional ``EmbeddingProvider``
            (default: ``BGEEmbeddingProvider``).
        vector_store: An optional ``VectorStore``
            (default: ``PgVectorStore`` over a fresh connection with the
            schema ensured and closed afterwards).
        batch_size: Number of chunks embedded and stored per batch.

    Returns:
        :class:`IngestStats` describing this run.

    The pipeline is idempotent (SPEC.md §3.4): before inserting any chunks, it
    deletes every existing row whose ``source_file`` is about to be
    (re)inserted — a re-run never duplicates chunks.
    """
    from docpilot.embeddings.provider import BGEEmbeddingProvider
    from docpilot.ingestion.chunker import MarkdownChunker
    from docpilot.ingestion.fastapi_loader import FastAPIDocumentLoader
    from docpilot.ingestion.parser import MarkdownParser

    store = vector_store
    conn = None
    if store is None:
        store, conn = _build_default_vector_store()

    try:
        loader = loader or FastAPIDocumentLoader(docs_dir=Path(corpus_dir))
        parser = parser or MarkdownParser()
        chunker = chunker or MarkdownChunker()
        embedding_provider = embedding_provider or BGEEmbeddingProvider()

        timings: dict[str, float] = {}

        # ── (1) load docs ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        documents = loader.load()
        timings["load"] = time.perf_counter() - t0
        logger.info("Loaded %d document(s)", len(documents))

        # ── (2) chunk every document ───────────────────────────────────────
        t0 = time.perf_counter()
        all_chunks: list[Chunk] = []
        chunks_per_source: dict[str, int] = {}
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
            chunks_per_source[doc.file_path] = len(chunks)
        timings["chunk"] = time.perf_counter() - t0

        # Approximate token size = word count (SPEC.md §3.13).
        sizes = [_word_count(c.content) for c in all_chunks]

        # ── (3) idempotency: delete rows for the sources about to be
        #        (re)inserted ───────────────────────────────────────────────
        source_files = [doc.file_path for doc in documents]
        t0 = time.perf_counter()
        deleted = store.delete_by_source(source_files) if source_files else 0
        timings["delete"] = time.perf_counter() - t0
        if source_files:
            logger.info(
                "Delete-before-insert: removed %d existing chunk(s) for %d source file(s)",
                deleted,
                len(source_files),
            )

        # ── (4) embed + store in batches ───────────────────────────────────
        embed_time = 0.0
        store_time = 0.0
        batch_count = 0
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start : start + batch_size]
            t = time.perf_counter()
            embeddings = embedding_provider.embed([c.content for c in batch])
            embed_time += time.perf_counter() - t
            t = time.perf_counter()
            store.add(batch, embeddings)
            store_time += time.perf_counter() - t
            batch_count += 1
        timings["embed"] = embed_time
        timings["store"] = store_time
        logger.info(
            "Embedded and stored %d chunk(s) in %d batch(es) of size ≤ %d",
            len(all_chunks),
            batch_count,
            batch_size,
        )

        return IngestStats(
            files_processed=len(documents),
            total_chunks=len(all_chunks),
            chunks_per_source=chunks_per_source,
            stage_timings=timings,
            avg_chunk_size=(sum(sizes) / len(sizes)) if sizes else 0.0,
            min_chunk_size=min(sizes) if sizes else 0,
            max_chunk_size=max(sizes) if sizes else 0,
        )
    finally:
        if conn is not None:
            conn.close()