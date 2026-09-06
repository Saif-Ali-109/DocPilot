"""Tests for SimpleRetriever and VectorStore interfaces.

Uses lightweight in-memory fakes — no live database or network required.
Also includes an optional integration test that exercises PgVectorStore
against a real PostgreSQL (skipped when unavailable).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from docpilot.core.models import Chunk, RetrieverResult
from docpilot.retrieval.vector_store import VectorStore
from docpilot.retrieval.retriever import SimpleRetriever, Retriever


# ======================================================================
# Fakes / in-memory implementations
# ======================================================================

class FakeEmbeddingProvider:
    """Deterministic embedding provider for testing.

    Returns a one-hot-style vector of dimension 384 where the active
    index is ``hash(text) % 384``.  This is deterministic and fast.
    """

    DIMENSION = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.DIMENSION), dtype=np.float32)
        for i, text in enumerate(texts):
            idx = hash(text) % self.DIMENSION
            result[i, idx] = 1.0
        return result

    @property
    def dimension(self) -> int:
        return self.DIMENSION


class InMemoryVectorStore(VectorStore):
    """In-memory VectorStore using numpy cosine similarity.

    For testing only — mirrors the PgVectorStore contract without any DB.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, chunks: list[Chunk], embeddings: Sequence[np.ndarray]) -> None:
        assert len(chunks) == len(embeddings)
        self._chunks.extend(chunks)
        self._embeddings.extend([e.astype(np.float32) for e in embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int, *, language: str | None = None) -> list[RetrieverResult]:
        if not self._chunks:
            return []
        # Compute cosine similarity (embeddings are assumed unit-normed)
        matrix = np.stack(self._embeddings)  # (n, 384)
        scores = matrix @ query_embedding.astype(np.float32)  # (n,)
        # Top-k indices (descending)
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            results.append(
                RetrieverResult(chunk=self._chunks[int(idx)], score=float(scores[int(idx)]))
            )
        return results

    def delete_by_source(self, source_files: list[str]) -> int:
        source_set = set(source_files)
        before = len(self._chunks)
        pairs = list(zip(self._chunks, self._embeddings))
        kept = [(c, e) for c, e in pairs if c.source_file not in source_set]
        self._chunks = [c for c, _ in kept]
        self._embeddings = [e for _, e in kept]
        return before - len(self._chunks)

    def count(self) -> int:
        return len(self._chunks)


# ======================================================================
# SimpleRetriever tests (using fakes)
# ======================================================================

class TestSimpleRetriever:
    """Test SimpleRetriever with in-memory fakes."""

    def _make_chunks(self, n: int = 5, source: str = "test.md") -> list[Chunk]:
        """Create n test chunks."""
        return [
            Chunk(
                id=f"chunk-{i}",
                content=f"Content of chunk {i}",
                heading_path=f"/Section {i}/",
                source_file=source,
                chunk_index=i,
                metadata={},
            )
            for i in range(n)
        ]

    def test_returns_top_k(self) -> None:
        """retrieve() returns exactly top_k results."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = self._make_chunks(10)
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)

        retriever = SimpleRetriever(provider, store)
        results = retriever.retrieve("Content of chunk 2", top_k=3)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_ordered_by_score_desc(self) -> None:
        """Results are ordered by descending similarity score."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = self._make_chunks(10)
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)

        retriever = SimpleRetriever(provider, store)
        results = retriever.retrieve("Content of chunk 0", top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not in descending score order"

    def test_exact_match_is_top(self) -> None:
        """A chunk whose content matches the query should score highest."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = self._make_chunks(5)
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)

        retriever = SimpleRetriever(provider, store)
        results = retriever.retrieve("Content of chunk 3", top_k=5)

        # The top result should be chunk-3 (same content → same hash → cosine=1.0)
        assert results[0].chunk.id == "chunk-3"
        assert results[0].score == pytest.approx(1.0)

    def test_returns_all_when_top_k_exceeds_count(self) -> None:
        """If top_k > available chunks, returns all available."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = self._make_chunks(3)
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)

        retriever = SimpleRetriever(provider, store)
        results = retriever.retrieve("anything", top_k=10)

        assert len(results) == 3

    def test_retriever_result_types(self) -> None:
        """Each result is a RetrieverResult with a Chunk and float score."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = self._make_chunks(3)
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)

        retriever = SimpleRetriever(provider, store)
        results = retriever.retrieve("test", top_k=3)

        for r in results:
            assert isinstance(r, RetrieverResult)
            assert isinstance(r.chunk, Chunk)
            assert isinstance(r.score, float)


# ======================================================================
# InMemoryVectorStore interface contract tests
# ======================================================================

class TestInMemoryVectorStore:
    """Direct interface-contract checks against InMemoryVectorStore."""

    def test_count_empty(self) -> None:
        store = InMemoryVectorStore()
        assert store.count() == 0

    def test_count_after_add(self) -> None:
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = [Chunk(id=f"c{i}", content=f"text {i}", source_file="a.md", chunk_index=i) for i in range(4)]
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)
        assert store.count() == 4

    def test_delete_by_source(self) -> None:
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        c1 = Chunk(id="c1", content="hello", source_file="file1.md", chunk_index=0)
        c2 = Chunk(id="c2", content="world", source_file="file2.md", chunk_index=0)
        store.add([c1, c2], provider.embed(["hello", "world"]))
        deleted = store.delete_by_source(["file1.md"])
        assert deleted == 1
        assert store.count() == 1

    def test_delete_by_source_returns_zero_when_no_match(self) -> None:
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        c1 = Chunk(id="c1", content="hello", source_file="file1.md", chunk_index=0)
        store.add([c1], provider.embed(["hello"]))
        deleted = store.delete_by_source(["nonexistent.md"])
        assert deleted == 0
        assert store.count() == 1

    def test_search_limit(self) -> None:
        """search(top_k) never returns more than top_k results."""
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        chunks = [Chunk(id=f"c{i}", content=f"text {i}", source_file="a.md", chunk_index=i) for i in range(20)]
        embeddings = provider.embed([c.content for c in chunks])
        store.add(chunks, embeddings)
        results = store.search(provider.embed(["text 5"])[0], top_k=5)
        assert len(results) == 5

    def test_search_returns_empty_on_empty_store(self) -> None:
        store = InMemoryVectorStore()
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        results = store.search(query, top_k=5)
        assert results == []


# ======================================================================
# Retriever interface check
# ======================================================================

class TestRetrieverInterface:
    """SimpleRetriever is-a Retriever."""

    def test_is_retriever(self) -> None:
        provider = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        retriever = SimpleRetriever(provider, store)
        assert isinstance(retriever, Retriever)


# ======================================================================
# Optional: PgVectorStore integration test (skipped without PostgreSQL)
# ======================================================================

@pytest.mark.integration
class TestPgVectorStoreIntegration:
    """Integration tests for PgVectorStore against a real PostgreSQL.

    These tests skip gracefully when PostgreSQL is unreachable.
    """

    @pytest.fixture(autouse=True)
    def _connect(self, request):
        """Try to connect; skip if PostgreSQL is unavailable."""
        try:
            from docpilot.db.connection import get_connection, ensure_schema
            from docpilot.retrieval.vector_store import PgVectorStore

            conn = get_connection()
            # Quick connectivity check
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            ensure_schema(conn)
            self._conn = conn
            self._store = PgVectorStore(conn)
        except Exception as exc:
            pytest.skip(f"PostgreSQL unavailable: {exc}")

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove test fixture rows from the shared DB after each test.

        These integration tests write marker source_files
        (``_integration_*.md``) straight into the real ``chunks`` table;
        without teardown they pollute production retrieval.
        """
        yield
        if getattr(self, "_store", None) is not None:
            try:
                self._store.delete_by_source(
                    ["_integration_test.md", "_integration_del.md", "_integration_search.md"]
                )
            except Exception:
                pass

    def test_add_and_count(self) -> None:
        # Clean slate
        self._store.delete_by_source(["_integration_test.md"])

        provider = FakeEmbeddingProvider()
        chunks = [Chunk(id="itest-0", content="integration test", source_file="_integration_test.md", chunk_index=0)]
        embeddings = provider.embed(["integration test"])
        self._store.add(chunks, embeddings)
        assert self._store.count() >= 1

    def test_delete_by_source(self) -> None:
        provider = FakeEmbeddingProvider()
        chunks = [Chunk(id="del-0", content="to delete", source_file="_integration_del.md", chunk_index=0)]
        embeddings = provider.embed(["to delete"])
        self._store.add(chunks, embeddings)
        deleted = self._store.delete_by_source(["_integration_del.md"])
        assert deleted >= 1

    def test_search(self) -> None:
        self._store.delete_by_source(["_integration_search.md"])
        provider = FakeEmbeddingProvider()
        chunks = [
            Chunk(id="s-0", content="hello world", source_file="_integration_search.md", chunk_index=0),
            Chunk(id="s-1", content="goodbye world", source_file="_integration_search.md", chunk_index=1),
        ]
        embeddings = provider.embed([c.content for c in chunks])
        self._store.add(chunks, embeddings)
        q = provider.embed(["hello world"])[0]
        results = self._store.search(q, top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.content == "hello world"
