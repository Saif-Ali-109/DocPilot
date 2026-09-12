"""Tests for hybrid retrieval — RRF fusion + Postgres FTS searcher.

Hermetic by default (fake retriever/searcher/connection); an integration
test against real PostgreSQL skips when the DB is unavailable.
"""

from __future__ import annotations

import pytest

from docpilot.core.models import Chunk, RetrieverResult
from docpilot.retrieval.hybrid import HybridRetriever, _rrf_fuse
from docpilot.retrieval.lexical import LexicalSearcher, PostgresFTSSearcher


def _result(
    file: str, heading: str, index: int, content: str = "content", score: float = 0.5
) -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(
            id=f"{file}-{heading}-{index}",
            content=content,
            heading_path=heading,
            source_file=file,
            chunk_index=index,
        ),
        score=score,
    )


class FakeVectorRetriever:
    """Canned Retriever-interface double that records calls."""

    def __init__(self, results: list[RetrieverResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query, top_k=5, *, language=None):
        self.calls.append((query, top_k, language))
        return self.results[:top_k]


class FakeLexicalSearcher(LexicalSearcher):
    """Canned LexicalSearcher double that records calls."""

    def __init__(self, results: list[RetrieverResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query, top_k, *, language=None):
        self.calls.append((query, top_k, language))
        return self.results[:top_k]


class TestRrfFuse:
    def test_interleaves_and_cuts_to_top_k(self) -> None:
        vec = [_result("a.md", "/h", 0), _result("b.md", "/h", 0)]
        lex = [_result("c.md", "/h", 0), _result("d.md", "/h", 0)]
        out = _rrf_fuse([(vec, 1.0), (lex, 1.0)], top_k=3, k=60)
        assert [r.chunk.source_file for r in out] == ["a.md", "c.md", "b.md"]
        assert len(out) == 3

    def test_duplicate_across_lists_keeps_vector_result(self) -> None:
        vec = [_result("a.md", "/h", 0, content="vector copy", score=0.9)]
        lex = [_result("a.md", "/h", 0, content="lexical copy", score=0.1)]
        out = _rrf_fuse([(vec, 1.0), (lex, 1.0)], top_k=5, k=60)
        assert len(out) == 1
        # Vector copy wins ties (kept first-seen); content + score preserved.
        assert out[0].chunk.content == "vector copy"
        assert out[0].score == 0.9

    def test_weighted_rrf_favours_heavier_list(self) -> None:
        vec = [_result("a.md", "/h", 0), _result("b.md", "/h", 0), _result("c.md", "/h", 0)]
        lex = [_result("x.md", "/h", 0)]
        light = _rrf_fuse([(vec, 1.0), (lex, 1.0)], top_k=4, k=60)
        heavy = _rrf_fuse([(vec, 1.0), (lex, 5.0)], top_k=4, k=60)
        # With equal weights the first vector ranks win; with heavy lexical
        # weight the lexical hit jumps to the top.
        assert light[0].chunk.source_file == "a.md"
        assert heavy[0].chunk.source_file == "x.md"


class TestHybridRetriever:
    def test_fuses_both_halves_and_propagates_language(self) -> None:
        vec = [_result("v1.md", "/h", 0), _result("v2.md", "/h", 0)]
        lex = [_result("l1.md", "/h", 0)]
        vector = FakeVectorRetriever(vec)
        lexical = FakeLexicalSearcher(lex)
        hybrid = HybridRetriever(vector, lexical, top_k_each=5, rrf_k=60)

        out = hybrid.retrieve("some query", top_k=3, language="en")
        assert len(out) == 3
        assert vector.calls == [("some query", 5, "en")]
        assert lexical.calls == [("some query", 5, "en")]
        assert "v1.md" in [r.chunk.source_file for r in out]

    def test_window_at_least_top_k_per_half(self) -> None:
        vec = FakeVectorRetriever([_result(f"v{i}.md", "/h", 0) for i in range(8)])
        lexical = FakeLexicalSearcher([_result(f"l{i}.md", "/h", 0) for i in range(8)])
        hybrid = HybridRetriever(vec, lexical, top_k_each=2, rrf_k=60)
        out = hybrid.retrieve("q", top_k=4)
        # Each half was pulled with max(top_k_each, top_k) = 4.
        assert vec.calls[0][1] == 4
        assert lexical.calls[0][1] == 4
        assert len(out) == 4


class _FakeCursor:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.executed_sql: str | None = None
        self.executed_params: list | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, rows) -> None:
        self._cursor = _FakeCursor(rows)

    def cursor(self):
        return self._cursor


class TestPostgresFTSSearcher:
    def test_empty_query_returns_no_results(self) -> None:
        searcher = PostgresFTSSearcher(_FakeConn([]))
        assert searcher.search("   ", top_k=5) == []

    def test_maps_rows_to_retriever_results(self) -> None:
        rows = [
            (
                "content here",
                "/tutorial/request",
                "en/docs/tutorial/request.md",
                2,
                '{"chunk_id": "det-id"}',
                0.95,
            )
        ]
        searcher = PostgresFTSSearcher(_FakeConn(rows))
        out = searcher.search("RequestValidationError", top_k=5, language="en")
        assert len(out) == 1
        r = out[0]
        assert r.chunk.source_file == "en/docs/tutorial/request.md"
        assert r.chunk.heading_path == "/tutorial/request"
        assert r.chunk.chunk_index == 2
        assert r.chunk.id == "det-id"  # recovered from metadata
        assert r.score == 0.95

    def test_sql_includes_language_filter_when_set(self) -> None:
        conn = _FakeConn([])
        searcher = PostgresFTSSearcher(conn)
        searcher.search("query terms", top_k=3, language="en")
        sql = conn.cursor().executed_sql
        assert "AND language = %s" in sql
        assert "LIMIT %s" in sql

    def test_invalid_fts_config_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="fts_config"):
            PostgresFTSSearcher(_FakeConn([]), fts_config="main; DROP TABLE chunks")

    def test_fts_config_bound_as_parameter_not_interpolated(self) -> None:
        """regconfig travels as %s::regconfig; no quoted config in SQL text."""
        conn = _FakeConn([])
        searcher = PostgresFTSSearcher(conn, fts_config="simple")
        searcher.search("query terms", top_k=3, language="en")
        cursor = conn.cursor()
        sql = cursor.executed_sql
        # Parameterized form for every regconfig use:
        assert "to_tsvector(%s::regconfig, content)" in sql
        assert "to_tsquery(%s::regconfig, %s)" in sql
        assert sql.count("%s::regconfig") == 4  # vector+query in SELECT and WHERE
        # The literal config never appears quoted in the SQL text...
        assert "'simple'" not in sql
        assert "'english'" not in sql
        # ...because it is bound as a parameter (2 per tsvector/tsquery pair).
        params = cursor.executed_params
        assert sum(1 for p in params if p == "simple") == 4
        assert params[-2:] == ["en", 3]  # language + LIMIT still follow

    @pytest.mark.integration
    def test_fts_against_real_postgres(self) -> None:
        """Real-PG sanity: exact identifier beats nothing, skips w/o PG."""
        try:
            from docpilot.db.connection import ensure_schema, get_connection
            from docpilot.retrieval.vector_store import PgVectorStore
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"docpilot deps unavailable: {exc}")

        try:
            conn = get_connection()
            ensure_schema(conn)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"PostgreSQL unavailable: {exc}")

        store = PgVectorStore(conn)
        searcher = PostgresFTSSearcher(conn)
        marker = "_integration_fts.md"
        try:
            store.delete_by_source([marker])
            unique_token = "zzzuniqueftsmarker"
            chunk = Chunk(
                id="fts-0",
                content=f"{unique_token} RequestValidationError is raised on invalid body data.",
                heading_path="/h",
                source_file=marker,
                chunk_index=0,
            )

            class _Dim384Provider:
                dimension = 384

                def embed(self, texts):
                    import numpy as np

                    out = np.zeros((len(texts), 384), dtype=np.float32)
                    for i, t in enumerate(texts):
                        out[i, hash(t) % 384] = 1.0
                    return out

            store.add([chunk], _Dim384Provider().embed([chunk.content]))

            hits = searcher.search(unique_token, top_k=5)
            assert any(h.chunk.source_file == marker for h in hits)
        finally:
            store.delete_by_source([marker])
            conn.close()