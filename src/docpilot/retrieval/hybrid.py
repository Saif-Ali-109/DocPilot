"""Hybrid retrieval: dense-vector + lexical fused by score-free rank fusion.

``HybridRetriever`` composes a vector ``Retriever`` (Phase 1 dense path,
optionally reranked) and a ``LexicalSearcher`` (keyword path) and merges
their candidate lists with Reciprocal Rank Fusion — a rank-based merge that
needs no shared score scale, so cosine similarity and ``ts_rank`` fuse
cleanly (PLAN §H finding 2).
"""

from __future__ import annotations

from docpilot.core.models import Chunk, RetrieverResult
from docpilot.retrieval.retriever import Retriever


class HybridRetriever(Retriever):
    """Merge vector + lexical candidates with weighted Reciprocal Rank Fusion.

    Weaponised recall where the dense index is thin: exact identifiers, error
    codes, and config keys that BGE-small does not prioritise but Postgres
    full-text matches exactly.  Fused lists are ordered by descending RRF
    score; when a chunk appears in both halves the vector result object is
    kept (it carries the cosine similarity the judge/UI already understand).
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        lexical_searcher,
        *,
        top_k_each: int = 5,
        rrf_k: int = 60,
        weight_vector: float = 1.0,
        weight_lexical: float = 1.0,
    ) -> None:
        """Create the hybrid retriever.

        Args:
            vector_retriever: The dense ``Retriever`` (e.g. ``SimpleRetriever``,
                optionally reranked) to fuse.
            lexical_searcher: A ``LexicalSearcher`` (e.g. ``PostgresFTSSearcher``).
            top_k_each: Candidate-list width pulled from each half before
                fusion.  The merged length is bounded by ``2 * top_k_each``.
            rrf_k: The RRF smoothing constant (standard value 60).  Higher k
                flattens the ranking advantage of near-top positions.
            weight_vector: Relative weight of the vector half.
            weight_lexical: Relative weight of the lexical half.
        """
        self._vector = vector_retriever
        self._lexical = lexical_searcher
        self._top_k_each = top_k_each
        self._rrf_k = rrf_k
        self._weight_vector = weight_vector
        self._weight_lexical = weight_lexical

    def retrieve(
        self, query: str, top_k: int = 5, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Fetch both halves and return the fused top ``top_k``.

        Args:
            query: The user's natural-language query.
            top_k: Maximum results to return.
            language: If set, restrict both halves to chunks matching this
                language tag. ``None`` disables filtering.
        """
        requested = max(top_k, 1)
        each = max(self._top_k_each, requested)

        vector_hits = self._vector.retrieve(query, top_k=each, language=language)
        lexical_hits = self._lexical.search(query, each, language=language)

        return _rrf_fuse(
            [(vector_hits, self._weight_vector), (lexical_hits, self._weight_lexical)],
            top_k=requested,
            k=self._rrf_k,
        )


def _rrf_fuse(
    lists: list[tuple[list[RetrieverResult], float]], *, top_k: int, k: int
) -> list[RetrieverResult]:
    """Fuse ranked result lists with weighted Reciprocal Rank Fusion.

    Each list contributes ``weight / (k + rank)`` per item (rank 1-based).
    Duplicates (same source_file + heading_path + chunk_index) accumulate;
    the fused ordering sorts by descending RRF score.  Ties keep first-seen
    order (vector lists passed first win visually).
    """
    merged: dict[tuple[str, str, int], tuple[float, RetrieverResult]] = {}
    for results, weight in lists:
        for rank, result in enumerate(results, start=1):
            key = _identity(result)
            rrf = weight / (k + rank)
            if key in merged:
                existing_score, existing_result = merged[key]
                merged[key] = (existing_score + rrf, existing_result)
            else:
                merged[key] = (rrf, result)

    ranked = sorted(merged.values(), key=lambda pair: pair[0], reverse=True)
    return [result for score, result in ranked[:top_k]]


def _identity(result: RetrieverResult) -> tuple[str, str, int]:
    """Identity of a chunk for dedup across lists."""
    chunk: Chunk = result.chunk
    return (chunk.source_file, chunk.heading_path or "", chunk.chunk_index)