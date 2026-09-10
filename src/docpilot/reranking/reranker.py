"""Reranker interface + cross-encoder implementation (PLAN §H finding 1).

Pure cosine top-k retrieval orders candidates by embedding similarity alone;
for technical docs dense vectors can underweight exact API names / error
codes that a cross-encoder re-scoring pass picks up.  The pipeline fetches a
wider candidate window from the vector store, runs the ``Reranker`` over it,
and keeps the re-scored ``top_k``.

The implementation uses ``BAAI/bge-reranker-base`` via sentence-transformers
``CrossEncoder`` (already a project dependency — same local-model pattern as
BGE-small embeddings).  Model files are downloaded lazily on first use, so
the live environment needs one-time HuggingFace access per model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from docpilot.core.models import RetrieverResult


class Reranker(ABC):
    """Interface for re-scoring retrieved chunks against the query."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrieverResult], top_k: int
    ) -> list[RetrieverResult]:
        """Re-score *results* for *query* and return the re-ordered ``top_k``.

        Args:
            query: The user's natural-language query.
            results: Candidate results (typically a wider window than top_k).
            top_k: Number of highest-scoring results to return; the list is
                ordered descending by the re-scored relevance.

        Returns:
            At most ``top_k`` :class:`RetrieverResult` objects ordered by
            descending re-scored relevance.  ``score`` carries the reranker's
            relevance score (scale depends on the implementation).
        """
        ...


class BCEReranker(Reranker):
    """Cross-encoder reranker over ``BAAI/bge-reranker-base`` (CPU).

    The model is loaded lazily on the first call (mirroring
    :class:`~docpilot.embeddings.provider.BGEEmbeddingProvider`) and reused
    for the lifetime of the instance.  Scores are the raw cross-encoder
    relevance logits — higher means more relevant; the absolute value is
    model-specific and should only be compared *within* a reranked set.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from docpilot import config

        self._model_name = model_name or config.RERANKER_MODEL
        self._model = None  # lazy-loaded

    def _get_or_load_model(self):
        """Return the loaded CrossEncoder, loading it lazily on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu")
        return self._model

    def rerank(
        self, query: str, results: list[RetrieverResult], top_k: int
    ) -> list[RetrieverResult]:
        """Re-score *results* with the cross-encoder and return the top_k.

        Args:
            query: The query string the candidates were retrieved for.
            results: Candidate results (ordered however the store returned
                them — ordering is ignored here).
            top_k: Number of results to return.

        Returns:
            At most ``top_k`` results re-ordered by descending cross-encoder
            relevance, with ``score`` replaced by the reranker's logit.
        """
        if not results:
            return []
        if top_k <= 0:
            return []

        model = self._get_or_load_model()
        pairs = [(query, r.chunk.content) for r in results]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(results, scores), key=lambda pair: pair[1], reverse=True
        )
        picked = ranked[:top_k]
        return [
            _result_with_score(result, float(score)) for result, score in picked
        ]


def _result_with_score(result: RetrieverResult, score: float) -> RetrieverResult:
    """Return *result* with its ``score`` replaced by *score* (dataclasses.replace)."""
    from dataclasses import replace

    return replace(result, score=score)