from abc import ABC, abstractmethod

from docpilot.core.models import RetrieverResult


class Retriever(ABC):
    """Interface for high-level semantic retrieval."""

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Embed the query and return top_k results from the vector store.

        Args:
            query: The user's natural-language query.
            top_k: Maximum results to return.
            language: If set, restrict results to chunks matching this
                language tag. ``None`` disables filtering.
        """
        ...


class SimpleRetriever(Retriever):
    """Basic top-k vector retrieval.

    Embeds the query via an :class:`EmbeddingProvider`, searches a
    :class:`VectorStore`, and returns the results ordered by descending
    cosine similarity.  Optionally a :class:`Reranker` re-scores a wider
    candidate window before the final ``top_k`` cut (PLAN §H finding 1).

    Default ``top_k`` comes from ``config.RETRIEVAL_TOP_K`` (5) when the
    caller does not specify one.
    """

    def __init__(
        self,
        embedding_provider,
        vector_store,
        reranker=None,
        rerank_candidates: int | None = None,
    ) -> None:
        """Create a SimpleRetriever.

        Args:
            embedding_provider: An :class:`EmbeddingProvider` instance.
            vector_store: A :class:`VectorStore` instance.
            reranker: An optional :class:`Reranker`; when set, ``retrieve()``
                fetches ``rerank_candidates`` results and re-scores them down
                to ``top_k`` (if ``rerank_candidates`` is not given it comes
                from ``config.RERANK_CANDIDATES``).
            rerank_candidates: Candidate-window width for reranking; ignored
                when ``reranker`` is ``None``.
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def retrieve(
        self, query: str, top_k: int | None = None, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Embed *query* and return the top_k most similar chunks.

        Args:
            query: The user's natural-language query.
            top_k: Maximum results to return.  Defaults to
                ``config.RETRIEVAL_TOP_K`` if not provided.
            language: If set, restrict results to chunks matching this
                language tag. ``None`` disables filtering.

        Returns:
            A list of :class:`RetrieverResult` ordered by descending
            cosine similarity — or, when a reranker is configured, by
            descending reranked relevance.
        """
        if top_k is None:
            from docpilot import config
            top_k = config.RETRIEVAL_TOP_K

        # Embed the query (single-element list)
        query_embedding = self._embedding_provider.embed([query])[0]

        if self._reranker is None:
            results = self._vector_store.search(
                query_embedding, top_k=top_k, language=language
            )
            return results

        # Reranking pass: wider candidate window, then re-score to top_k.
        candidates = (
            self._rerank_candidates
            if self._rerank_candidates is not None
            else _rerank_candidates_default()
        )
        fetched = self._vector_store.search(
            query_embedding, top_k=max(candidates, top_k), language=language
        )
        return self._reranker.rerank(query, fetched, top_k)


def _rerank_candidates_default() -> int:
    from docpilot import config

    return config.RERANK_CANDIDATES
