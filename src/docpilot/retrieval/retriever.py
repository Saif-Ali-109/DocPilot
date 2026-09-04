from abc import ABC, abstractmethod

from docpilot.core.models import RetrieverResult


class Retriever(ABC):
    """Interface for high-level semantic retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrieverResult]:
        """Embed the query and return top_k results from the vector store."""
        ...


class SimpleRetriever(Retriever):
    """Basic top-k vector retrieval.

    Embeds the query via an :class:`EmbeddingProvider`, searches a
    :class:`VectorStore`, and returns the results ordered by descending
    cosine similarity.

    Default ``top_k`` comes from ``config.RETRIEVAL_TOP_K`` (5) when the
    caller does not specify one.
    """

    def __init__(self, embedding_provider, vector_store) -> None:
        """Create a SimpleRetriever.

        Args:
            embedding_provider: An :class:`EmbeddingProvider` instance.
            vector_store: A :class:`VectorStore` instance.
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrieverResult]:
        """Embed *query* and return the top_k most similar chunks.

        Args:
            query: The user's natural-language query.
            top_k: Maximum results to return.  Defaults to
                ``config.RETRIEVAL_TOP_K`` if not provided.

        Returns:
            A list of :class:`RetrieverResult` ordered by descending
            cosine similarity score.
        """
        if top_k is None:
            from docpilot import config
            top_k = config.RETRIEVAL_TOP_K

        # Embed the query (single-element list)
        query_embedding = self._embedding_provider.embed([query])[0]

        # Search the vector store
        results = self._vector_store.search(query_embedding, top_k=top_k)

        return results
