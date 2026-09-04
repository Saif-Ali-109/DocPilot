from abc import ABC, abstractmethod

from docpilot.core.models import RetrieverResult


class Retriever(ABC):
    """Interface for high-level semantic retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrieverResult]:
        """Embed the query and return top_k results from the vector store."""
        ...
