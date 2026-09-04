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
