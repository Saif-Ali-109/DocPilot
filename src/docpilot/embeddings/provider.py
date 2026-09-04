from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Interface for generating text embeddings."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return embeddings as a 2D numpy array."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g. 384 for BGE-small)."""
        ...
