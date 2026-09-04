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


class BGEEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by ``BAAI/bge-small-en-v1.5`` (384-dim).

    The model is loaded lazily on the first call to :meth:`embed` and
    reused for the lifetime of the instance (singleton-ish per process).

    Vectors are L2-normalized so that inner-product search in pgvector is
    equivalent to cosine similarity.
    """

    _expected_dimension: int = 384

    def __init__(self, model_name: str | None = None) -> None:
        from docpilot import config

        self._model_name = model_name or config.EMBEDDING_MODEL
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_or_load_model(self):
        """Return the loaded model, loading it lazily on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* and return an ``(n, 384)`` unit-normed array.

        Args:
            texts: A list of strings (always a list — never a bare string).

        Returns:
            numpy array of shape ``(len(texts), 384)`` with each row
            L2-normalized to unit length.
        """
        if not isinstance(texts, list):
            raise TypeError(f"embed() expects a list[str], got {type(texts).__name__}")
        if len(texts) == 0:
            return np.zeros((0, self._expected_dimension), dtype=np.float32)

        model = self._get_or_load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for BGE-small)."""
        return self._expected_dimension
