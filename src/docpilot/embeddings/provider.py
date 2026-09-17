import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


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
            import time

            from sentence_transformers import SentenceTransformer

            _load_started = time.perf_counter()
            self._model = SentenceTransformer(self._model_name, device="cpu")
            logger.info(
                "Loaded embedding model %s in %.1f s",
                self._model_name,
                time.perf_counter() - _load_started,
            )
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


# ---------------------------------------------------------------------------
# Process-wide default provider
# ---------------------------------------------------------------------------
# _build_default_retriever() used to construct a fresh BGEEmbeddingProvider per
# call, which lazy-loaded the ~450 MB BGE-small model from disk on every
# ask()/API/agentic run (~16 s each — measured). The default provider below is
# created lazily once per process and reused; embed() is a pure function of
# its inputs, so sharing one instance never changes retrieval results. The
# lock makes the lazy init safe when two requests race on the first call.
# Direct BGEEmbeddingProvider(model_name=...) construction is unaffected and
# is the (uncached) path for custom models.

_default_provider: "BGEEmbeddingProvider | None" = None
_default_provider_lock = threading.Lock()


def get_default_embedding_provider() -> "BGEEmbeddingProvider":
    """Return the process-wide BGE provider, loading the model once.

    The model (weights + tokenizer) is loaded from disk exactly once per
    process and reused by every retriever and ingest run. The first call
    pays the one-time load; every subsequent call is a cache hit. Safe to
    call from multiple threads (FastAPI worker threads): concurrent first
    callers serialize on the lock and share one instance.
    """
    global _default_provider
    if _default_provider is None:
        with _default_provider_lock:
            if _default_provider is None:  # double-checked, race-safe
                _default_provider = BGEEmbeddingProvider()
    return _default_provider
