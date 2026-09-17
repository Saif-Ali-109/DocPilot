"""Tests for BGEEmbeddingProvider.

The BGE model (~130 MB) downloads once to ``~/.cache/huggingface`` on first
use.  If no network is available the download will fail and the test is
skipped with a clear message.
"""

import threading
import time

import numpy as np
import pytest

try:
    from docpilot.embeddings.provider import BGEEmbeddingProvider

    _IMPORT_OK = True
except Exception as _exc:
    _IMPORT_OK = False
    _IMPORT_ERROR = _exc


@pytest.fixture(scope="module")
def provider() -> BGEEmbeddingProvider:
    """Create a BGEEmbeddingProvider, skip if model unavailable."""
    if not _IMPORT_OK:
        pytest.skip(f"Could not import BGEEmbeddingProvider: {_IMPORT_ERROR}")
    p = BGEEmbeddingProvider()
    try:
        # Trigger the lazy model load during fixture setup.
        # If the model isn't cached and there's no network, this will raise.
        p.embed(["model load probe"])
    except Exception as exc:
        pytest.skip(f"BGE model unavailable (no network or download failure): {exc}")
    return p


class TestBGEEmbeddingProvider:
    """Validate BGEEmbeddingProvider contract."""

    def test_single_string(self, provider: BGEEmbeddingProvider) -> None:
        """Embedding a single string returns shape (1, 384)."""
        result = provider.embed(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384), f"Expected (1, 384), got {result.shape}"

    def test_multiple_strings(self, provider: BGEEmbeddingProvider) -> None:
        """Embedding three strings returns shape (3, 384)."""
        texts = ["hello world", "fastapi documentation", "install fastapi"]
        result = provider.embed(texts)
        assert result.shape == (3, 384), f"Expected (3, 384), got {result.shape}"

    def test_all_finite(self, provider: BGEEmbeddingProvider) -> None:
        """All embedding values should be finite (no NaN/Inf)."""
        result = provider.embed(["test text", "another text"])
        assert np.all(np.isfinite(result)), "Embeddings contain NaN or Inf values"

    def test_normalized(self, provider: BGEEmbeddingProvider) -> None:
        """Each row should be approximately unit-normed (L2 norm ≈ 1)."""
        result = provider.embed(["hello world", "test embedding", "fastapi docs"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(
            norms, 1.0, atol=1e-5,
            err_msg=f"Embedding norms not close to 1.0: {norms}",
        )

    def test_dimension_property(self, provider: BGEEmbeddingProvider) -> None:
        """dimension property should return 384."""
        assert provider.dimension == 384

    def test_empty_list(self, provider: BGEEmbeddingProvider) -> None:
        """An empty list returns an empty array of shape (0, 384)."""
        result = provider.embed([])
        assert result.shape == (0, 384)

    def test_rejects_bare_string(self, provider: BGEEmbeddingProvider) -> None:
        """A bare string (not a list) should raise TypeError."""
        with pytest.raises(TypeError):  # type: ignore[arg-type]
            provider.embed("not a list")  # type: ignore[arg-type]

    def test_deterministic(self, provider: BGEEmbeddingProvider) -> None:
        """Same input should produce identical embeddings."""
        result1 = provider.embed(["deterministic test"])
        result2 = provider.embed(["deterministic test"])
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# get_default_embedding_provider() — process-wide singleton + race safety
# ---------------------------------------------------------------------------


class _CountingSlowProvider:
    """Fake replacing BGEEmbeddingProvider for the singleton tests.

    The constructor sleeps to widen any race window, and counts how many
    instances were created, so the tests can assert the lazy init builds
    exactly one provider even under concurrency. No model/network involved.
    """

    created = 0
    _count_lock = threading.Lock()

    def __init__(self, *args: object, **kwargs: object) -> None:
        with type(self)._count_lock:
            type(self).created += 1
        time.sleep(0.05)


def test_default_provider_is_cached(monkeypatch) -> None:
    """get_default_embedding_provider() returns the same instance every call."""
    import docpilot.embeddings.provider as provider_module

    monkeypatch.setattr(provider_module, "_default_provider", None)
    monkeypatch.setattr(
        provider_module, "BGEEmbeddingProvider", _CountingSlowProvider
    )
    _CountingSlowProvider.created = 0

    first = provider_module.get_default_embedding_provider()
    second = provider_module.get_default_embedding_provider()

    assert first is second
    assert _CountingSlowProvider.created == 1


def test_default_provider_race_constructs_once(monkeypatch) -> None:
    """Concurrent first calls must construct exactly one shared provider."""
    import docpilot.embeddings.provider as provider_module

    monkeypatch.setattr(provider_module, "_default_provider", None)
    monkeypatch.setattr(
        provider_module, "BGEEmbeddingProvider", _CountingSlowProvider
    )
    _CountingSlowProvider.created = 0

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    instances: list[object] = []
    errors: list[BaseException] = []

    def _call() -> None:
        try:
            barrier.wait()
            instances.append(provider_module.get_default_embedding_provider())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == [], f"unexpected thread errors: {errors}"
    assert _CountingSlowProvider.created == 1, (
        "expected exactly 1 construction under concurrency, "
        f"got {_CountingSlowProvider.created}"
    )
    assert len({id(p) for p in instances}) == 1, (
        "all threads must share the same provider instance"
    )
