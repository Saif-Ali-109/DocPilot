"""Retry/backoff behavior of GroqGenerator — 429 retry-after honoring (SPEC §6.5).

The generator must honor a short server retry-after (TPM self-heal), cap a
huge one (daily quota wall fails fast instead of hanging retries), and fail
fast on non-transient errors.
"""
import types

import pytest

from docpilot.generation.generator import (
    GroqGenerator,
    _RETRY_AFTER_MAX_SECONDS,
)


class _FakeRateLimit(Exception):
    """Stand-in for groq.RateLimitError (a 429 carrying message + headers)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = types.SimpleNamespace(headers=headers)


class _FakeCompletions:
    def __init__(self, responses):
        # responses: list of Exception | str content, popped in order
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):  # noqa: ARG002 - fake API surface
        self.calls += 1
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content=r))
            ]
        )


class _FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(responses))


def _gen(responses, max_retries: int = 3) -> GroqGenerator:
    gen = GroqGenerator(api_key="test-key", model="test-model", max_retries=max_retries)
    gen._client = _FakeClient(responses)  # noqa: SLF001 - test seam
    return gen


# ---------------------------------------------------------------------------
# _server_retry_after parsing
# ---------------------------------------------------------------------------


class TestServerRetryAfter:
    def test_parses_header(self):
        exc = _FakeRateLimit("rate limit", retry_after="0.5")
        assert GroqGenerator._server_retry_after(exc) == 0.5

    def test_parses_header_seconds(self):
        exc = _FakeRateLimit("rate limit", retry_after="10")
        assert GroqGenerator._server_retry_after(exc) == 10.0

    def test_header_wins_over_message(self):
        exc = _FakeRateLimit("Please try again in 1m4.8s.", retry_after="0.25")
        assert GroqGenerator._server_retry_after(exc) == 0.25

    def test_parses_millisecond_message(self):
        exc = _FakeRateLimit("Rate limit reached ... Please try again in 495ms. Need more?", retry_after=None)
        assert GroqGenerator._server_retry_after(exc) == pytest.approx(0.495)

    def test_parses_minutes_message(self):
        exc = _FakeRateLimit("... Please try again in 1m4.8s.", retry_after=None)
        assert GroqGenerator._server_retry_after(exc) == 60.0

    def test_parses_long_minutes_message(self):
        exc = _FakeRateLimit("... Please try again in 13m29.136s.", retry_after=None)
        assert GroqGenerator._server_retry_after(exc) == pytest.approx(780.0)

    def test_missing_wait_returns_none(self):
        exc = _FakeRateLimit("Generic rate limit", retry_after=None)
        assert GroqGenerator._server_retry_after(exc) is None


# ---------------------------------------------------------------------------
# generate() retry behavior
# ---------------------------------------------------------------------------


class TestGenerateRetry:
    def test_recovers_with_short_server_wait(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)
        gen = _gen(
            [
                _FakeRateLimit("Please try again in 495ms.", retry_after=None),
                "OK",
            ],
            max_retries=3,
        )
        assert gen.generate("hi") == "OK"
        assert sleeps == [pytest.approx(0.495)]

    def test_caps_long_wait_and_fails_fast(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)
        gen = _gen(
            [
                _FakeRateLimit("Please try again in 300s.", retry_after=None),
                _FakeRateLimit("Please try again in 300s.", retry_after=None),
                _FakeRateLimit("Please try again in 300s.", retry_after=None),
            ],
            max_retries=3,
        )
        with pytest.raises(_FakeRateLimit):
            gen.generate("hi")
        # daily-quota wall: every wait is capped, never the server's 300s
        assert sleeps == [_RETRY_AFTER_MAX_SECONDS] * 3

    def test_header_wait_is_capped_too(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)
        gen = _gen(
            [_FakeRateLimit("rate limit", retry_after="999")] * 3,
            max_retries=3,
        )
        with pytest.raises(_FakeRateLimit):
            gen.generate("hi")
        assert sleeps == [_RETRY_AFTER_MAX_SECONDS] * 3

    def test_falls_back_to_jitter_without_server_wait(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)
        monkeypatch.setattr(
            GroqGenerator,
            "_backoff",
            staticmethod(lambda attempt, base=1.0: 0.25),
        )
        gen = _gen(
            [_FakeRateLimit("generic", retry_after=None)] * 3,
            max_retries=3,
        )
        with pytest.raises(_FakeRateLimit):
            gen.generate("hi")
        assert sleeps == [0.25] * 3

    def test_non_transient_fails_immediately(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)

        class _BadRequest(Exception):
            pass

        bad = _BadRequest("400")
        bad.status_code = 400
        gen = _gen([bad], max_retries=3)
        with pytest.raises(_BadRequest):
            gen.generate("hi")
        assert sleeps == []

    def test_success_on_first_attempt_when_no_error(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("docpilot.generation.generator.time.sleep", sleeps.append)
        assert _gen(["OK"], max_retries=3).generate("hi") == "OK"
        assert sleeps == []