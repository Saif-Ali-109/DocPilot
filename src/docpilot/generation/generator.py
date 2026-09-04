"""Generator interface and Groq-backed implementation."""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod

from docpilot import config
from docpilot.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Generator(ABC):
    """Interface for LLM-based answer generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the LLM and return the generated answer string."""
        ...


class GroqGenerator(Generator):
    """Groq-backed answer generation with exponential backoff retry."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_key = api_key or config.GROQ_API_KEY
        self._model = model or config.GROQ_MODEL
        self._max_retries = max_retries if max_retries is not None else config.GROQ_MAX_RETRIES

        import groq as _groq

        self._client = _groq.Groq(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Convenience: build the full prompt from structured inputs and call
    # the primitive generate() method.
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        context_text: str,
        sources_text: str,
        question: str,
    ) -> str:
        """Build the full prompt by substituting into SYSTEM_PROMPT and
        delegate to generate()."""
        full_prompt = SYSTEM_PROMPT.format(
            context=context_text,
            sources=sources_text,
            question=question,
        )
        return self.generate(full_prompt)

    # ------------------------------------------------------------------
    # Primitive interface method
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Call the Groq chat completions API with retry/backoff.

        Retry only on transient failures (429, 5xx, connection errors).
        Non-transient errors (4xx other than 429) are raised immediately.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()  # type: ignore[union-attr]

            except Exception as exc:
                last_exc = exc
                if self._is_transient(exc):
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Groq request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    # Non-transient: fail fast
                    raise

        # Exhausted all retries
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Return True if the exception represents a transient / retryable error."""
        # Connection-level errors (requests, httpx, urllib3, etc.)
        transient_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        if isinstance(exc, transient_types):
            return True

        # groq.APIStatusError carries a status_code attribute
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status in (429, 500, 502, 503, 504)

        return False

    @staticmethod
    def _backoff(attempt: int, base: float = 1.0) -> float:
        """Exponential backoff with full jitter.

        Returns a delay in seconds: random(0, base * 2^attempt).
        """
        return random.uniform(0, base * (2 ** attempt))
