"""Generator interface and Groq-backed implementation."""

from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from docpilot import config
from docpilot.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Maximum wait (seconds) honored from a server retry-after on a rate-limit
# error. A short TPM wait (~0.5s) is honored so the request self-heals; a
# huge TPD wait (minutes) is capped so the daily wall fails fast instead of
# hanging the retry loop for ~3× the advertised wait.
_RETRY_AFTER_MAX_SECONDS = 10.0

# Appended to a prompt when the hosted model emits a tool call despite no
# tools being declared (Groq 400 code tool_use_failed — "Tool choice is none,
# but model called a tool") or returns an empty/whitespace-only completion.
# Model-side anomalies, retried once with this guard.
_TOOL_USE_GUARD_SUFFIX = (
    "\n\nRespond in plain prose only. Do not call any tools, do not emit "
    "function-call JSON, and do not reference file paths or line ranges."
)


class _EmptyCompletion(Exception):
    """Raised internally when the model returns an empty/whitespace completion."""


def _int_header(headers, name: str) -> int | None:
    """Parse an integer response header (Groq's x-ratelimit-* values)."""
    val = headers.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a one-call probe (SPEC §6.5).

    ``limit``/``used``/``remaining`` reflect Groq's **per-minute** token bucket
    (8k/min on the free tier) as reported by response headers. Groq does NOT
    expose the daily (TPD) token bucket via the API, so no probe can read it —
    TPD risk is carried by the run machinery, not this gate.
    """

    ok: bool
    completion: str | None = None
    limit: int | None = None
    used: int | None = None
    remaining: int | None = None
    requests_remaining: int | None = None
    reason: str = ""


class Generator(ABC):
    """Interface for LLM-based answer generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the LLM and return the generated answer string."""
        ...

    def generate_system_user(self, system: str, user: str) -> str:
        """Send separate *system* and *user* messages to the LLM.

        Default implementation concatenates them into a single prompt and
        delegates to :meth:`generate` — byte-identical to today's behaviour
        for all concrete fakes that only implement ``generate()``.
        """
        return self.generate(f"{system}\n\n{user}")

    def generate_stream(self, prompt: str):
        """Yield text deltas for *prompt* (Phase 5 streaming, SPEC §7).

        Default: not supported — raises ``NotImplementedError``.  Concrete
        streaming implementations (``GroqGenerator``) override this; callers
        that need streaming must fall back to :meth:`generate` when this
        raises, keeping every non-streaming fake byte-identical.
        """
        raise NotImplementedError("generate_stream not supported")

    def generate_answer_stream(self, context_text: str, sources_text: str, question: str):
        """Streaming variant of the answered-prompt helper.

        Builds the full prompt from ``SYSTEM_PROMPT`` and yields text deltas.
        Default: raises ``NotImplementedError`` (see :meth:`generate_stream`).
        """
        raise NotImplementedError("generate_answer_stream not supported")


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
        # Per-call token accounting (SPEC §7 hardening — follow-up §5.3):
        # the most recent call's Groq `usage` dict, or None.  The API service
        # surfaces this in the SSE `done` event; the eval harness integration
        # stays a recorded follow-up.
        self.last_usage: dict | None = None

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

    def generate_answer_stream(self, context_text: str, sources_text: str, question: str):
        """Streaming variant of :meth:`generate_answer`.

        Builds the same ``SYSTEM_PROMPT`` prompt and yields text deltas from
        :meth:`generate_stream` instead of returning one string.
        """
        full_prompt = SYSTEM_PROMPT.format(
            context=context_text,
            sources=sources_text,
            question=question,
        )
        yield from self.generate_stream(full_prompt)

    # ------------------------------------------------------------------
    # Primitive interface method
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Call the Groq chat completions API with retry/backoff.

        Retry only on transient failures (429, 5xx, connection errors).
        Non-transient errors (4xx other than 429) are raised immediately —
        except the hosted-model tool-use glitch (400 ``tool_use_failed``:
        the model emits a tool call despite no tools being declared), which
        is retried once with a plain-text guard appended.

        On a rate-limit error the server's ``retry-after`` is honored
        (bounded by ``_RETRY_AFTER_MAX_SECONDS``); without one, exponential
        jitter applies.
        """
        messages = [{"role": "system", "content": prompt}]
        return self._complete(messages, prompt)

    def generate_system_user(self, system: str, user: str) -> str:
        """Send separate ``system`` and ``user`` messages to Groq.

        Same retry/backoff and tool-use-guard machinery as :meth:`generate`.
        The system message is sent as ``role: system``, the user message as
        ``role: user`` — both through the shared :meth:`_complete` helper.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete(messages, f"{system}\n\n{user}")

    # ------------------------------------------------------------------
    # Internal send loop (shared by generate / generate_system_user)
    # ------------------------------------------------------------------

    def _complete(self, messages: list[dict], original_prompt: str) -> str:
        """Send *messages* to Groq with retry/backoff.

        *original_prompt* is the single-string prompt used for the
        tool-use-guard fallback comparison (``effective_prompt is
        original_prompt``).  For ``generate()`` it is the raw prompt; for
        ``generate_system_user()`` it is the concatenated form — both paths
        use the same guard-suffix logic.
        """
        last_exc: Exception | None = None
        effective_prompt = original_prompt

        for attempt in range(self._max_retries):
            try:
                # Build the messages list for this attempt.  On the first
                # attempt ``effective_prompt is original_prompt`` so we use the
                # caller-supplied messages verbatim.  When the tool-use/empty
                # glitch fires we replace the system message content with the
                # guarded prompt (and clear any user message — same as the
                # original single-message path).
                if effective_prompt is original_prompt:
                    attempt_messages = messages
                else:
                    attempt_messages = [{"role": "system", "content": effective_prompt}]

                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=attempt_messages,
                    temperature=0,
                )
                content = response.choices[0].message.content  # type: ignore[union-attr]
                self._record_usage(getattr(response, "usage", None))
                if not content or not content.strip():
                    raise _EmptyCompletion("empty completion")
                return content.strip()

            except Exception as exc:
                last_exc = exc
                if effective_prompt is original_prompt and (
                    self._is_tool_use_glitch(exc) or isinstance(exc, _EmptyCompletion)
                ):
                    effective_prompt = original_prompt + _TOOL_USE_GUARD_SUFFIX
                    logger.warning(
                        "Groq tool-use/empty glitch (attempt %d/%d): %s — retrying "
                        "with plain-prose guard",
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    continue
                if self._is_transient(exc):
                    server_wait = self._server_retry_after(exc)
                    if server_wait is not None:
                        # Honor a short wait (TPM self-heal) but fail fast on
                        # a wall-of-death wait (exhausted daily quota).
                        wait = min(server_wait, _RETRY_AFTER_MAX_SECONDS)
                    else:
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

    def generate_stream(self, prompt: str):
        """Stream *prompt* from Groq, yielding text deltas (Phase 5, SPEC §7).

        Same retry frame as :meth:`generate` (transient 429/5xx/connection
        errors, the hosted-model tool-use glitch, empty completions), but only
        *until the first delta is yielded* — a failure mid-stream cannot be
        retried without corrupting the consumer's already-received text, so it
        propagates.  On success records `usage` (streamed responses expose it
        after full iteration) into ``self.last_usage``.
        """
        last_exc: Exception | None = None
        effective_prompt = prompt
        yielded_any = False

        for attempt in range(self._max_retries):
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": effective_prompt,
                        },
                    ],
                    temperature=0,
                    stream=True,
                    # groq<=1.7.0's typed surface rejects stream_options as a
                    # direct kwarg but forwards extra_body into the request —
                    # the API then tags the final chunk with usage.
                    extra_body={"stream_options": {"include_usage": True}},
                )
                produced = False
                last_chunk = None
                for chunk in stream:
                    last_chunk = chunk
                    choices = getattr(chunk, "choices", None) or []
                    delta = (choices[0].delta.content or "") if choices else ""
                    if delta:
                        produced = True
                        yielded_any = True
                        yield delta
                if not produced:
                    # The request succeeded but the model gave no text —
                    # treat like the empty-completion glitch (retry once with
                    # the plain-prose guard).
                    raise _EmptyCompletion("empty stream completion")
                # Streamed usage arrives on the final chunk when
                # stream_options=include_usage is set (OpenAI-compatible SDKs);
                # some SDK versions only expose it on the stream object.
                self._record_usage(
                    getattr(stream, "usage", None)
                    or (getattr(last_chunk, "usage", None) if last_chunk is not None else None)
                )
                return

            except Exception as exc:
                last_exc = exc
                if yielded_any:
                    # Started streaming and failed mid-stream: propagate —
                    # the consumer already holds deltas, no retry is safe.
                    raise
                if effective_prompt is prompt and (
                    self._is_tool_use_glitch(exc) or isinstance(exc, _EmptyCompletion)
                ):
                    effective_prompt = prompt + _TOOL_USE_GUARD_SUFFIX
                    logger.warning(
                        "Groq stream tool-use/empty glitch (attempt %d/%d): %s "
                        "— retrying with plain-prose guard",
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    continue
                if self._is_transient(exc):
                    server_wait = self._server_retry_after(exc)
                    if server_wait is not None:
                        wait = min(server_wait, _RETRY_AFTER_MAX_SECONDS)
                    else:
                        wait = self._backoff(attempt)
                    logger.warning(
                        "Groq stream failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    # Non-transient: fail fast
                    raise

        raise last_exc  # type: ignore[misc]

    def _record_usage(self, usage) -> None:
        """Store the last call's Groq ``usage`` object (token accounting).

        Attribute-defensive: SDK versions vary (``Usage`` dataclass vs dict).
        ``None`` input or missing fields leave ``last_usage`` as-is.
        """
        if usage is None:
            return
        record: dict = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, name, None)
            if isinstance(value, int):
                record[name] = value
        if record:
            self.last_usage = record

    def probe(self) -> ProbeResult:
        """One minimal completions call that reads the per-minute rate-limit
        headers (``x-ratelimit-*-tokens`` / ``-requests``).

        Functional gate: verifies the key authenticates, the model is
        reachable, and a real call can be served right now. Groq exposes only
        per-minute buckets in headers — the daily (TPD) token bucket that
        actually kills long runs is NOT exposed, so this probe makes no claim
        about daily headroom (SPEC §6.5). Single attempt, no retries.
        """
        try:
            raw = self._client.chat.completions.with_raw_response.create(
                model=self._model,
                messages=[{"role": "system", "content": "Reply exactly OK."}],
                temperature=0,
            )
            headers = getattr(raw, "headers", None) or {}
            content = raw.parse().choices[0].message.content.strip()  # type: ignore[union-attr]
            return ProbeResult(
                ok=True,
                completion=content,
                limit=_int_header(headers, "x-ratelimit-limit-tokens"),
                used=_int_header(headers, "x-ratelimit-used-tokens"),
                remaining=_int_header(headers, "x-ratelimit-remaining-tokens"),
                requests_remaining=_int_header(headers, "x-ratelimit-remaining-requests"),
            )
        except Exception as exc:  # noqa: BLE001 - any failure becomes a verdict
            return ProbeResult(ok=False, reason=f"{type(exc).__name__}: {exc}")

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
    def _is_tool_use_glitch(exc: Exception) -> bool:
        """True for Groq 400 ``tool_use_failed`` — the hosted model emitted a
        tool call although the request declared no tools and the server's
        ``tool_choice`` is none. A model-side glitch (not a genuine bad
        request); retried once with a plain-prose guard (2026-09-09 live
        failure on the GroundingChecker NLI call).
        """
        status = getattr(exc, "status_code", None)
        if status != 400:
            return False
        body = str(getattr(exc, "message", "") or exc)
        return (
            "tool_use_failed" in body
            or "Tool choice is none" in body
            or "model called a tool" in body
        )

    @staticmethod
    def _server_retry_after(exc: Exception) -> float | None:
        """Seconds the server asked us to wait, if it told us.

        Prefers the HTTP ``retry-after`` header; falls back to Groq's message
        text ("Please try again in 495ms. / in 1m4.8s. / in 13m29.136s.").
        Returns ``None`` when the server gave no usable wait.
        """
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            val = headers.get("retry-after")
            if val is not None:
                try:
                    return max(0.0, float(val))
                except (TypeError, ValueError):
                    pass

        body = str(getattr(exc, "message", "") or exc)
        m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s|m)", body)
        if m:
            seconds = float(m.group(1))
            unit = m.group(2)
            if unit == "ms":
                return seconds / 1000.0
            if unit == "m":
                return seconds * 60.0
            return seconds
        return None

    @staticmethod
    def _backoff(attempt: int, base: float = 1.0) -> float:
        """Exponential backoff with full jitter.

        Returns a delay in seconds: random(0, base * 2^attempt).
        """
        return random.uniform(0, base * (2 ** attempt))
