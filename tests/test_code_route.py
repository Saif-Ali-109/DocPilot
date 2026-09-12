"""Hermetic tests for Phase 6 T3: the code-route dispatch + T1→T2 wiring.

No LLM, no database — fakes drive both the code route (ask_code) and the
standard fallback (ask) through :func:`run_code_route`, and the T1 wiring is
asserted against the committed fixture corpus (unvalidated-vs-valid code).
"""

from __future__ import annotations

import pytest

from docpilot import config
from docpilot.agent.code_route import (
    CodeIntentClassifier,
    HeuristicCodeIntentClassifier,
    decide_code_route,
    run_code_route,
)
from docpilot.core.models import Chunk, RetrieverResult
from docpilot.validation import RetrieveThenValidate

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, *, top_k, language):
        self.calls.append((question, top_k, language))
        return self.results


class _FakeGenerator:
    """Understands both the answer path (generate_answer) and the code path
    (generate)."""

    def __init__(self, answer_text="", code_text=""):
        self.answer_text = answer_text
        self.code_text = code_text
        self.answer_calls = 0
        self.code_calls = 0

    def generate_answer(self, context_text, sources_text, question):
        self.answer_calls += 1
        return self.answer_text

    def generate(self, prompt):
        self.code_calls += 1
        return self.code_text


def _result(content, file, heading, cid):
    return RetrieverResult(
        chunk=Chunk(id=cid, content=content, heading_path=heading, source_file=file),
        score=0.9,
    )


TUTORIAL = _result(
    "```python\n"
    "from fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n'
    "```",
    "en/docs/tutorial/first-steps.md",
    "First Steps",
    "c1",
)

VALID_CODE = """\
Here is the app [1]:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```
"""

INVALID_CODE = """\
```python
from flask import Flask

app = Flask(__name__)
```
"""

# ---------------------------------------------------------------------------
# Classifier — zero LLM, deterministic
# ---------------------------------------------------------------------------


def test_classifier_detects_code_intent_phrases():
    classifier = HeuristicCodeIntentClassifier()
    assert classifier.classify("write code to call the API").code
    assert classifier.classify("show me the code for query params").code
    assert classifier.classify("What is query parameters?").code is False


def test_classifier_handles_empty_input():
    classifier = HeuristicCodeIntentClassifier()
    assert classifier.classify("").code is False
    assert classifier.classify("   ").code is False


def test_classifier_phrase_override():
    classifier = HeuristicCodeIntentClassifier(phrases=("cookie recipe",))
    assert classifier.classify("give me a cookie recipe").code
    assert classifier.classify("write code").code is False


def test_classifier_is_abc():
    assert issubclass(HeuristicCodeIntentClassifier, CodeIntentClassifier)


# ---------------------------------------------------------------------------
# Dispatch rules (§7.5: opt-in + intent, never the default path)
# ---------------------------------------------------------------------------


def test_dispatch_explicit_opt_in_wins_even_when_not_armed():
    decision = decide_code_route("anything at all", enabled=False, explicit=True)
    assert decision.code
    assert "explicit code opt-in" in decision.reason


def test_dispatch_requires_arming_for_intent():
    assert decide_code_route("write code", enabled=False).code is False
    assert decide_code_route("write code", enabled=True).code


def test_dispatch_falls_back_without_code_intent():
    decision = decide_code_route("how do i add a query parameter?", enabled=True)
    assert decision.code is False


def test_dispatch_uses_config_lever_by_default():
    # The route is OFF by default (never the default answer path) — this
    # protects the §7.5 contract against accidental arming.
    assert config.CODE_ROUTE_ENABLED is False
    assert config.CODE_INTENT_PHRASES


# ---------------------------------------------------------------------------
# run_code_route — routing + T1→T2 wiring (no loop, that is T4)
# ---------------------------------------------------------------------------


def test_non_code_query_falls_back_to_standard_ask():
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(
        answer_text="A query parameter is declared with a default value [1]."
    )
    result = run_code_route(
        "what is a query parameter?",
        enabled=True,
        retriever=retriever,
        generator=generator,
    )
    assert not result.took_code_route
    assert result.code is None
    assert result.ask is not None
    assert "query parameter" in result.answer
    assert generator.answer_calls == 1
    assert generator.code_calls == 0


def test_code_route_runs_and_attaches_verdicts():
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(code_text=VALID_CODE)
    result = run_code_route(
        "write code for a minimal fastapi app",
        enabled=True,
        retriever=retriever,
        generator=generator,
        validator=RetrieveThenValidate(),
    )
    assert result.took_code_route
    assert result.code is not None
    request = result.code
    assert len(request.code_blocks) == 1
    assert len(request.block_verdicts) == 1
    assert request.verdict is not None
    assert request.verdict.passed

    # The answer carries the citation footer (real doc page).
    assert "[1]" in request.answer
    assert "en/docs/tutorial/first-steps.md" in request.footer
    assert generator.code_calls == 1


def test_code_route_fails_unvalidated_code():
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(code_text=INVALID_CODE)
    result = run_code_route(
        "write code to serve requests",
        enabled=True,
        retriever=retriever,
        generator=generator,
        validator=RetrieveThenValidate(),
    )
    request = result.code
    assert request is not None
    assert request.verdict is not None
    assert not request.verdict.passed
    assert any("Flask" in reason for reason in request.verdict.reasons)


def test_explicit_opt_in_takes_code_route_without_lever():
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(code_text=VALID_CODE)
    result = run_code_route(
        "give me code",
        enabled=False,
        explicit_code=True,
        retriever=retriever,
        generator=generator,
    )
    assert result.took_code_route
    assert result.code is not None
    assert result.code.has_code


def test_validator_defaults_to_retrieve_then_validate():
    # T4: validator=None → structural validation runs by default (§7.5 —
    # unvalidated code is never returned).  T2-style raw output is only
    # available through ask_code() directly.
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(code_text=VALID_CODE)
    result = run_code_route(
        "write code",
        enabled=True,
        retriever=retriever,
        generator=generator,
        validator=None,
        max_validation_turns=0,
    )
    assert result.code is not None
    assert len(result.code.block_verdicts) == 1
    assert result.code.verdict is not None
    assert result.code.verdict.passed


def test_run_code_route_dispatches_to_standard_path_when_off():
    retriever = _FakeRetriever([TUTORIAL])
    generator = _FakeGenerator(answer_text="no code for you [1].")
    result = run_code_route(
        "write code",
        enabled=False,
        retriever=retriever,
        generator=generator,
    )
    assert not result.took_code_route
    assert "no code for you [1]." in result.answer