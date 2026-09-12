"""Hermetic tests for Phase 6 T4: the validation/reformulation loop.

No LLM, no database — a sequence-based fake drives the loop through
generate → validate → reformulate → ... and the exit behaviours are asserted:
fully validated output is returned, model refusals break the loop, and a
still-failing candidate after the budget is refused with sources instead of
being returned (§7.5: unvalidated code is never returned).
"""

from __future__ import annotations

from docpilot.agent.code_route import run_code_route
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


class _SequenceGenerator:
    """Returns the next canned response per ``generate`` call (the loop's
    reformulations), recording every prompt it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("generator exhausted — loop over-spent budget")
        return self.responses.pop(0)


def _result(content, file, heading, cid):
    return RetrieverResult(
        chunk=Chunk(id=cid, content=content, heading_path=heading, source_file=file),
        score=0.9,
    )


_EVIDENCE = _result(
    "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n'
    "```",
    "en/docs/tutorial/first-steps.md",
    "First Steps",
    "c1",
)

_GOOD_CODE = """\
Here is the minimal app [1]:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```
"""

_BAD_IMPORT_CODE = """\
```python
from flask import Flask

app = Flask(__name__)
```
"""

_REFUSAL = "I don't know — the available documentation does not cover this question."

# ---------------------------------------------------------------------------


def _run(*, responses, turns=2, enabled=True):
    retriever = _FakeRetriever([_EVIDENCE])
    generator = _SequenceGenerator(responses)
    result = run_code_route(
        "write code for a minimal fastapi app",
        enabled=enabled,
        retriever=retriever,
        generator=generator,
        validator=RetrieveThenValidate(),
        max_validation_turns=turns,
    )
    assert result.took_code_route
    assert result.code is not None
    return generator, result.code


def test_first_attempt_validated_passes_without_reformulation():
    generator, request = _run(responses=[_GOOD_CODE])
    assert len(generator.prompts) == 1
    assert request.verdict is not None
    assert request.verdict.passed
    assert not request.validation_failed
    assert request.validation_reasons == []
    assert "first-steps.md" in request.footer  # cited, real doc page


def test_failure_reformulates_once_then_passes():
    generator, request = _run(responses=[_BAD_IMPORT_CODE, _GOOD_CODE])
    assert len(generator.prompts) == 2  # initial + one reformulation
    assert request.verdict is not None
    assert request.verdict.passed
    assert not request.validation_failed

    # The reformulation prompt carried the failure reasons back to the model.
    fix_prompt = generator.prompts[1]
    assert "VALIDATION FAILURES FROM THE PREVIOUS ATTEMPT" in fix_prompt
    assert "Flask" in fix_prompt


def test_persistent_failure_refuses_with_sources():
    generator, request = _run(responses=[_BAD_IMPORT_CODE, _BAD_IMPORT_CODE, _BAD_IMPORT_CODE])
    assert len(generator.prompts) == 3  # 1 initial + 2 reformulations (turns=2)
    assert request.validation_failed
    assert request.verdict is not None
    assert not request.verdict.passed
    assert any("Flask" in reason for reason in request.validation_reasons)

    # Refusal-style answer, WITH the retrieved sources — never the code.
    assert "couldn't validate" in request.answer
    assert "[1] en/docs/tutorial/first-steps.md (tutorial) → First Steps" in request.footer
    assert request.footer.startswith("Sources:")
    # The failed candidate stays attached for the debug layer…
    assert request.has_code
    # …but is not the returned answer.
    assert "from flask import Flask" not in request.answer


def test_zero_turns_means_single_attempt_then_refuse():
    generator, request = _run(responses=[_BAD_IMPORT_CODE], turns=0)
    assert len(generator.prompts) == 1
    assert request.validation_failed
    assert request.footer.startswith("Sources:")


def test_model_refusal_breaks_loop_without_reformulation():
    generator, request = _run(responses=[_REFUSAL])
    assert len(generator.prompts) == 1  # loop breaks — no reformulation
    assert request.refused
    assert not request.validation_failed
    assert not request.has_code
    assert request.verdict is None
    assert request.footer == ""