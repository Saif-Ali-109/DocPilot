"""Hermetic tests for Phase 6 T2: the code-generation pipeline (PLAN §7.2 T2).

No LLM, no database — fakes drive ``ask_code`` end-to-end through
retrieve → prompt → generate → cite, and the whole module is tested against
the same deterministic paths the live pipeline will use.
"""

from __future__ import annotations

from docpilot.codegen import CODE_PROMPT, ask_code
from docpilot.core.models import Chunk, RetrieverResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRetriever:
    def __init__(self, results: list[RetrieverResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, question: str, *, top_k: int, language: str | None):
        self.calls.append((question, top_k, language))
        return self.results


class FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _result(content: str, file: str, heading: str, cid: str) -> RetrieverResult:
    return RetrieverResult(
        chunk=Chunk(
            id=cid, content=content, heading_path=heading, source_file=file
        ),
        score=0.9,
    )


_TUTORIAL = _result(
    "from fastapi import FastAPI\n\napp = FastAPI()\n"
    '\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}',
    "en/docs/tutorial/first-steps.md",
    "First Steps",
    "c1",
)
_UVICORN = _result(
    'import uvicorn\n\nif __name__ == "__main__":\n'
    '    uvicorn.run(app, host="0.0.0.0", port=8000)',
    "en/docs/tutorial/running.md",
    "Running the server",
    "c2",
)

_CODE_RESPONSE = """\
Here is a minimal FastAPI app from the tutorial [1]:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

Serve it with uvicorn [2]:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
"""

# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_valid_request_end_to_end():
    retriever = FakeRetriever([_TUTORIAL, _UVICORN])
    generator = FakeGenerator(_CODE_RESPONSE)
    request = ask_code(
        "Write a minimal FastAPI app.",
        retriever=retriever,
        generator=generator,
    )

    assert request.has_code
    assert not request.refused
    assert len(request.code_blocks) == 2
    assert request.code_blocks[0].startswith("from fastapi import FastAPI")
    assert request.code_blocks[1].startswith("import uvicorn")

    # Citations: [1] and [2] markers resolve to the real doc pages.
    assert "[1]" in request.answer and "[2]" in request.answer
    assert request.footer.startswith("Sources:")
    assert "[1] en/docs/tutorial/first-steps.md → First Steps" in request.footer
    assert "[2] en/docs/tutorial/running.md → Running the server" in request.footer

    # Source refs carry the derived kind (tutorial).
    assert [s.kind for s in request.sources] == ["tutorial", "tutorial"]

    # Observability: default prompt template with context/question inline.
    assert request.raw_prompt.startswith("You are DocPilot's code generator")
    assert "CONTEXT:" in request.raw_prompt
    assert "from fastapi import FastAPI" in request.raw_prompt
    assert "REQUEST:" in request.raw_prompt
    assert "Write a minimal FastAPI app." in request.raw_prompt

    assert request.latency_ms >= 0.0
    assert retriever.calls[0][0] == "Write a minimal FastAPI app."
    assert generator.prompts[0] == request.raw_prompt


def test_citation_engine_and_footer_are_optional_paths():
    # No [N] markers → no fabricated footer (§7.5: never invent citations).
    retriever = FakeRetriever([_TUTORIAL])
    generator = FakeGenerator("```python\nx = 1\n```")
    request = ask_code("write code", retriever=retriever, generator=generator)
    assert request.code_blocks == ["x = 1"]
    assert request.footer == ""
    assert request.answer == "```python\nx = 1\n```"


# ---------------------------------------------------------------------------
# Refusal / empty-evidence path
# ---------------------------------------------------------------------------


def test_empty_retrieval_refuses_with_spec_sentence():
    retriever = FakeRetriever([])
    refusal = (
        "I don't know — the available documentation does not cover this question."
    )
    generator = FakeGenerator(refusal)
    request = ask_code("generate a widget", retriever=retriever, generator=generator)

    assert request.refused
    assert not request.has_code
    assert request.code_blocks == []
    assert request.sources == []
    assert request.footer == ""
    assert request.raw_prompt == CODE_PROMPT.format(
        context="[no context retrieved — the documentation may not cover this question.]",
        sources="",
        question="generate a widget",
    )


# ---------------------------------------------------------------------------
# Interface passthrough
# ---------------------------------------------------------------------------


def test_top_k_and_language_are_resolved_and_passed_through():
    retriever = FakeRetriever([_TUTORIAL])
    generator = FakeGenerator("```python\napp = FastAPI()\n```")

    ask_code(
        "fastapi",
        retriever=retriever,
        generator=generator,
        top_k=3,
        language="any",
    )
    assert retriever.calls[0] == ("fastapi", 3, None)  # "any" → no filter

    # Default top_k comes from config; default language filter applies.
    from docpilot import config

    ask_code("fastapi", retriever=retriever, generator=generator)
    query, top_k, language = retriever.calls[1]
    assert top_k == config.RETRIEVAL_TOP_K
    assert language == config.RETRIEVAL_LANGUAGE or language == "en"


def test_custom_prompt_template():
    retriever = FakeRetriever([_TUTORIAL])
    generator = FakeGenerator("```python\napp = FastAPI()\n```")
    request = ask_code(
        "fastapi",
        retriever=retriever,
        generator=generator,
        prompt_template="CUSTOM {context} | {sources} | {question}",
    )
    assert request.raw_prompt.startswith("CUSTOM ")
    assert "from fastapi import FastAPI" in request.raw_prompt
    assert request.raw_prompt.endswith("| fastapi")


# ---------------------------------------------------------------------------
# Verdict field is T4's hook (role check only — no validation in T2)
# ---------------------------------------------------------------------------


def test_verdict_placeholder_is_none_until_t3_t4_wiring():
    retriever = FakeRetriever([_TUTORIAL])
    generator = FakeGenerator("```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```")
    request = ask_code("fastapi", retriever=retriever, generator=generator)
    assert request.verdict is None
    assert request.has_code