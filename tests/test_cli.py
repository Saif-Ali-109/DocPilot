"""CLI tests: stdout/stderr separation, --json / --debug behaviour, exit codes.

The pipelines are always exercised with **injected fakes** (SPEC.md §3.17,
PLAN.md §2.3 AGENT E guards) — no PostgreSQL, no Groq API, no embedding model.
Fakes are shared from ``test_pipeline_e2e``.

Never asserts on real .env values or secrets.
"""

from __future__ import annotations

import json

import pytest

from docpilot import cli
from docpilot.citations.engine import StandardCitationEngine
from docpilot.ingestion.chunker import MarkdownChunker
from docpilot.ingestion.parser import MarkdownParser
from docpilot.pipeline_ask import ask as pipeline_ask
from docpilot.retrieval.retriever import SimpleRetriever

from test_pipeline_e2e import (
    FIXTURE_DOCS,
    FakeEmbeddingProvider,
    FakeGenerator,
    FakeLoader,
    InMemoryVectorStore,
)

QUESTION = "How do I install FastAPI?"
CANNED_ANSWER = "To install FastAPI, run `pip install fastapi`. [1]"


def _ask_fakes() -> dict:
    """Return ask-pipeline fakes wired over the fixture corpus."""
    chunker = MarkdownChunker()
    all_chunks: list = []
    for doc in FIXTURE_DOCS:
        all_chunks.extend(chunker.chunk(doc))
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.add(all_chunks, embedder.embed([c.content for c in all_chunks]))
    retriever = SimpleRetriever(embedder, store)
    generator = FakeGenerator(CANNED_ANSWER)
    engine = StandardCitationEngine()
    return {
        "retriever": retriever,
        "generator": generator,
        "citation_engine": engine,
    }


def _ingest_fakes() -> dict:
    """Return ingest-pipeline fakes (loader fake; parser/chunker real and hermetic)."""
    return {
        "loader": FakeLoader(FIXTURE_DOCS),
        "parser": MarkdownParser(),
        "chunker": MarkdownChunker(),
        "embedding_provider": FakeEmbeddingProvider(),
        "vector_store": InMemoryVectorStore(),
    }


# ---------------------------------------------------------------------------
# ask --json
# ---------------------------------------------------------------------------


def test_ask_json_output_is_clean_and_parseable(capsys) -> None:
    code = cli.main(["ask", QUESTION, "--json"], **_ask_fakes())
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)  # raises if any log leaked to stdout
    # Phase 1 keys are preserved (Phase 2 adds ``strategy``/``trace``/
    # ``direct``/``refused`` — additive only, never replacing).
    assert {"question", "answer", "sources"} <= set(payload)

    assert payload["question"] == QUESTION
    assert "pip install" in payload["answer"]
    assert payload["sources"], "expected at least one source"
    for src in payload["sources"]:
        assert set(src) == {"ref", "file", "heading"}
        assert isinstance(src["ref"], int) and src["ref"] >= 1
        assert src["file"]

    # No prompt/raw/secret material in the JSON, ever.
    assert "prompt" not in payload
    assert "raw" not in payload
    assert "key" not in payload
    assert "GROQ" not in captured.out
    assert "POSTGRES" not in captured.out

    # Info logs still reach stderr (not stdout).
    assert "Retrieved" in captured.err


# ---------------------------------------------------------------------------
# ask plain text
# ---------------------------------------------------------------------------


def test_ask_plain_stdout_is_exactly_the_display_string(capsys) -> None:
    fakes = _ask_fakes()
    code = cli.main(["ask", QUESTION], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    # stdout must contain ONLY the answer + footer (SPEC.md §3.11).
    expected = pipeline_ask(QUESTION, **fakes)
    assert captured.out.strip() == expected.display.strip()
    assert captured.out.strip().startswith("To install FastAPI")
    assert "\n\nSources:" in captured.out
    assert "[1]" in captured.out


def test_ask_plain_unknown_answer_has_no_footer(capsys) -> None:
    fakes = _ask_fakes()
    fakes["generator"] = FakeGenerator(
        "I don't know — the available documentation does not cover this question."
    )
    code = cli.main(["ask", "What is the meaning of life?"], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == (
        "I don't know — the available documentation does not cover this question."
    )
    assert "Sources:" not in captured.out


# ---------------------------------------------------------------------------
# ask --debug
# ---------------------------------------------------------------------------


def test_ask_debug_logs_retrieval_prompt_and_response_to_stderr(capsys) -> None:
    fakes = _ask_fakes()
    code = cli.main(["ask", QUESTION, "--debug"], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    # stdout is STILL only the answer.
    expected = pipeline_ask(QUESTION, **fakes)
    assert captured.out.strip() == expected.display.strip()

    err = captured.err
    assert "Query embedding dimension: 384" in err
    assert "Retrieved chunk" in err
    assert "score=" in err
    assert "file=" in err
    assert "Full prompt sent to LLM" in err
    # The full prompt is visible for debugging (SPEC.md §3.13).
    assert "DocPilot" in err and "CONTEXT:" in err
    assert "Raw LLM response" in err
    assert CANNED_ANSWER in err
    assert "latency" in err.lower()


def test_ask_default_info_level_hides_debug_detail(capsys) -> None:
    fakes = _ask_fakes()
    code = cli.main(["ask", QUESTION], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    # Without --debug the raw prompt must NOT appear on stderr.
    assert "Full prompt sent to LLM" not in captured.err
    assert CANNED_ANSWER not in captured.err


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def test_ingest_with_injected_fakes(capsys) -> None:
    fakes = _ingest_fakes()
    code = cli.main(["ingest"], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    store: InMemoryVectorStore = fakes["vector_store"]
    assert store.count() > 0
    # Concise summary line on stdout / nothing else.
    assert captured.out.strip() == (
        f"Ingested {store.count()} chunks from {len(FIXTURE_DOCS)} files"
    )
    # INFO stage summary reaches stderr.
    assert "Ingest complete" in captured.err


def test_ingest_debug_reports_chunk_stats(capsys) -> None:
    fakes = _ingest_fakes()
    code = cli.main(["ingest", "--debug"], **fakes)
    captured = capsys.readouterr()

    assert code == 0
    err = captured.err
    assert "Chunk size stats" in err
    assert "Chunks per source file" in err
    assert all(
        src in err
        for src in ("en/docs/guide/installation.md", "en/docs/tutorial/first-steps.md")
    )
    # stdout stays a single summary line even in debug mode.
    assert captured.out.strip().startswith("Ingested ")


# ---------------------------------------------------------------------------
# exit codes / usage errors
# ---------------------------------------------------------------------------


def test_missing_subcommand_exits_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_ask_without_question_exits_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ask"])
    assert excinfo.value.code == 2


def test_unknown_subcommand_exits_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["frobnicate"])
    assert excinfo.value.code == 2


def test_runtime_error_returns_1(capsys) -> None:
    fakes = _ask_fakes()

    class OutageGenerator(FakeGenerator):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("simulated LLM outage")

        def generate_answer(
            self, context_text: str, sources_text: str, question: str
        ) -> str:
            raise RuntimeError("simulated LLM outage")

    fakes["generator"] = OutageGenerator()
    code = cli.main(["ask", QUESTION], **fakes)
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""  # no partial answer on failure
    assert "Error:" in captured.err
    assert "simulated LLM outage" in captured.err


# ---------------------------------------------------------------------------
# injection seam sanity
# ---------------------------------------------------------------------------


def test_injected_prompt_template_matches_system_prompt() -> None:
    """The pipeline's reconstructed prompt equals SYSTEM_PROMPT substitution."""
    fakes = _ask_fakes()
    result = pipeline_ask(QUESTION, **fakes)
    # Sanity-check the template source (substitution is the generator's job).
    assert result.raw_prompt.startswith("You are DocPilot")
    assert "{context}" not in result.raw_prompt
    assert "{sources}" not in result.raw_prompt
    assert "{question}" not in result.raw_prompt