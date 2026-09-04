"""Fast, hermetic end-to-end tests for the ingest/ask pipelines (SPEC.md §3.17).

No network, no PostgreSQL, no live Groq API, no embedding model. The real
``MarkdownParser`` and ``MarkdownChunker`` run against inline fixture
documents; embeddings and the vector store are deterministic fakes. The real
pgvector/DB paths are covered by Agent C/D unit tests — these tests cover the
*wiring* (:func:`docpilot.pipeline_ingest.ingest_corpus` and
:func:`docpilot.pipeline_ask.ask`).

The fakes defined here are also imported by ``tests/test_cli.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from docpilot.citations.engine import StandardCitationEngine
from docpilot.core.models import Chunk, Document, RetrieverResult
from docpilot.embeddings.provider import EmbeddingProvider
from docpilot.generation.generator import Generator
from docpilot.ingestion.chunker import MarkdownChunker
from docpilot.ingestion.loader import DocumentLoader
from docpilot.ingestion.parser import MarkdownParser
from docpilot.pipeline_ask import _NO_CONTEXT_NOTE, ask
from docpilot.pipeline_ingest import ingest_corpus
from docpilot.retrieval.retriever import SimpleRetriever
from docpilot.retrieval.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Fixture corpus (inline — no filesystem dependency)
# ---------------------------------------------------------------------------

FIXTURE_DOCS: list[Document] = [
    Document(
        file_path="guide/installation.md",
        content=(
            "# Installation\n"
            "\n"
            "FastAPI is a modern web framework for building Python APIs.\n"
            "\n"
            "To install FastAPI run:\n"
            "\n"
            "```console\n"
            '$ pip install "fastapi[standard]"\n'
            "```\n"
            "\n"
            "This installs FastAPI and a few default optional dependencies.\n"
            "\n"
            "| Method | Command |\n"
            "|--------|---------|\n"
            "| pip    | `pip install fastapi` |\n"
            "| uv     | `uv add fastapi` |\n"
            "\n"
            "## Verification\n"
            "\n"
            "Run `fastapi --version` to confirm the installation.\n"
        ),
    ),
    Document(
        file_path="tutorial/first-steps.md",
        content=(
            "# First Steps\n"
            "\n"
            "Create a file named `main.py`:\n"
            "\n"
            "```python\n"
            "from fastapi import FastAPI\n"
            "\n"
            "app = FastAPI()\n"
            "\n"
            "\n"
            '@app.get("/")\n'
            "def read_root():\n"
            '    return {"Hello": "World"}\n'
            "```\n"
        ),
    ),
    Document(
        file_path="advanced/dependencies.md",
        content=(
            "# Dependencies\n"
            "\n"
            "FastAPI has a powerful dependency injection system.\n"
            "\n"
            "## Declaring Dependencies\n"
            "\n"
            "Add parameters to your path operation functions.\n"
            "\n"
            "## Sub-dependencies\n"
            "\n"
            "Dependencies can declare their own dependencies.\n"
        ),
    ),
]

UNKNOWN_ANSWER = "I don't know — the available documentation does not cover this question."


def _chunk_all(chunker: MarkdownChunker) -> list[Chunk]:
    """Chunk every fixture document (used by several tests)."""
    chunks: list[Chunk] = []
    for doc in FIXTURE_DOCS:
        chunks.extend(chunker.chunk(doc))
    return chunks


# ---------------------------------------------------------------------------
# Shared fakes (also used by tests/test_cli.py)
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings: unit vectors seeded by the text's SHA-256.

    Identical input → identical vector (SPEC's "deterministic for identical
    input" property), dimension 384, L2-normalized so cosine search behaves
    like the real BGE provider.
    """

    _dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError(
                f"embed() expects a list[str], got {type(texts).__name__}"
            )
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.random(self._dimension).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            vectors.append(vec / norm if norm > 0 else vec)
        return np.stack(vectors)


class InMemoryVectorStore(VectorStore):
    """In-memory VectorStore mirroring PgVectorStore's observable behaviour.

    API-compatible with the ``VectorStore`` interface; ``add`` caches the
    deterministic ``chunk_id`` inside ``metadata`` (like the real store), and
    ``search`` recovers it when building results. Search is exact cosine
    similarity, matching Phase 1's brute-force pgvector search.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []

    @property
    def chunks(self) -> list[Chunk]:
        """Read-only view of stored chunks (test inspection)."""
        return list(self._chunks)

    def add(self, chunks: list[Chunk], embeddings) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length"
            )
        for chunk, emb in zip(chunks, embeddings):
            meta = dict(chunk.metadata)
            meta["chunk_id"] = chunk.id
            self._chunks.append(
                Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    heading_path=chunk.heading_path,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    metadata=meta,
                )
            )
            self._vectors.append(np.asarray(emb, dtype=np.float32).reshape(-1))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrieverResult]:
        if not self._chunks:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(query))
        scores: list[float] = []
        for vec in self._vectors:
            v_norm = float(np.linalg.norm(vec))
            if q_norm == 0.0 or v_norm == 0.0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(query, vec) / (q_norm * v_norm)))
        order = sorted(
            range(len(self._chunks)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: list[RetrieverResult] = []
        for i in order:
            chunk = self._chunks[i]
            meta = dict(chunk.metadata)
            chunk_id = meta.pop("chunk_id", chunk.id)
            results.append(
                RetrieverResult(
                    chunk=Chunk(
                        id=chunk_id,
                        content=chunk.content,
                        heading_path=chunk.heading_path,
                        source_file=chunk.source_file,
                        chunk_index=chunk.chunk_index,
                        metadata=meta,
                    ),
                    score=scores[i],
                )
            )
        return results

    def delete_by_source(self, source_files: list[str]) -> int:
        excluded = set(source_files)
        before = len(self._chunks)
        keep = [
            (c, v)
            for c, v in zip(self._chunks, self._vectors)
            if c.source_file not in excluded
        ]
        self._chunks = [c for c, _ in keep]
        self._vectors = [v for _, v in keep]
        return before - len(self._chunks)

    def count(self) -> int:
        return len(self._chunks)


class FakeLoader(DocumentLoader):
    """Loader that returns a fixed document list (no filesystem access)."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def load(self) -> list[Document]:
        return list(self._documents)


class FakeGenerator(Generator):
    """Canned-response generator that records what it was given."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_context: str | None = None
        self.last_sources: str | None = None
        self.last_question: str | None = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response

    def generate_answer(
        self, context_text: str, sources_text: str, question: str
    ) -> str:
        from docpilot.generation.prompts import SYSTEM_PROMPT

        self.calls += 1
        self.last_context = context_text
        self.last_sources = sources_text
        self.last_question = question
        self.last_prompt = SYSTEM_PROMPT.format(
            context=context_text, sources=sources_text, question=question
        )
        return self.response


def build_ask_wiring() -> tuple[SimpleRetriever, FakeGenerator, InMemoryVectorStore]:
    """Load the fixture corpus into an in-memory store over real chunking.

    Returns ``(retriever, generator, store)`` with the generator wired to a
    canned cited answer; the store is seeded with all fixture chunks.
    """
    chunker = MarkdownChunker()
    all_chunks = _chunk_all(chunker)
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.add(all_chunks, embedder.embed([c.content for c in all_chunks]))
    retriever = SimpleRetriever(embedder, store)
    generator = FakeGenerator("To install FastAPI, run `pip install fastapi`. [1]")
    return retriever, generator, store


# ---------------------------------------------------------------------------
# Test A — ingest e2e + idempotency
# ---------------------------------------------------------------------------


def test_ingest_e2e_with_fakes() -> None:
    chunker = MarkdownChunker()
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    loader = FakeLoader(FIXTURE_DOCS)

    expected = sum(len(chunker.chunk(doc)) for doc in FIXTURE_DOCS)
    assert expected > 0

    stats = ingest_corpus(
        loader=loader,
        parser=MarkdownParser(),
        chunker=chunker,
        embedding_provider=embedder,
        vector_store=store,
    )

    # Wired counts match the chunker's own output.
    assert stats.files_processed == len(FIXTURE_DOCS)
    assert stats.total_chunks == expected
    assert store.count() == expected
    assert sum(stats.chunks_per_source.values()) == expected
    for src in FIXTURE_DOCS:
        assert src.file_path in stats.chunks_per_source

    # Stage timings cover every pipeline stage.
    assert set(stats.stage_timings) == {"load", "chunk", "delete", "embed", "store"}
    assert all(v >= 0.0 for v in stats.stage_timings.values())

    # Chunk-size stats (words as approximate tokens).
    assert stats.min_chunk_size >= 1
    assert stats.avg_chunk_size <= stats.max_chunk_size

    # Idempotency: re-running must NOT grow the store (delete-before-insert).
    stats2 = ingest_corpus(
        loader=FakeLoader(FIXTURE_DOCS),
        parser=MarkdownParser(),
        chunker=chunker,
        embedding_provider=embedder,
        vector_store=store,
    )
    assert stats2.total_chunks == expected
    assert store.count() == expected


def test_ingest_empty_corpus_is_harmless() -> None:
    store = InMemoryVectorStore()
    stats = ingest_corpus(
        loader=FakeLoader([]),
        chunker=MarkdownChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )
    assert stats.files_processed == 0
    assert stats.total_chunks == 0
    assert store.count() == 0


# ---------------------------------------------------------------------------
# Test B — ask e2e: citations + footer match the retrieved chunk
# ---------------------------------------------------------------------------


def test_ask_e2e_citations_match_retrieved_chunk() -> None:
    retriever, generator, _store = build_ask_wiring()

    result = ask(
        "How do I install FastAPI?",
        retriever=retriever,
        generator=generator,
        citation_engine=StandardCitationEngine(),
    )

    assert result.results, "expected at least one retrieved chunk"
    top = result.results[0]

    # The cited answer keeps its [1] marker.
    assert "[1]" in result.answer
    # Footer exists and its [1] line points at the RIGHT chunk.
    assert result.footer.startswith("Sources:")
    assert result.sources and result.sources[0].ref == 1
    assert result.sources[0].file == top.chunk.source_file
    assert result.sources[0].heading == top.chunk.heading_path
    assert result.sources[0].heading, "fixture chunks carry heading paths"
    assert f"[1] {top.chunk.source_file} → {top.chunk.heading_path}" in result.footer

    # Observability fields flow through.
    assert result.raw_response == generator.response
    assert result.raw_prompt.startswith("You are DocPilot")
    assert "CONTEXT:" in result.raw_prompt and top.chunk.source_file in result.raw_prompt
    assert result.latency_ms >= 0.0

    # Display string = answer + blank line + footer.
    assert result.display == f"{result.answer}\n\n{result.footer}"


# ---------------------------------------------------------------------------
# Test C — I-don't-know path (no markers → no fabricated footer)
# ---------------------------------------------------------------------------


def test_ask_unknown_answer_has_no_fabricated_citations() -> None:
    retriever, _generator, _store = build_ask_wiring()
    generator = FakeGenerator(UNKNOWN_ANSWER)

    result = ask(
        "What is the meaning of life?",
        retriever=retriever,
        generator=generator,
        citation_engine=StandardCitationEngine(),
    )

    assert result.answer == UNKNOWN_ANSWER
    assert result.footer == ""
    assert result.display == result.answer


# ---------------------------------------------------------------------------
# Test D — empty retrieval must not crash and must not fabricate sources
# ---------------------------------------------------------------------------


def test_ask_empty_retrieval_does_not_crash() -> None:
    embedder = FakeEmbeddingProvider()
    empty_store = InMemoryVectorStore()
    retriever = SimpleRetriever(embedder, empty_store)
    generator = FakeGenerator(UNKNOWN_ANSWER)

    result = ask(
        "Does FastAPI support websockets?",
        retriever=retriever,
        generator=generator,
        citation_engine=StandardCitationEngine(),
    )

    assert result.results == []
    assert result.sources == []
    assert result.footer == ""
    assert result.answer == UNKNOWN_ANSWER
    assert result.raw_response == UNKNOWN_ANSWER
    assert "no context retrieved" in result.raw_prompt
    # The placeholder note is the context the generator saw.
    assert _NO_CONTEXT_NOTE in result.raw_prompt


# ---------------------------------------------------------------------------
# Test E — code-block (and table) integrity through the pipeline
# ---------------------------------------------------------------------------


def test_code_and_table_integrity_through_pipeline() -> None:
    chunker = MarkdownChunker()
    store = InMemoryVectorStore()
    ingest_corpus(
        loader=FakeLoader(FIXTURE_DOCS),
        parser=MarkdownParser(),
        chunker=chunker,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    stored = store.chunks
    assert stored, "expected chunks to be stored"

    # Every stored chunk flagged as containing code keeps its fences intact.
    code_chunks = [c for c in stored if c.metadata.get("code_block")]
    assert code_chunks, "expected at least one code-bearing chunk"
    for chunk in code_chunks:
        assert chunk.content.count("```") >= 2
        assert chunk.content.count("```") % 2 == 0, "unbalanced fence"

    # The FastAPI sample app code block survives verbatim.
    steps_chunk = next(
        c for c in stored if c.source_file == "tutorial/first-steps.md"
    )
    assert steps_chunk.metadata.get("code_block")
    assert steps_chunk.content.startswith("Create a file named `main.py`:")
    assert "from fastapi import FastAPI" in steps_chunk.content
    assert '@app.get("/")' in steps_chunk.content
    assert steps_chunk.content.rstrip().endswith("```"), "closing fence intact"

    # The Markdown table survives as consecutive rows in one chunk.
    install_chunk = next(
        c for c in stored if c.source_file == "guide/installation.md"
    )
    table_block = "\n".join(
        [
            "| Method | Command |",
            "|--------|---------|",
            "| pip    | `pip install fastapi` |",
            "| uv     | `uv add fastapi` |",
        ]
    )
    assert table_block in install_chunk.content
    assert install_chunk.metadata.get("table")


def test_retrieval_roundtrip_uses_stored_vector_shape() -> None:
    """Embed → store → search returns 384-dim-compatible results (no LLM)."""
    retriever, generator, store = build_ask_wiring()
    assert store.count() >= 3
    result = ask(
        "How do I install FastAPI?",
        retriever=retriever,
        generator=generator,
        citation_engine=StandardCitationEngine(),
    )
    assert result.results
    assert all(-1.0 <= r.score <= 1.0 + 1e-6 for r in result.results)