"""Command-line interface for DocPilot (SPEC.md §3.11–§3.13).

Usage::

    python -m docpilot ingest [--debug]
    python -m docpilot ask "QUESTION" [--debug] [--json] [--lang LANGUAGE]

Stream discipline:
    * **stdout** carries only program output — the answer (plain mode), the
      JSON payload (``--json``), or the ingest summary line. Never logs.
    * **stderr** carries all logging (INFO by default, DEBUG with ``--debug``).
    * Secrets/API keys are never logged, at any level.

Exit codes: ``0`` success, ``1`` runtime error (message on stderr), ``2``
argparse usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger("docpilot.cli")


class _DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time.

    Binding ``sys.stderr`` once at handler construction breaks under pytest's
    ``capsys`` fixture: the captured stream is closed when the test ends, and a
    later log record would write to a closed object. Re-resolving on every
    emit keeps logging robust across in-process CLI/pipeline calls.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)

# Injection seam (test-only): keys in ``**injected`` are forwarded to the
# matching pipeline parameters. Unrelated keys are ignored.
_INGEST_INJECTION_KEYS = (
    "loader",
    "parser",
    "chunker",
    "embedding_provider",
    "vector_store",
)
_ASK_INJECTION_KEYS = ("retriever", "generator", "citation_engine")


def _configure_logging(debug: bool) -> None:
    """Route the ``docpilot`` logger tree to stderr at the right level.

    Idempotent — repeated in-process calls (e.g. tests) never stack handlers.
    ``propagate=False`` keeps records off stdout and out of other handler
    trees; the pytest ``capsys`` capture still sees stderr unchanged.
    """
    root = logging.getLogger("docpilot")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    handler = _DynamicStderrHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docpilot",
        description="DocPilot — evidence-driven RAG over technical documentation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest the docs/ corpus into the vector store.")
    ingest_p.add_argument(
        "--debug",
        action="store_true",
        help="Log full per-source chunk statistics and stage timings.",
    )

    ask_p = sub.add_parser("ask", help="Answer a question using the ingested corpus.")
    ask_p.add_argument("question", help="The question to answer.")
    ask_p.add_argument(
        "--debug",
        action="store_true",
        help="Log embedding dimension, retrieval details, prompt and raw response.",
    )
    ask_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON object {question, answer, sources} instead of plain text.",
    )
    ask_p.add_argument(
        "--lang",
        default=None,
        help="Retrieval language filter (default: RETRIEVAL_LANGUAGE env; 'any' = no filter).",
    )

    return parser


def _run_ingest(args: argparse.Namespace, injected: dict[str, Any]) -> int:
    from docpilot.pipeline_ingest import ingest_corpus

    kwargs = {k: injected[k] for k in _INGEST_INJECTION_KEYS if k in injected}
    stats = ingest_corpus(**kwargs)

    logger.info(
        "Ingest complete: %d file(s) → %d chunk(s) in %.2fs",
        stats.files_processed,
        stats.total_chunks,
        sum(stats.stage_timings.values()),
    )
    logger.info(
        "Stage timings: %s",
        {k: f"{v:.3f}s" for k, v in stats.stage_timings.items()},
    )
    # --debug: per-source breakdown + chunk-size statistics (SPEC.md §3.13).
    logger.debug(
        "Chunk size stats — avg=%d words, min=%d, max=%d",
        round(stats.avg_chunk_size),
        stats.min_chunk_size,
        stats.max_chunk_size,
    )
    logger.debug(
        "Chunks per source file: %d file(s)",
        len(stats.chunks_per_source),
    )
    for source_file, count in sorted(stats.chunks_per_source.items()):
        logger.debug("  %s: %d chunk(s)", source_file, count)

    # stdout carries only the summary line.
    print(f"Ingested {stats.total_chunks} chunks from {stats.files_processed} files")
    return 0


def _run_ask(args: argparse.Namespace, injected: dict[str, Any]) -> int:
    from docpilot.pipeline_ask import ask

    kwargs = {k: injected[k] for k in _ASK_INJECTION_KEYS if k in injected}
    result = ask(args.question, language=args.lang, **kwargs)

    if args.json:
        payload = {
            "question": result.question,
            "answer": result.answer,
            "sources": [
                {"ref": s.ref, "file": s.file, "heading": s.heading}
                for s in result.sources
            ],
        }
        # Never include the raw prompt, raw response or any secrets in JSON.
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(result.display)
    return 0


def main(argv: list[str] | None = None, **injected: Any) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (``None`` → ``sys.argv[1:]``).
        **injected: Test-only seam. Keys matching pipeline parameters
            (``loader``, ``parser``, ``chunker``, ``embedding_provider``,
            ``vector_store`` for ``ingest``; ``retriever``, ``generator``,
            ``citation_engine`` for ``ask``) are forwarded to the pipelines.
            When absent, the real production defaults are built.

    Returns:
        Exit code: ``0`` on success, ``1`` on runtime error. Argparse raises
        ``SystemExit(2)`` on bad usage.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "debug", False))

    try:
        if args.command == "ingest":
            return _run_ingest(args, injected)
        if args.command == "ask":
            return _run_ask(args, injected)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: surface any failure.
        logger.error("Command failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0  # pragma: no cover — unreachable: subparsers are required.