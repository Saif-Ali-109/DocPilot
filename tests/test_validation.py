"""Hermetic tests for Phase 6 T1 (CodeValidator) + T7 fixture corpus (PLAN §7.2).

No LLM, no network — every verdict is deterministic over the committed
fixtures in ``tests/fixtures/codegen`` (retrieved API spec ↔ valid/invalid
sample code pairs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docpilot.validation import (
    CheckStatus,
    RetrieveThenValidate,
    extract_code_blocks,
    harvest_api_surface,
)

FIXTURES = Path(__file__).parent / "fixtures" / "codegen"
VALIDATOR = RetrieveThenValidate()


def _read(*parts: str) -> str:
    return (FIXTURES / Path(*parts)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Evidence → API surface
# ---------------------------------------------------------------------------


def test_extract_code_blocks_basic():
    text = "before\n```\nfoo()\n```\nafter\n```\nbar()\n```"
    blocks = extract_code_blocks(text)
    assert blocks == ["foo()", "bar()"]


def test_extract_code_blocks_lang_tag_and_tilde():
    text = "```python\nx = 1\n```\n~~~\ny = 2\n~~~"
    blocks = extract_code_blocks(text)
    assert blocks == ["x = 1", "y = 2"]


def test_extract_code_blocks_unclosed_partial():
    text = "```python\npartial = True\n"
    blocks = extract_code_blocks(text)
    assert blocks == ["partial = True"]


def test_harvest_surface_from_fastapi_spec():
    surface = harvest_api_surface([_read("fastapi_basic", "api_spec.md")])
    assert {"fastapi", "FastAPI", "uvicorn", "app", "read_root"} <= surface


def test_harvest_surface_ignores_prose():
    surface = harvest_api_surface(["The `FastAPI` class creates an app; no code here."])
    assert surface == frozenset()


# ---------------------------------------------------------------------------
# Verdicts over the fixture corpus
# ---------------------------------------------------------------------------


def test_fastapi_valid_passes():
    code = _read("fastapi_basic", "valid_example.py")
    evidence = [_read("fastapi_basic", "api_spec.md")]
    verdict = VALIDATOR.validate(code, evidence)
    assert verdict.passed
    assert all(c.status is CheckStatus.PASS for c in verdict.checks)
    assert verdict.reasons == []


def test_pydantic_valid_passes():
    verdict = VALIDATOR.validate(
        _read("pydantic_schema", "valid_example.py"),
        [_read("pydantic_schema", "api_spec.md")],
    )
    assert verdict.passed


def test_unknown_import_fails():
    verdict = VALIDATOR.validate(
        _read("fastapi_basic", "invalid_unknown_import.py"),
        [_read("fastapi_basic", "api_spec.md")],
    )
    assert not verdict.passed
    assert "Flask" in verdict.reasons[0]
    assert "import" in verdict.reasons[0]


def test_undefined_symbol_fails():
    verdict = VALIDATOR.validate(
        _read("fastapi_basic", "invalid_undefined_symbol.py"),
        [_read("fastapi_basic", "api_spec.md")],
    )
    assert not verdict.passed
    assert "ThrottleMiddleware" in verdict.reasons[0]


def test_syntax_error_fails():
    verdict = VALIDATOR.validate(
        _read("fastapi_basic", "invalid_syntax.py"),
        [_read("fastapi_basic", "api_spec.md")],
    )
    assert not verdict.passed
    assert "syntax" in verdict.reasons[0]


def test_pydantic_invalid_fails_on_undeclared_symbol():
    verdict = VALIDATOR.validate(
        _read("pydantic_schema", "invalid_example.py"),
        [_read("pydantic_schema", "api_spec.md")],
    )
    assert not verdict.passed
    assert "field_validator" in verdict.reasons[0]


def test_star_import_without_evidence_fails():
    code = "from thrift_py import *\nprint(Thing())\n"
    verdict = VALIDATOR.validate(code, ["```python\nfrom fastapi import FastAPI\n```"])
    assert not verdict.passed
    assert "thrift_py" in verdict.reasons[0]


# ---------------------------------------------------------------------------
# Edge semantics — never a fabricated pass; never a spurious fail
# ---------------------------------------------------------------------------


def test_empty_evidence_skips_grounding_but_parses():
    verdict = VALIDATOR.validate("def f(x: int) -> int:\n    return x + 1\n", [])
    assert verdict.passed  # parseable + nothing to ground against
    statuses = {c.name: c.status for c in verdict.checks}
    assert statuses["imports_evidence"] is CheckStatus.SKIP
    assert statuses["symbols_evidence"] is CheckStatus.SKIP
    assert verdict.reasons == []


def test_stdlib_import_and_builtins_pass_without_evidence_mention():
    code = "import os\n\ndef size(path: str) -> int:\n    return len(os.listdir(path))\n"
    # Evidence contains a code block, but nothing about os / the standard library.
    verdict = VALIDATOR.validate(code, ["```python\nfrom fastapi import FastAPI\n```"])
    assert verdict.passed  # os is stdlib; len is a builtin


def test_dunder_guard_is_not_a_grounding_violation():
    code = (
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
        "if __name__ == \"__main__\":\n    import uvicorn\n    uvicorn.run(app)\n"
    )
    verdict = VALIDATOR.validate(code, [_read("fastapi_basic", "api_spec.md")])
    assert verdict.passed


def test_verdict_summary_forms():
    ok = VALIDATOR.validate(
        "from fastapi import FastAPI\napp = FastAPI()\n",
        ["```python\nfrom fastapi import FastAPI\n```"],
    )
    assert ok.summary().startswith("PASS")
    bad = VALIDATOR.validate(
        "from flask import Flask\n",
        ["```python\nfrom fastapi import FastAPI\n```"],
    )
    assert bad.summary().startswith("FAIL:")
    assert len(bad.reasons) >= 1