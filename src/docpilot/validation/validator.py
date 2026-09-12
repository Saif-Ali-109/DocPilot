"""``CodeValidator`` interface + structural-first implementation (PLAN §7.2 T1).

Phase 6 scope (SPEC §8): documentation retrieval → generate code → *validate
against retrieved API/schema/examples* → return code + sources.  T1 establishes
the validator half with deterministic, hermetic checks first:

1. **parse** — the candidate must be valid Python (``ast.parse``).
2. **imports_evidence** — every non-stdlib import the candidate makes must be
   observable in the retrieved evidence (its API-surface identifiers).
3. **symbols_evidence** — every external (unbound, non-builtin) symbol the
   candidate references must also be observable in the retrieved evidence.

Evidence is reduced to an *API surface*: the identifiers appearing inside fenced
code blocks of the retrieved doc chunks (specs/examples), harvested by
:func:`harvest_api_surface`.  Checks are skipped — never silently passed — when
there is no evidence to ground against, and the LLM judge for semantic fit is
deliberately out of scope here (T4).

No LLM, no network; fully hermetic on the committed fixture corpus
(``tests/fixtures/codegen``, T7).
"""

from __future__ import annotations

import ast
import builtins
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence

from docpilot.validation.verdict import (
    CheckStatus,
    ValidationCheck,
    ValidationVerdict,
)

_BUILTINS: frozenset[str] = frozenset(dir(builtins))
_STDLIB: frozenset[str] = frozenset(sys.stdlib_module_names)
_IGNORED_SYMBOLS: frozenset[str] = frozenset(
    {
        "_",
        # ``if __name__ == "__main__":`` — intrinsic, never evidence-able.
        "__name__",
    }
)
_FENCE_OPEN_RE = re.compile(r"^(```|~~~)([A-Za-z0-9_+-]*)\s*$")


# ---------------------------------------------------------------------------
# Evidence → API-surface reduction (T2/T4 reuse these)
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[str]:
    """Return the contents of every fenced code block in *text*.

    Handles both ````` ``` ```` and ``~~~`` fences, with or without a language
    tag on the opener; content is the raw lines between opener and closer.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    fence_char = ""
    for line in text.splitlines():
        if current is None:
            m = _FENCE_OPEN_RE.match(line)
            if m:
                fence_char = m.group(1)
                current = []
            continue
        if line.startswith(fence_char) and len(line) >= 3:
            blocks.append("\n".join(current))
            current = None
            fence_char = ""
        else:
            current.append(line)
    if current is not None:
        # Unclosed fence — still surface the partial block (doc snippets are
        # often truncated); the caller can decide how much to trust it.
        blocks.append("\n".join(current))
    return blocks


def _surface_identifiers(code: str) -> set[str]:
    """Identifiers in one code chunk: AST names + imports, regex fallback.

    Rich by design — the surface should over-approximate the documented API so
    valid candidates are never failed on a harvest miss; the candidate side of
    the check (``_external_names``) is the precise one.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def harvest_api_surface(evidence: Sequence[str]) -> frozenset[str]:
    """API surface = identifiers inside fenced code blocks across *evidence*."""
    surface: set[str] = set()
    for chunk in evidence:
        for block in extract_code_blocks(chunk):
            surface |= _surface_identifiers(block)
    return frozenset(surface)


# ---------------------------------------------------------------------------
# Candidate-code analysis (the precise side)
# ---------------------------------------------------------------------------


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """Every name the candidate binds locally (defs, args, imports, targets…)."""

    def _target_names(node: ast.AST) -> set[str]:
        return {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name)
        }

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    bound.add(node.module or "*")
                else:
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bound |= _target_names(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bound |= _target_names(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bound |= _target_names(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bound |= _target_names(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            bound |= _target_names(node.target)
        elif isinstance(node, ast.comprehension):
            bound |= _target_names(node.target)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
    return bound


def _external_names(tree: ast.AST, bound: set[str]) -> set[str]:
    """Name *loads* that are neither locally bound nor builtins/ignored."""
    loads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.add(node.id)
    return (
        loads
        - bound
        - _BUILTINS
        - _IGNORED_SYMBOLS
        - {n for n in loads if n.startswith("__") and n.endswith("__")}
    )


def _candidate_imports(tree: ast.AST) -> list[str]:
    """Top-level names the candidate imports (module or symbol)."""
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                if alias.name == "*":
                    if node.module:
                        imports.append(node.module.split(".")[0])
                else:
                    imports.append(alias.asname or alias.name)
    return imports


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class CodeValidator(ABC):
    """Interface for judging a generated code output against retrieved evidence."""

    @abstractmethod
    def validate(self, code: str, retrieved: Sequence[str]) -> ValidationVerdict:
        """Return a pass/fail verdict for *code* given the retrieved doc chunks.

        Args:
            code: The generated candidate code.
            retrieved: Evidence chunks (retrieved docs/specs/examples) the code
                must be grounded in.

        Returns:
            A :class:`ValidationVerdict`; ``passed=False`` must carry at least
            one failure reason — never a bare fail without an explanation.
        """
        ...


class RetrieveThenValidate(CodeValidator):
    """Structural-first validator: parse → imports grounded → symbols grounded.

    Deterministic by construction: same (code, evidence) always yields the same
    verdict, so it is hermetic-testable without an LLM (PLAN §7.5).  Semantic
    fit ("is this the right approach?") is deliberately not judged here — that
    is the T4 LLM-judge slice.
    """

    def validate(self, code: str, retrieved: Sequence[str]) -> ValidationVerdict:
        checks: list[ValidationCheck] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            detail = f"syntax error: {exc.msg} at line {exc.lineno or '?'}"
            return ValidationVerdict(
                passed=False,
                checks=(ValidationCheck("parse", CheckStatus.FAIL, detail),),
            )
        checks.append(ValidationCheck("parse", CheckStatus.PASS, "parses cleanly"))

        surface = harvest_api_surface(retrieved)
        bound = _collect_bound_names(tree)

        checks.extend(self._check_imports(tree, surface, bool(retrieved)))
        checks.extend(self._check_symbols(tree, bound, surface, bool(retrieved)))

        return ValidationVerdict(
            passed=all(c.status is not CheckStatus.FAIL for c in checks),
            checks=tuple(checks),
        )

    @staticmethod
    def _check_imports(
        tree: ast.AST, surface: frozenset[str], has_evidence: bool
    ) -> list[ValidationCheck]:
        if not has_evidence:
            return [
                ValidationCheck(
                    "imports_evidence",
                    CheckStatus.SKIP,
                    "no retrieved evidence — import grounding skipped",
                )
            ]
        missing = [
            name
            for name in _candidate_imports(tree)
            if name not in surface and name not in _STDLIB
        ]
        if missing:
            return [
                ValidationCheck(
                    "imports_evidence",
                    CheckStatus.FAIL,
                    "import(s) not grounded in retrieved evidence: "
                    + ", ".join(sorted(missing)),
                )
            ]
        return [
            ValidationCheck(
                "imports_evidence",
                CheckStatus.PASS,
                "all imports grounded in evidence or stdlib",
            )
        ]

    @staticmethod
    def _check_symbols(
        tree: ast.AST,
        bound: set[str],
        surface: frozenset[str],
        has_evidence: bool,
    ) -> list[ValidationCheck]:
        if not has_evidence:
            return [
                ValidationCheck(
                    "symbols_evidence",
                    CheckStatus.SKIP,
                    "no retrieved evidence — symbol grounding skipped",
                )
            ]
        missing = sorted(_external_names(tree, bound) - surface)
        if missing:
            return [
                ValidationCheck(
                    "symbols_evidence",
                    CheckStatus.FAIL,
                    "symbol(s) not found in retrieved evidence: "
                    + ", ".join(missing),
                )
            ]
        return [
            ValidationCheck(
                "symbols_evidence",
                CheckStatus.PASS,
                "all external symbols present in retrieved evidence",
            )
        ]