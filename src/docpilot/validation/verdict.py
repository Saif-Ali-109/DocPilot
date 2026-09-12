"""Validation verdict types (PLAN §7.2 T1).

A verdict is a frozen record: did validation pass, and what did each check
conclude.  ``reasons`` collects the human-readable failures for caller-side
refusal/reformulation (T4).

No LLM, no network — pure data types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class CheckStatus(str, Enum):
    """Per-check outcome. ``SKIP`` means the check could not be evaluated
    (e.g. no retrieved evidence to ground against) — never a silent pass."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class ValidationCheck:
    """One structural/static check's outcome."""

    name: str
    status: CheckStatus
    detail: str = ""


@dataclass(frozen=True)
class ValidationVerdict:
    """Container for all checks run against one code output.

    Attributes:
        passed: ``True`` iff no check failed (``SKIP`` does not fail — it is
            reported so the caller knows what was *not* proven).
        checks: Every check in evaluation order (parse → imports → symbols).
    """

    passed: bool
    checks: tuple[ValidationCheck, ...] = ()

    @property
    def reasons(self) -> list[str]:
        """Human-readable failure reasons (``FAIL`` checks only)."""
        return [c.detail for c in self.checks if c.status is CheckStatus.FAIL]

    def summary(self) -> str:
        """One-line summary, e.g. ``PASS (3 checks)`` / ``FAIL: <first reason>``."""
        if self.passed:
            return f"PASS ({len(self.checks)} checks)"
        first = self.reasons[0] if self.reasons else "unknown failure"
        return f"FAIL: {first}"


def combine_verdicts(verdicts: Sequence[ValidationVerdict]) -> ValidationVerdict:
    """Merge per-block validation verdicts into one overall verdict.

    Each check is prefixed with its block index (``block[0].parse`` …) so the
    combined verdict stays inspectable; the overall verdict passes only when
    every block passes.  Used by the T3/T4 code route to judge a multi-block
    code output as a whole.
    """
    checks: list[ValidationCheck] = []
    for index, verdict in enumerate(verdicts):
        for check in verdict.checks:
            checks.append(
                ValidationCheck(
                    f"block[{index}].{check.name}", check.status, check.detail
                )
            )
    return ValidationVerdict(
        passed=all(c.status is not CheckStatus.FAIL for c in checks),
        checks=tuple(checks),
    )