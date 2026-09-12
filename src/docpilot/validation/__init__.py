"""Code validation for Phase 6 (SPEC §8, PLAN §7).

T1 scope (PLAN §7.2): ``CodeValidator`` interface + ``RetrieveThenValidate`` —
given a code output plus retrieved API/schema/examples, return a pass/fail
verdict with reasons.  Structural/static checks are deterministic and hermetic
(no LLM); the LLM judge is reserved for semantic fit in the later T4 slice.
"""

from docpilot.validation.validator import (
    CodeValidator,
    RetrieveThenValidate,
    extract_code_blocks,
    harvest_api_surface,
)
from docpilot.validation.verdict import (
    CheckStatus,
    ValidationCheck,
    ValidationVerdict,
)

__all__ = [
    "CodeValidator",
    "RetrieveThenValidate",
    "ValidationCheck",
    "ValidationVerdict",
    "CheckStatus",
    "extract_code_blocks",
    "harvest_api_surface",
]