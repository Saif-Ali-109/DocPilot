"""Phase 6 code generation (SPEC §8, PLAN §7).

T2 (this package): ``ask_code`` retrieves documentation with the existing
``Retriever``, generates code grounded in the retrieved API/schema/examples
via the existing ``Generator`` interface, and attaches ``CitationEngine``
sources. Validation wiring (T3/T4) lives in the agent/route layers — this
capability stays a clean, injectable pipeline.
"""

from docpilot.codegen.pipeline_ask_code import (
    VALIDATION_REFUSAL,
    CodeRequest,
    ask_code,
)
from docpilot.codegen.prompts import CODE_FIX_PROMPT, CODE_PROMPT

__all__ = [
    "CodeRequest",
    "ask_code",
    "CODE_PROMPT",
    "CODE_FIX_PROMPT",
    "VALIDATION_REFUSAL",
]