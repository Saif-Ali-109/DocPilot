"""``python -m docpilot.eval`` — Phase 4 eval entry point.

First slice: judge two-prompt A/B calibration (SPEC §6.2).
"""

from __future__ import annotations

from docpilot.eval.judge_ab import main

if __name__ == "__main__":
    raise SystemExit(main())