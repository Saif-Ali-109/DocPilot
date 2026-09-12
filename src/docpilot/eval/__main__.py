"""``python -m docpilot.eval`` — Phase 4 eval entry point.

Subcommands (SPEC §6.2 slices):
  - ``judge-ab``        (default) judge two-prompt A/B calibration
  - ``tool-necessity``  tri-class tool-trigger necessity measurement
  - ``benchmark``       §6.1 classic-vs-agentic comparison over the benchmark
  - ``code-benchmark``  Phase 6 code eval — validation-pass + citations (T5)
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "tool-necessity":
        from docpilot.eval.tool_necessity import main as tool_necessity_main

        return tool_necessity_main(args[1:])
    if args and args[0] == "benchmark":
        from docpilot.eval.benchmark import main as benchmark_main

        return benchmark_main(args[1:])
    if args and args[0] == "code-benchmark":
        from docpilot.eval.code_benchmark import main as code_benchmark_main

        return code_benchmark_main(args[1:])
    from docpilot.eval.judge_ab import main as judge_ab_main

    return judge_ab_main(args)


if __name__ == "__main__":
    raise SystemExit(main())