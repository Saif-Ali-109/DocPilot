"""Module entry point — run the CLI with ``python -m docpilot``.

Examples:
    python -m docpilot ingest [--debug]
    python -m docpilot ask "How do I install FastAPI?" [--debug] [--json]

Exit codes: 0 success, 1 runtime error, 2 usage error (see cli.main).
"""

import sys

from docpilot.cli import main

if __name__ == "__main__":
    sys.exit(main())