"""DocPilot — evidence-driven agentic RAG for technical documentation.

The ``docpilot`` console script (see pyproject.toml ``[project.scripts]``)
delegates to the argparse CLI in :mod:`docpilot.cli`, which is also reachable
via ``python -m docpilot`` (see :mod:`docpilot.__main__`).
"""

import sys

from docpilot.cli import main

if __name__ == "__main__":  # defensive; console scripts call main() directly
    sys.exit(main())