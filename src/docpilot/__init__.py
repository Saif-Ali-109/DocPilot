"""DocPilot — evidence-driven agentic RAG for technical documentation.

The ``docpilot`` console script (see pyproject.toml ``[project.scripts]``)
delegates to the argparse CLI in :mod:`docpilot.cli`, which is also reachable
via ``python -m docpilot`` (see :mod:`docpilot.__main__`).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load the project .env BEFORE anything can import ragkit.config (which reads
# os.environ at import time and caches it). Importing any ``docpilot.*``
# submodule runs this package init first, so the framework keys that ragkit
# owns (POSTGRES_*, EMBEDDING_MODEL, GROQ_*, RERANK_*, RETRIEVAL_TOP_K) are
# populated before a ragkit module first reads them.
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

from docpilot.cli import main

if __name__ == "__main__":  # defensive; console scripts call main() directly
    sys.exit(main())