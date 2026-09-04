import logging
from pathlib import Path

from docpilot.core.models import Document
from docpilot.ingestion.loader import DocumentLoader

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")


class FastAPIDocumentLoader(DocumentLoader):
    """Load Markdown/MDX source files from the FastAPI documentation corpus."""

    def __init__(self, docs_dir: Path = DOCS_DIR) -> None:
        self.docs_dir = docs_dir
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.docs_dir}")

    def load(self) -> list[Document]:
        md_files: list[Path] = []
        for ext in ("*.md", "*.mdx"):
            md_files.extend(self.docs_dir.rglob(ext))
        logger.info("Found %d document files in %s", len(md_files), self.docs_dir)
        documents: list[Document] = []
        for path in sorted(md_files):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                relative = str(path.relative_to(self.docs_dir))
                documents.append(Document(file_path=relative, content=content))
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
        logger.info("Loaded %d documents", len(documents))
        return documents
