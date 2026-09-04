from abc import ABC, abstractmethod

from docpilot.core.models import Document, Chunk


class Parser(ABC):
    """Interface for parsing a Document into structured content."""

    @abstractmethod
    def parse(self, doc: Document) -> list[Chunk]:
        """Parse a document and return its initial chunks."""
        ...
