from abc import ABC, abstractmethod

from docpilot.core.models import Document, Chunk


class Chunker(ABC):
    """Interface for splitting documents into semantic chunks."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document into chunks respecting semantic boundaries."""
        ...
