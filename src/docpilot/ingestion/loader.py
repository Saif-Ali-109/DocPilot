from abc import ABC, abstractmethod

from docpilot.core.models import Document


class DocumentLoader(ABC):
    """Interface for loading raw documentation files from a corpus."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return all documents from the corpus."""
        ...
