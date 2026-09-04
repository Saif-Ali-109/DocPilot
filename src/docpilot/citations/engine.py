from abc import ABC, abstractmethod

from docpilot.core.models import SourceRef


class CitationEngine(ABC):
    """Interface for formatting answers with inline citation markers."""

    @abstractmethod
    def format_answer(self, raw_answer: str, sources: list[SourceRef]) -> tuple[str, str]:
        """Format an LLM answer with citation markers.

        Returns:
            (answer_with_markers, footer_string)
        """
        ...
