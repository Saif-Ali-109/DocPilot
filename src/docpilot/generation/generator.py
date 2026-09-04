from abc import ABC, abstractmethod


class Generator(ABC):
    """Interface for LLM-based answer generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the LLM and return the generated answer string."""
        ...
