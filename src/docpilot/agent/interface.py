"""Agent interface and result type (SPEC §4.2, PLAN §3.3).

:class:`AgentResult` is the return type of every ``Agent.run()`` call and
embeds enough data for callers to produce Phase 1-identical output on the
fast path or a rich trace on the agentic path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from docpilot.core.models import SourceRef
from docpilot.agent.types import LoopTraceStep


@dataclass
class AgentResult:
    """Everything the agent produced for a single question.

    Attributes:
        question: The user's original question.
        answer: The final display string — the citation-formatted answer on
            the answering path, or the verbatim "I don't know" sentence
            when refused.
        sources: The numbered :class:`SourceRef` list offered to the LLM.
        trace: The full :class:`LoopTraceStep` trace (gate, search, judge,
            answer / refuse) for inspectability.
        refused: ``True`` when the budget was exhausted and the agent could
            not answer.
        max_retries: The configured maximum number of judge/reformulate
            iterations.
        direct: ``True`` when the gate decided the fast path — the loop was
            never entered and output is identical to a Phase 1 ``ask()``.

    Note:
        This type carries only data.  Stream discipline (answers on stdout,
        logs on stderr) is the caller's responsibility — the agent never
        writes to either stream itself.
    """

    question: str
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    trace: list[LoopTraceStep] = field(default_factory=list)
    refused: bool = False
    max_retries: int = 2
    direct: bool = False


class Agent(ABC):
    """High-level interface for question answering.

    Concrete implementations (AGENT G's graph-backed agent) inject the
    retriever, generator, judge, citation engine and strategy at call time
    or construction time — the interface stays generic.
    """

    @abstractmethod
    def run(self, question: str, **kwargs) -> AgentResult:
        """Answer *question* using the agentic or direct strategy.

        Args:
            question: The user's natural-language question.
            **kwargs: Strategy and dependency overrides (retriever, generator,
                judge, citation_engine, top_k, strategy, language, …).
        """
        ...
