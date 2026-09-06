"""DocPilot agent core — Phase 2 agentic retrieval (SPEC §4).

Exports the public API of the Phase 2 agent package.  No langgraph imports
at the package level — the graph wiring lives in :mod:`docpilot.agent.graph`
(AGENT G's responsibility) and is imported lazily by callers.

Typical usage (once the graph is wired)::

    from docpilot.agent import Agent, AgentResult
    from docpilot.agent.gate import HeuristicQueryClassifier
    from docpilot.agent.judge import LLMSufficiencyJudge
    from docpilot.agent.types import DEFAULT_MAX_RETRIES
"""

from docpilot.agent.interface import Agent, AgentResult
from docpilot.agent.types import (
    DEFAULT_MAX_RETRIES,
    AgentLoopState,
    GateDecision,
    Judgment,
    LoopTraceStep,
)

__all__ = [
    # Interfaces
    "Agent",
    "AgentResult",
    # Types / contract
    "GateDecision",
    "Judgment",
    "LoopTraceStep",
    "AgentLoopState",
    # Constants
    "DEFAULT_MAX_RETRIES",
]
