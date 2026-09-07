"""Tool interface and shared request/result types (SPEC §5, PLAN §9).

Phase 3's locked contract: a **plain**, self-contained external-data tool
behind the :class:`Tool` interface — no MCP protocol or SDK.  The agent wiring
imports exactly ``Tool``, ``ToolRequest``, ``ToolResult`` and ``GitHubTool``
from :mod:`docpilot.tools`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolRequest:
    """A single tool invocation.

    Attributes:
        name: The tool-scoped action to run, e.g. ``"github.search_issues"``.
        params: Action-specific parameters (str keys, JSON-ish values such as
            strings, ints and small lists).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of a tool invocation.

    Attributes:
        ok: ``True`` when the tool produced usable evidence, ``False`` for
            configuration errors, HTTP errors and network failures.
        summary: Concise, human-readable evidence text for an answer LLM to
            consume directly (never the raw upstream payload).
        source_label: Citation footer label for the *result as a whole*.
            The GitHub tool keeps this ``None`` and carries a per-record
            ``source_label`` in each ``items`` entry instead — the agent
            wiring builds per-source citation labels from ``items``.
        items: Trimmed JSON-ish records for inspectability/debug views only.
        error: Machine-usable error message when ``ok`` is ``False``.
    """

    ok: bool
    summary: str
    source_label: str | None = None
    items: list[dict] = field(default_factory=list)
    error: str | None = None


class Tool(ABC):
    """Interface for external-data tools (live issues, repo state, PRs, …).

    Concrete implementations expose a ``name`` (e.g. ``"github"``) and a set
    of actions selected via :attr:`ToolRequest.name`.  The interface stays
    vendor-free: business logic calls :meth:`execute`, never a raw vendor SDK
    or HTTP call.

    Implementations must **never raise** for expected failures — missing
    configuration, HTTP errors and timeouts all become a non-``ok``
    :class:`ToolResult`.
    """

    name: str

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResult:
        """Run the action described by *request* and return its result.

        Args:
            request: The action name and its parameters.

        Returns:
            A :class:`ToolResult` — ``ok=True`` with evidence, or ``ok=False``
            with an explanatory ``error`` (never an exception).
        """
        ...