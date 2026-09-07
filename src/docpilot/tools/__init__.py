"""DocPilot tools package — Phase 3 GitHub tooling (SPEC §5, PLAN §4).

The locked Phase 3 contract is a **plain** external-data tool behind the
:class:`~docpilot.tools.base.Tool` interface — no MCP protocol or SDK.  The
agent wiring imports exactly these four names::

    from docpilot.tools import GitHubTool, Tool, ToolRequest, ToolResult
"""

from docpilot.tools.base import Tool, ToolRequest, ToolResult
from docpilot.tools.github import GitHubTool

__all__ = ["GitHubTool", "Tool", "ToolRequest", "ToolResult"]