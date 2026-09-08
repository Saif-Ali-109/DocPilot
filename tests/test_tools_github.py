"""Hermetic tests for the Phase 3 GitHub tool (SPEC §5, PLAN §4).

All tests inject a fake ``request_fn`` into :class:`GitHubTool` — zero network
calls.  Coverage of the locked Tool contract and GitHubTool behaviours:

    * the ``Tool`` ABC cannot be instantiated directly;
    * ``GitHubTool.name == "github"``;
    * search_issues: success shape (summary, trimmed items, per-item
      ``source_label``), default ``per_page``, honest qualifier pass-through
      vs. scoped ``q`` construction, non-empty-query guard;
    * list_issues: success with default state=open / sort=created, request
      ``owner``/``repo`` overriding the defaults;
    * get_commits: success, explicit ``ref`` → ``sha`` param, omitted ``ref``
      → no ``sha`` (default branch);
    * HTTP >= 400 and network exceptions → ``ok=False``, never raised;
    * missing PAT → exact "GITHUB_PAT not configured" message and **zero**
      HTTP calls (locked decision);
    * missing repo → exact "no repository configured" message;
    * unknown action → error, no HTTP call;
    * ``items`` bounded by ``MAX_ITEMS``;
    * DEBUG logs never contain the PAT.
"""

from __future__ import annotations

import logging

import pytest
import requests

from docpilot import config
from docpilot.tools import GitHubTool, Tool, ToolRequest

API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# Fakes (no network anywhere in this file)
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal ``requests.Response`` stand-in: status_code, headers, .json()."""

    def __init__(
        self,
        status_code: int,
        payload,
        headers: dict | None = None,
        reason: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = dict(headers or {})
        self.reason = reason

    def json(self):
        return self._payload


class FakeRequester:
    """Injected ``request_fn``: serves canned responses, records every call."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("FakeRequester has no canned response left")
        return self.responses.pop(0)


def make_issue(
    number: int = 1234,
    title: str = "OAuth token expires",
    state: str = "open",
    created: str = "2026-09-01T10:00:00Z",
) -> dict:
    return {
        "number": number,
        "title": title,
        "state": state,
        "html_url": f"https://github.com/acme/widget/issues/{number}",
        "created_at": created,
        "labels": [{"name": "bug"}, {"name": "auth"}],
    }


def make_commit(
    sha: str = "a" * 40,
    message: str = "Fix oauth refresh flow\n\nLong body here.",
    date: str = "2026-08-30T09:00:00Z",
    author_name: str = "Test Bot",
) -> dict:
    return {
        "sha": sha,
        "html_url": f"https://github.com/acme/widget/commit/{sha}",
        "commit": {
            "message": message,
            "author": {"name": author_name, "date": date},
        },
    }


def make_tool(requester, *, repo_owner: str = "acme", repo_name: str = "widget", **kwargs) -> GitHubTool:
    """GitHubTool with explicit repo defaults (hermetic vs. local .env)."""
    return GitHubTool(
        pat="ghp_test_token",
        api_base=API_BASE,
        repo_owner=repo_owner,
        repo_name=repo_name,
        request_fn=requester,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


def test_tool_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Tool()  # abstract execute → TypeError


def test_github_tool_name_is_github() -> None:
    assert GitHubTool.name == "github"


# ---------------------------------------------------------------------------
# search_issues
# ---------------------------------------------------------------------------


def test_search_issues_success_shape_and_default_per_page() -> None:
    requester = FakeRequester(
        [
            FakeResponse(
                200,
                {"items": [make_issue(1234), make_issue(5678, title="Rate limit 429 errors")]},
                headers={"X-RateLimit-Remaining": "4992"},
            )
        ]
    )
    tool = make_tool(requester)

    result = tool.execute(
        ToolRequest("github.search_issues", {"query": "oauth token expiration"})
    )

    assert result.ok is True
    assert result.error is None
    assert result.source_label is None  # per-item labels only (locked contract)

    # Summary is human-readable evidence for an answer LLM.
    assert "#1234" in result.summary and "OAuth token expires" in result.summary
    assert "acme/widget" in result.summary
    assert "oauth token expiration" in result.summary
    assert "(created 2026-09-01)" in result.summary

    # One GET to /search/issues with the constructed q + default per_page=3.
    (method, url, kwargs) = requester.calls[0]
    assert method == "GET"
    assert url == f"{API_BASE}/search/issues"
    assert kwargs["params"]["q"] == "oauth token expiration repo:acme/widget in:title type:issue"
    assert kwargs["params"]["per_page"] == "3"

    # Locked request headers (auth header present on the wire, never logged).
    headers = kwargs["headers"]
    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["Authorization"] == "Bearer ghp_test_token"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"

    # Trimmed inspectability records with per-item source labels.
    assert len(result.items) == 2
    item = result.items[0]
    assert item["number"] == 1234
    assert item["title"] == "OAuth token expires"
    assert item["state"] == "open"
    assert item["html_url"].endswith("/issues/1234")
    assert item["created_at"] == "2026-09-01T10:00:00Z"
    assert item["labels"] == ["bug", "auth"]
    assert item["source_label"] == "github:acme/widget#1234"
    assert set(item) == {
        "number", "title", "state", "html_url", "created_at", "labels", "source_label",
    }


def test_search_issues_constrained_query_passes_through_verbatim() -> None:
    requester = FakeRequester([FakeResponse(200, {"items": []})])
    tool = make_tool(requester)
    q_in = "oauth is:open repo:fastapi/fastapi in:title"

    result = tool.execute(ToolRequest("github.search_issues", {"query": q_in}))

    assert result.ok is True
    # A query that already constrains itself is never augmented (honest
    # pass-through of the judge's terms).
    assert requester.calls[0][2]["params"]["q"] == q_in


def test_search_issues_requires_nonempty_query() -> None:
    requester = FakeRequester([])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.search_issues", {}))

    assert result.ok is False
    assert "non-empty 'query'" in (result.error or "")
    assert requester.calls == []  # no HTTP call for a malformed request


# ---------------------------------------------------------------------------
# list_issues
# ---------------------------------------------------------------------------


def test_list_issues_success_default_state_open() -> None:
    requester = FakeRequester([FakeResponse(200, [make_issue(10), make_issue(11)])])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.list_issues", {}))

    assert result.ok is True
    (method, url, kwargs) = requester.calls[0]
    assert method == "GET"
    assert url == f"{API_BASE}/repos/acme/widget/issues"
    assert kwargs["params"] == {"state": "open", "sort": "created", "per_page": "3"}

    assert result.summary.startswith("2 open issue(s) in acme/widget")
    assert "#10" in result.summary and "(created 2026-09-01)" in result.summary
    assert result.items[0]["source_label"] == "github:acme/widget#10"


def test_list_issues_request_params_override_default_repo() -> None:
    requester = FakeRequester([FakeResponse(200, [make_issue(9)])])
    tool = make_tool(requester)  # defaults to acme/widget

    result = tool.execute(
        ToolRequest("github.list_issues", {"owner": "fastapi", "repo": "fastapi"})
    )

    assert result.ok is True
    assert requester.calls[0][1] == f"{API_BASE}/repos/fastapi/fastapi/issues"
    assert result.items[0]["source_label"] == "github:fastapi/fastapi#9"
    assert "fastapi/fastapi" in result.summary


# ---------------------------------------------------------------------------
# get_commits
# ---------------------------------------------------------------------------


def test_get_commits_success_without_ref_uses_default_branch() -> None:
    requester = FakeRequester([FakeResponse(200, [make_commit()])])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.get_commits", {}))

    assert result.ok is True
    (method, url, kwargs) = requester.calls[0]
    assert method == "GET"
    assert url == f"{API_BASE}/repos/acme/widget/commits"
    assert kwargs["params"] == {"per_page": "3"}  # no sha → GitHub default branch
    assert "<default_branch>" in result.summary

    item = result.items[0]
    assert item["short_sha"] == "aaaaaaa"
    assert item["message_first_line"] == "Fix oauth refresh flow"
    assert item["date"] == "2026-08-30T09:00:00Z"
    assert item["author_name"] == "Test Bot"
    assert "by Test Bot" in result.summary
    assert item["source_label"] == "github:acme/widget@aaaaaaa"


def test_get_commits_success_with_ref_sets_sha_param() -> None:
    requester = FakeRequester([FakeResponse(200, [make_commit(sha="b" * 40)])])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.get_commits", {"ref": "main"}))

    assert result.ok is True
    assert requester.calls[0][2]["params"]["sha"] == "main"
    assert "@main" in result.summary
    assert result.items[0]["source_label"] == "github:acme/widget@bbbbbbb"


# ---------------------------------------------------------------------------
# Failure modes — never raises
# ---------------------------------------------------------------------------


def test_http_error_404_returns_error_never_raises() -> None:
    requester = FakeRequester([FakeResponse(404, {"message": "Not Found"})])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.list_issues", {}))

    assert result.ok is False
    assert result.error == "GitHub error 404: Not Found"
    assert result.items == []
    assert result.summary == ""


def test_network_exception_returns_error_never_raises() -> None:
    def boom(method: str, url: str, **kwargs):
        raise requests.ConnectionError("connection refused")

    tool = make_tool(boom)

    result = tool.execute(ToolRequest("github.list_issues", {}))

    assert result.ok is False
    assert "GitHub request failed" in (result.error or "")
    assert "connection refused" in (result.error or "")


# ---------------------------------------------------------------------------
# Configuration guards (locked decisions)
# ---------------------------------------------------------------------------


def test_missing_pat_short_circuits_with_zero_http_calls(monkeypatch) -> None:
    # pat=None resolves from config.GITHUB_PAT — patched empty to stay hermetic
    # regardless of the local .env.
    monkeypatch.setattr(config, "GITHUB_PAT", "")
    requester = FakeRequester([])
    tool = GitHubTool(pat=None, api_base=API_BASE, request_fn=requester)

    result = tool.execute(ToolRequest("github.list_issues", {}))

    assert result.ok is False
    assert (
        result.error
        == "GITHUB_PAT not configured — set GITHUB_PAT in .env to enable the GitHub tool"
    )
    assert requester.calls == []  # zero HTTP calls (locked decision)


def test_missing_repo_returns_error() -> None:
    requester = FakeRequester([])
    tool = GitHubTool(
        pat="ghp_test_token",
        api_base=API_BASE,
        repo_owner="",  # explicit empty → no repo anywhere
        repo_name="",
        request_fn=requester,
    )

    result = tool.execute(ToolRequest("github.search_issues", {"query": "oauth"}))

    assert result.ok is False
    assert (
        result.error
        == "no repository configured — set GITHUB_OWNER/GITHUB_REPO in .env "
        "or pass owner/repo in the request params"
    )
    assert requester.calls == []


def test_unknown_action_returns_error_no_http() -> None:
    requester = FakeRequester([])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.delete_repo", {}))

    assert result.ok is False
    assert "unknown action 'github.delete_repo'" in (result.error or "")
    assert requester.calls == []


# ---------------------------------------------------------------------------
# Bounds and inspectability
# ---------------------------------------------------------------------------


def test_items_bounded_to_max_items() -> None:
    issues = [make_issue(n, title=f"issue {n}") for n in range(1, 11)]
    requester = FakeRequester([FakeResponse(200, issues)])
    tool = make_tool(requester)

    result = tool.execute(ToolRequest("github.list_issues", {"per_page": 10}))

    assert requester.calls[0][2]["params"]["per_page"] == "10"
    assert len(result.items) == 5  # MAX_ITEMS bound protects LLM context


def test_debug_logs_never_contain_pat(caplog) -> None:
    requester = FakeRequester([FakeResponse(200, [make_issue()])])
    tool = GitHubTool(
        pat="supersecret123",
        api_base=API_BASE,
        repo_owner="acme",
        repo_name="widget",
        request_fn=requester,
    )

    with caplog.at_level(logging.DEBUG, logger="docpilot.tools.github"):
        result = tool.execute(ToolRequest("github.list_issues", {}))

    assert result.ok is True
    assert "supersecret123" not in caplog.text  # PAT never logged (AGENTS.md)
    assert "/repos/acme/widget/issues" in caplog.text
    assert "github.list_issues" in caplog.text