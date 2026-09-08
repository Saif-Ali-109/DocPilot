"""GitHub REST tool behind the :class:`Tool` interface (SPEC §5, PLAN §4).

Phase 3's locked design: a **plain** GitHub REST API client — no MCP protocol
or SDK.  The tool resolves a repository (request params ``owner``/``repo`` >
constructor > config), makes exactly one GET per action, and never raises:
every expected failure (missing PAT, missing repo, HTTP >= 400, request
exception/timeout) becomes a non-``ok`` :class:`ToolResult`.

Actions (selected via :attr:`ToolRequest.name`):

``github.search_issues``
    * params: ``query`` (required), ``per_page`` (default 3)
    * endpoint: ``GET {api_base}/search/issues``
    * ``q`` construction rule (locked decision, PLAN §4): the caller's query
      passes through **verbatim**.  Only when it contains no GitHub search
      qualifier at all (no whitespace-delimited ``key:value`` token such as
      ``repo:``, ``is:open``, ``label:``) do we append
      ``repo:{owner}/{repo} in:title type:issue``.  The repo scoping is what
      makes the per-item ``source_label`` (``github:{owner}/{repo}#{n}``)
      truthful — an unconstrained search across all of GitHub could surface
      issues from other repos and mislabel them.  A query that already
      constrains itself is never augmented: the judge's terms pass through
      honestly.

``github.list_issues``
    * params: ``state`` ``open``|``closed``|``all`` (default ``open``),
      ``sort`` ``created``|``updated`` (default ``created``),
      ``per_page`` (default 3)
    * endpoint: ``GET {api_base}/repos/{owner}/{repo}/issues``

``github.get_commits``
    * params: ``ref`` (default: omitted → GitHub's own default branch),
      ``per_page`` (default 3)
    * endpoint: ``GET {api_base}/repos/{owner}/{repo}/commits`` (+ ``sha=ref``
      when ``ref`` is given)

Every record carries a per-item ``source_label`` (issues:
``github:{owner}/{repo}#{number}``; commits:
``github:{owner}/{repo}@{sha[:7]}``) so the agent wiring builds per-source
citations from :attr:`ToolResult.items`; the result-level ``source_label``
stays ``None``.

Hermetic-test seam: pass a ``request_fn`` callable to the constructor.  It is
invoked with the same kwargs as ``requests.request`` (``params``, ``headers``,
``timeout``) and its return value must expose ``status_code``, ``headers``
(a dict) and ``.json()``.  When omitted, the real ``requests`` library is
used.  Auth headers are attached in :meth:`GitHubTool._request` and are never
logged — only the action, URL, params, status and rate-limit header land in
DEBUG logs (AGENTS.md inspectability requirement).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import requests

from docpilot import config
from docpilot.tools.base import Tool, ToolRequest, ToolResult

logger = logging.getLogger(__name__)

#: Maximum number of records surfaced in ``ToolResult.items`` / the summary.
MAX_ITEMS = 5
#: Per-record text length cap (titles, commit message first lines, …).
MAX_TEXT_CHARS = 200
#: Hard ceiling for the upstream ``per_page`` parameter (GitHub API limit).
MAX_PER_PAGE = 100

#: Exact locked error for a missing PAT — no HTTP call is ever attempted.
NO_PAT_ERROR = (
    "GITHUB_PAT not configured — set GITHUB_PAT in .env to enable the GitHub tool"
)
#: Exact locked error for a missing repository resolution.
NO_REPO_ERROR = (
    "no repository configured — set GITHUB_OWNER/GITHUB_REPO in .env "
    "or pass owner/repo in the request params"
)

#: Matches a GitHub search qualifier token (e.g. ``repo:fastapi/fastapi``,
#: ``is:open``, ``in:title``).  Requires a non-empty value after the colon so
#: prose such as ``"Python: what is …"`` is not mistaken for a qualifier.
_QUALIFIER_RE = re.compile(r"\b[a-z_][a-z0-9_-]*:[^\s:]+", re.IGNORECASE)


class GitHubTool(Tool):
    """Plain GitHub REST tool with three actions (see module docstring).

    Constructor args (each ``None`` falls back to ``docpilot.config``):

        pat: GitHub Personal Access Token.  ``None`` → ``config.GITHUB_PAT``.
            An empty PAT disables the tool: :meth:`execute` returns a
            non-``ok`` result and makes **no** HTTP call (locked decision).
        api_base: REST base URL.  ``None`` → ``config.GITHUB_API_BASE``.
        repo_owner / repo_name: default repository.  ``None`` →
            ``config.GITHUB_OWNER`` / ``config.GITHUB_REPO``.  Request params
            ``owner``/``repo`` win over these (each side resolved
            independently).
        timeout: per-request timeout in seconds (default 15).
        request_fn: hermetic-test seam — a ``requests.request``-compatible
            callable.  ``None`` → real ``requests``.
    """

    name = "github"

    def __init__(
        self,
        *,
        pat: str | None = None,
        api_base: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        timeout: int = 15,
        request_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._pat = pat if pat is not None else config.GITHUB_PAT
        self._api_base = (
            api_base if api_base is not None else config.GITHUB_API_BASE
        ).rstrip("/")
        self._repo_owner = repo_owner if repo_owner is not None else config.GITHUB_OWNER
        self._repo_name = repo_name if repo_name is not None else config.GITHUB_REPO
        self._timeout = timeout
        self._request_fn = request_fn

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def actions(self) -> dict[str, Callable[[dict[str, Any]], ToolResult]]:
        """Map of action name → bound handler (the action surface)."""
        return {
            "github.search_issues": self._search_issues,
            "github.list_issues": self._list_issues,
            "github.get_commits": self._get_commits,
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        """Run one of the three GitHub actions.

        Never raises: every expected failure — missing PAT, unknown action,
        missing repo, HTTP error or network exception — returns a non-``ok``
        result.  A missing PAT short-circuits before any HTTP call.
        """
        if not self._pat:
            return ToolResult(ok=False, summary="", error=NO_PAT_ERROR)

        handler = self.actions.get(request.name)
        if handler is None:
            return ToolResult(
                ok=False,
                summary="",
                error=(
                    f"unknown action {request.name!r} — "
                    f"expected one of {sorted(self.actions)}"
                ),
            )
        return handler(request.params)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _search_issues(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query")
        if not query or not str(query).strip():
            return ToolResult(
                ok=False,
                summary="",
                error="github.search_issues requires a non-empty 'query' param",
            )
        query = str(query).strip()

        repo = self._resolve_repo(params)
        if repo is None:
            return ToolResult(ok=False, summary="", error=NO_REPO_ERROR)
        owner, repo_name = repo

        q = self._build_search_q(query, owner, repo_name)
        http_params: dict[str, str] = {
            "q": q,
            "per_page": str(self._clamp_per_page(params.get("per_page"))),
        }
        url = f"{self._api_base}/search/issues"
        payload, failure = self._get_json(url, http_params, "github.search_issues")
        if failure is not None:
            return failure

        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        items = [self._issue_record(i, owner, repo_name) for i in raw_items[:MAX_ITEMS]]
        summary = self._issue_summary(items, owner, repo_name, q=q)
        return ToolResult(ok=True, summary=summary, items=items)

    def _list_issues(self, params: dict[str, Any]) -> ToolResult:
        state = str(params.get("state") or "open")
        sort = str(params.get("sort") or "created")
        if state not in ("open", "closed", "all"):
            return ToolResult(
                ok=False,
                summary="",
                error=f"invalid state {state!r} — expected 'open', 'closed' or 'all'",
            )
        if sort not in ("created", "updated"):
            return ToolResult(
                ok=False,
                summary="",
                error=f"invalid sort {sort!r} — expected 'created' or 'updated'",
            )

        repo = self._resolve_repo(params)
        if repo is None:
            return ToolResult(ok=False, summary="", error=NO_REPO_ERROR)
        owner, repo_name = repo

        http_params: dict[str, str] = {
            "state": state,
            "sort": sort,
            "per_page": str(self._clamp_per_page(params.get("per_page"))),
        }
        url = f"{self._api_base}/repos/{owner}/{repo_name}/issues"
        payload, failure = self._get_json(url, http_params, "github.list_issues")
        if failure is not None:
            return failure

        raw_items = payload if isinstance(payload, list) else []
        items = [self._issue_record(i, owner, repo_name) for i in raw_items[:MAX_ITEMS]]
        summary = self._issue_summary(items, owner, repo_name, state=state)
        return ToolResult(ok=True, summary=summary, items=items)

    def _get_commits(self, params: dict[str, Any]) -> ToolResult:
        repo = self._resolve_repo(params)
        if repo is None:
            return ToolResult(ok=False, summary="", error=NO_REPO_ERROR)
        owner, repo_name = repo

        ref = params.get("ref")
        ref = str(ref).strip() if ref is not None else None
        http_params: dict[str, str] = {
            "per_page": str(self._clamp_per_page(params.get("per_page"))),
        }
        if ref:
            # Omitting ``sha`` makes GitHub list the repository's *default
            # branch* — that is the tool's "<default_branch>" default.
            http_params["sha"] = ref

        url = f"{self._api_base}/repos/{owner}/{repo_name}/commits"
        payload, failure = self._get_json(url, http_params, "github.get_commits")
        if failure is not None:
            return failure

        raw_items = payload if isinstance(payload, list) else []
        items = [self._commit_record(i, owner, repo_name) for i in raw_items[:MAX_ITEMS]]
        ref_label = ref or "<default_branch>"
        parts = []
        for i in items:
            by = f", by {i['author_name']}" if i.get("author_name") else ""
            parts.append(f"{i['short_sha']} {i['message_first_line']} ({str(i['date'])[:10]}{by})")
        head = f"{len(parts)} commit(s) in {owner}/{repo_name}@{ref_label}"
        summary = f"{head}: " + " — ".join(parts) if parts else head
        return ToolResult(ok=True, summary=summary, items=items)

    # ------------------------------------------------------------------
    # HTTP plumbing (never raises; never logs auth)
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> Any:
        """Low-level GET seam; returns a response-like object.

        Auth headers are attached here and are **never** logged.  With an
        injected ``request_fn`` the call is delegated to it (hermetic tests);
        otherwise the real ``requests`` library is used.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._pat}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "docpilot",
        }
        if self._request_fn is not None:
            return self._request_fn(
                method, url, params=params, headers=headers, timeout=self._timeout
            )
        return requests.request(
            method, url, params=params, headers=headers, timeout=self._timeout
        )

    def _get_json(
        self, url: str, params: dict[str, str], action: str
    ) -> tuple[Any, ToolResult | None]:
        """GET *url*; return ``(payload, None)`` on success or
        ``(None, ToolResult)`` on any failure.  Never raises.

        Logs the request at DEBUG (action + URL + params — never headers, so
        the PAT stays out of the logs) and the outcome (status, item count,
        ``X-RateLimit-Remaining`` when present).
        """
        logger.debug("github tool action=%s url=%s params=%s", action, url, params)
        try:
            response = self._request("GET", url, params=params)
        except requests.RequestException as exc:
            logger.debug(
                "github tool action=%s url=%s request failed: %s", action, url, exc
            )
            return None, ToolResult(
                ok=False, summary="", error=f"GitHub request failed: {exc}"
            )

        status = getattr(response, "status_code", None)
        headers = getattr(response, "headers", None) or {}
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if status is not None and status >= 400:
            message = self._error_message(payload, response, status)
            logger.debug(
                "github tool action=%s status=%s message=%s", action, status, message
            )
            return None, ToolResult(
                ok=False, summary="", error=f"GitHub error {status}: {message}"
            )

        logger.debug(
            "github tool action=%s status=%s items=%s rate_limit_remaining=%s",
            action,
            status,
            self._payload_count(payload),
            headers.get("X-RateLimit-Remaining"),
        )
        return payload, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_repo(self, params: dict[str, Any]) -> tuple[str, str] | None:
        """Resolve ``(owner, repo)`` from request params → constructor/config.

        Each side is resolved independently: ``params["owner"]`` wins for the
        owner, then ``repo_owner``; same for the repo name.  ``None`` is
        returned when either side ends up empty.
        """
        owner = params.get("owner") or self._repo_owner
        repo = params.get("repo") or self._repo_name
        if not owner or not repo:
            return None
        return str(owner), str(repo)

    @staticmethod
    def _build_search_q(query: str, owner: str, repo: str) -> str:
        """Build the ``q`` value for ``/search/issues`` (see module docstring).

        Rule: the caller's query passes through verbatim.  Only when it
        carries no GitHub search qualifier do we append
        ``repo:{owner}/{repo} in:title type:issue`` (repo scoping keeps the
        per-item ``source_label`` truthful); a query that already constrains
        itself is never augmented.
        """
        if _QUALIFIER_RE.search(query):
            return query
        return f"{query} repo:{owner}/{repo} in:title type:issue"

    def _issue_record(self, issue: dict, owner: str, repo: str) -> dict:
        number = issue.get("number")
        labels = [
            str(label.get("name", ""))
            for label in issue.get("labels", [])
            if isinstance(label, dict)
        ]
        return {
            "number": number,
            "title": self._trim(issue.get("title")),
            "state": issue.get("state", ""),
            "html_url": issue.get("html_url", ""),
            "created_at": issue.get("created_at", ""),
            "labels": labels,
            "source_label": f"github:{owner}/{repo}#{number}",
        }

    def _commit_record(self, commit: dict, owner: str, repo: str) -> dict:
        sha = str(commit.get("sha", ""))
        commit_obj = commit.get("commit") or {}
        message = str(commit_obj.get("message", "")).strip()
        first_line = message.splitlines()[0].strip() if message else ""
        author = commit_obj.get("author") or {}
        return {
            "sha": sha,
            "short_sha": sha[:7],
            "message_first_line": self._trim(first_line),
            "html_url": commit.get("html_url", ""),
            "date": author.get("date", ""),
            "author_name": author.get("name", ""),
            "source_label": f"github:{owner}/{repo}@{sha[:7]}",
        }

    def _issue_summary(
        self,
        items: list[dict],
        owner: str,
        repo: str,
        *,
        q: str | None = None,
        state: str | None = None,
    ) -> str:
        """Human-readable evidence line: counts, repo, matching ``q`` or
        ``state``, and one compact segment per issue."""
        parts = [
            f"#{i['number']} {i['title']} (created {str(i['created_at'])[:10]})"
            for i in items
        ]
        if q is not None:
            head = f"{len(parts)} matching issue(s) in {owner}/{repo} for {q!r}"
        else:
            head = f"{len(parts)} {state} issue(s) in {owner}/{repo}"
        return f"{head}: " + " — ".join(parts) if parts else head

    @staticmethod
    def _clamp_per_page(value: Any, default: int = 3) -> int:
        """Coerce *value* into ``[1, MAX_PER_PAGE]`` (default 3)."""
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = default
        return max(1, min(n, MAX_PER_PAGE))

    @staticmethod
    def _trim(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
        """Collapse *text* to ≤ *limit* characters (single-line, stripped)."""
        s = str(text or "").strip()
        if len(s) <= limit:
            return s
        return s[: limit - 1] + "…"

    @staticmethod
    def _error_message(payload: Any, response: Any, status: int) -> str:
        """Best-effort human message from a GitHub error response body."""
        if isinstance(payload, dict):
            message = str(payload.get("message") or "")
            if message:
                return message
        reason = getattr(response, "reason", None)
        if reason:
            return str(reason)
        return f"HTTP {status}"

    @staticmethod
    def _payload_count(payload: Any) -> int:
        """Record count of a success payload (list or ``{"items": [...]}``)."""
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                return len(items)
        return 0