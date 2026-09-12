"""Lexical retrieval: keyword match over chunk content (PLAN §H finding 2).

The dense embedding index is great at semantics but can miss exact API
identifiers / error codes that a keyword matcher would nail.  The
``LexicalSearcher`` interface backs the keyword half of hybrid retrieval;
``PostgresFTSSearcher`` implements it on the existing ``chunks`` table using
PostgreSQL's built-in full-text search (``to_tsvector`` / ``to_tsquery`` /
``@@``) with a GIN expression index declared in ``db/schema.sql`` — no new
dependency, no second store.

**OR-semantics by design:** ``plainto_tsquery`` ANDs every term, so a
natural-language *question* (``"how do I set a custom status code..."``)
never matches any single chunk.  We therefore tokenize the query, drop
English question stopwords, and OR the remaining lexemes — the corpus then
matches any chunk sharing *some* of the query's vocabulary, and ``ts_rank``
orders by how much of it the chunk shares.  Exact identifiers (``Header``,
``response_model``) are preserved as their own lexemes; underscore
identifiers parse into AND-ed parts inside one OR branch.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from docpilot.core.models import Chunk, RetrieverResult

_FTS_CONFIG = "english"  # matches the corpus language (config.RETRIEVAL_LANGUAGE)

# PostgreSQL text-search (regconfig) names that ``PostgresFTSSearcher`` will
# accept.  The regconfig is user/kv-configurable input and must never reach
# the SQL text unvalidated — every accepted name here is a literal server
# keyword, so members can be bound safely as query parameters
# (``%s::regconfig``).  Keeping the whitelist at module scope means the
# constructor's ``fts_config`` argument gets the same guarantee no matter
# where it comes from (default, caller, or future config plumbing).
_VALID_FTS_CONFIGS = frozenset(
    {
        "english", "simple", "spanish", "french", "german", "italian",
        "portuguese", "dutch", "russian", "danish", "finnish", "norwegian",
        "swedish", "turkish", "hungarian", "romanian", "czech", "polish",
        "greek", "arabic", "hindi", "japanese", "korean", "chinese",
        "thai", "catalan",
    }
)

# English question scaffolding — noise tokens that would make the OR-query
# match unrelated chunks (tsvector stores most real words, so document-side
# stopwords are *not* stripped — only the query side).
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "for", "on", "in", "with",
        "from", "at", "by", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "how", "what", "why", "when", "where", "which",
        "while", "who", "whom", "whose", "can", "could", "would", "should",
        "will", "shall", "may", "might", "must", "not", "no", "yes", "so",
        "if", "then", "than", "that", "this", "these", "those", "it", "its",
        "i", "me", "my", "we", "us", "our", "you", "your", "they", "them",
        "their", "he", "him", "his", "she", "her", "as", "such", "also",
        "each", "any", "all", "both", "same", "like", "want", "like", "use",
        "used", "make", "get", "one", "into", "out", "over", "up", "down",
        "more", "most", "other", "between", "against", "about",
    }
)


class LexicalSearcher(ABC):
    """Interface for keyword/lexical search over the chunk corpus."""

    @abstractmethod
    def search(
        self, query: str, top_k: int, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Return chunks matching *query* by keyword, ordered by relevance.

        Args:
            query: The user's natural-language query (free text; the
                implementation is responsible for tokenizing/query-building).
            top_k: Maximum results to return.
            language: If set, restrict to chunks whose ``language`` column
                matches. ``None`` disables filtering.
        """
        ...


class PostgresFTSSearcher(LexicalSearcher):
    """Lexical search over the ``chunks`` table via PostgreSQL full-text.

    ``plainto_tsquery`` folds the query into lexemes (stems + exact tokens),
    so ``"RequestValidationError"`` and ``"request validation error"`` both
    match the same corpus vector.  Ranking uses ``ts_rank`` — its absolute
    scale differs from cosine similarity, which is exactly why hybrid fusion
    uses Reciprocal Rank Fusion on *positions* instead of raw scores.
    """

    def __init__(self, conn, fts_config: str = _FTS_CONFIG) -> None:
        """Create the searcher over an open psycopg 3 connection.

        Args:
            conn: A psycopg 3 connection with the schema applied
                (``db.connection.ensure_schema``).
            fts_config: PostgreSQL text-search configuration (regconfig) used
                for both ``to_tsvector`` and ``to_tsquery``.  Mono-lingual
                corpus today, but the knob exists because the corpus language
                is a deployment decision, not a hardcoded one.  The name is
                validated against ``_VALID_FTS_CONFIGS`` and then bound as a
                query parameter — invalid input raises here and never touches
                SQL.

        Raises:
            ValueError: If *fts_config* is not a known regconfig name.
        """
        if fts_config not in _VALID_FTS_CONFIGS:
            raise ValueError(
                f"fts_config {fts_config!r} is not a supported PostgreSQL "
                f"text-search configuration; supported: "
                f"{', '.join(sorted(_VALID_FTS_CONFIGS))}."
            )
        self._conn = conn
        self._fts_config = fts_config

    def search(
        self, query: str, top_k: int, *, language: str | None = None
    ) -> list[RetrieverResult]:
        """Return chunks whose full-text vector matches *query*.

        Rows duplicate the :meth:`PgVectorStore.search` result shape so the
        downstream pipeline cannot tell the two retrieval halves apart.
        """
        if not query.strip():
            return []

        terms = _query_terms(query)
        if not terms:
            return []

        # OR-joined lexemes: match first, rank by how many terms the chunk
        # shares.  ts_rank needs the parsed query.  The regconfig is bound as
        # a query parameter (%s::regconfig) — never interpolated — so a bad
        # config can't smuggle SQL; the constructor already rejected anything
        # outside _VALID_FTS_CONFIGS.
        select = (
            "SELECT content, heading_path, source_file, chunk_index, metadata, "
            "ts_rank(to_tsvector(%s::regconfig, content), "
            "to_tsquery(%s::regconfig, %s)) AS rank "
        )
        where = (
            "FROM chunks WHERE to_tsvector(%s::regconfig, content) @@ "
            "to_tsquery(%s::regconfig, %s) "
        )
        if language is not None:
            where += "AND language = %s "
        order = "ORDER BY rank DESC LIMIT %s"

        joined = " | ".join(terms)
        # Placeholder order follows the concatenated SQL text: SELECT's
        # tsvector/tsquery/payload, then WHERE's tsvector/tsquery/payload,
        # then optional language, then LIMIT.
        cfg = self._fts_config
        params: list = [cfg, cfg, joined, cfg, cfg, joined]
        if language is not None:
            params.append(language)
        params.append(top_k)

        with self._conn.cursor() as cur:
            cur.execute(select + where + order, params)
            rows = cur.fetchall()

        results: list[RetrieverResult] = []
        for content, heading_path, source_file, chunk_index, metadata, rank in rows:
            meta = _parse_metadata(metadata)
            chunk_id = meta.pop("chunk_id", "")
            results.append(
                RetrieverResult(
                    chunk=Chunk(
                        id=chunk_id,
                        content=content,
                        heading_path=heading_path,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata=meta,
                    ),
                    score=float(rank),
                )
            )
        return results


def _query_terms(query: str) -> list[str]:
    """Tokenize a question into lexical query lexemes (stopwords dropped)."""
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query)]
    seen: set[str] = set()
    terms: list[str] = []
    for t in tokens:
        if len(t) <= 1 or t in _QUERY_STOPWORDS or t in seen:
            continue
        seen.add(t)
        terms.append(t)
    return terms


def _parse_metadata(metadata) -> dict:
    """Deserialize the ``metadata`` JSONB column (mirrors vector_store)."""
    import json

    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    try:
        return dict(json.loads(metadata))
    except (TypeError, ValueError):
        return {}