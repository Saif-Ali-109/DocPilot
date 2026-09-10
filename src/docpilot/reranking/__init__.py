"""Reranking — cross-encoder pass over top-k candidates (PLAN §H finding 1).

The ``Reranker`` interface lives here (AGENTS.md first-class interface, newly
implemented): a reranker takes the top-*candidates* results from the vector
store and re-scores them against the *query* so the final ``top_k`` ordering
reflects cross-encoder relevance instead of raw cosine similarity alone.
"""