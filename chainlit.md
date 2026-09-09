# DocPilot

Evidence-driven agentic RAG for technical documentation.

Ask questions about FastAPI. DocPilot retrieves evidence, judges whether it is
sufficient (retrying/reformulating when it isn't), reaches for live GitHub
data when static docs can't answer, and says **"I don't know"** rather than
hallucinating.

- 📄 **Retrieval debug panel** — every answer shows the retrieved chunks with
  scores, so each claim is inspectable.
- 🔎 **Agentic steps** — gate/judge/tool decisions render as trace steps on
  complex questions.
- 💬 **Streaming** — the answer streams token-by-token (first token in seconds
  on the fast path).
- 📚 **Citations** — numbered inline markers resolve to a source footer.