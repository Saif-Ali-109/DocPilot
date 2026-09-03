## Project

**DocPilot** — an evidence-driven agentic RAG system for technical documentation. It ingests Markdown/MDX docs (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. It is not a plain retrieve-and-answer chatbot: it judges whether its own evidence is sufficient, retries/reformulates searches when it isn't, calls out to GitHub via MCP when static docs can't answer a question (e.g. live issues/repo state), and returns "I don't know" rather than hallucinating. Every retrieval and tool decision must stay inspectable — this is also a demo/debugging project, not just a production pipeline.

This is also the learning/validation project for a future reusable RAG framework — DocPilot comes first, the framework gets extracted from it later, not the other way around.

## Build order

`Reliable classic RAG → Agentic retrieval → MCP tools → Evaluation → API/UI → optional code validation → extract reusable framework components`

1. **Never implement a later phase's functionality to unblock an earlier phase.** If Phase 1 retrieval seems weak, fix retrieval — don't paper over it with an agentic loop or a tool call.
2. **Never skip Phase 4 (Evaluation) to get to Phase 5 (API/UI) faster.** If asked to "just wire up the UI first," build it against the current phase's output but say explicitly that evaluation is still owed before the project can claim the agentic layer is worth its cost.
3. **Do not add code generation/validation (Phase 6) into the core loop** unless explicitly told the project is now at that phase. If a task description implies it, ask first.
4. **Do not build "the framework"** — reusable SDK-shaped generalization — before DocPilot itself works end-to-end and has been evaluated. Interfaces should be clean (see below) but extraction into a standalone package happens later, explicitly.

## Definition of done, per phase

- **Phase 1 (Classic RAG):** ingestion → code-aware chunking (code blocks intact) → local embeddings (BGE-small) → vector DB (Qdrant/pgvector) → retrieval → LLM (Groq) → cited answer. A debug view shows retrieved chunks and scores. This is measurable on its own before any agent logic touches it.
- **Phase 2 (Agentic retrieval):** LangGraph loop that can analyze → search → judge evidence sufficiency → reformulate/retry → answer or refuse. Simple questions must still route through the fast/cheap path — agentic looping is conditional, never mandatory for every query. If you find yourself routing everything through the agent, stop and flag it.
- **Phase 3 (MCP/GitHub):** the agent reaches for GitHub only when static docs are demonstrably insufficient for the question. If you can't construct a real example where this is true, don't wire in the tool call for its own sake.
- **Phase 4 (Evaluation):** a benchmark dataset and tracked metrics exist — retrieval quality, answer correctness, citation correctness, groundedness/hallucination rate, "I don't know" accuracy, latency, retrieval/tool call count — with a classic-RAG-vs-agentic-RAG comparison. Claims of improvement must be backed by this, not asserted.
- **Phase 5 (API/UI):** FastAPI backend, streaming responses, citations and a retrieved-context debug panel surfaced in the UI, basic session/chat history. Streamlit first; React/Next.js only if there's slack in the schedule. Frontend work never delays or reshapes the RAG/agent core.
- **Phase 6 (Code gen/validation):** documentation retrieval → generate code → validate against retrieved API/schema/examples → return code + sources. Only after Phase 1–5 are solid.

## Interfaces (from day one, every phase)

Design around clean interfaces so components are swappable later, even though only one implementation of each exists now. Do not hard-code call sites to a specific vendor/library where an interface should exist instead:

- `DocumentLoader`, `Parser`, `Chunker`, `EmbeddingProvider`, `VectorStore`, `Retriever`, `Reranker`, `Tool`, `Agent`, `Generator`, `CitationEngine`, `Evaluator`

Concretely: don't sprinkle `groq.chat(...)` or `qdrant_client.search(...)` calls directly through business logic — put them behind the relevant interface, even if there's only one backend today. This is what will let BGE/Qdrant/Groq/LangGraph get swapped out later without a rewrite.

## When a request conflicts with this plan

- If asked to add a feature from a later phase early: implement it behind the appropriate interface if truly needed now, but say plainly that this pulls forward work from Phase N and ask whether that's intended.
- If asked to make the agentic loop run on every query "for consistency": push back — this violates the Phase 2 guardrail and the entire cost/latency argument for the phased approach.
- If asked to skip evaluation and go straight to polish: implement what's asked, but note in the response that Phase 4 is still owed and nothing about "agentic RAG being better" is proven yet.
- If unsure which phase a task belongs to, ask before restructuring existing code to fit it.

## Output expectations

- Keep README/pitch language honest: don't describe capabilities (e.g. code validation) as implemented if they're still Phase 6 and unbuilt.
- Every retrieval/tool decision the agent makes at runtime should be loggable/inspectable — this is a stated project requirement, not a nice-to-have.
- Prefer fewer, well-scoped interfaces over premature generalization; the framework-extraction goal is explicitly deferred, so don't over-engineer for a framework that doesn't exist yet.
