## Project

**DocPilot** — an evidence-driven agentic RAG system for technical documentation. It ingests Markdown/MDX docs (code blocks, nested headings, cross-references) and answers questions using retrieved evidence. It is not a plain retrieve-and-answer chatbot: it judges whether its own evidence is sufficient, retries/reformulates searches when it isn't, calls out to GitHub — a plain REST tool behind the `Tool` interface, no MCP protocol (SPEC §5) — when static docs can't answer a question (e.g. live issues/repo state), and returns "I don't know" rather than hallucinating. Every retrieval and tool decision must stay inspectable — this is also a demo/debugging project, not just a production pipeline.

This is also the learning/validation project for a future reusable RAG framework — DocPilot comes first, the framework gets extracted from it later, not the other way around.

## Source of truth

- **SPEC.md is the source of truth** — the approved spec holds every locked decision for every phase.
- **PLAN.md is the working state**, derived from SPEC.md — it holds the HOW: task breakdown, file ownership, execution order, and per-phase exit checklists.
- **ACTIVE.md is the derived current view** — phase, live gates, contract constants, and the load-index table (which SPEC/PLAN lines to read). Never authoritative: if it contradicts SPEC/PLAN, fix ACTIVE.md.
- When they disagree, **SPEC.md wins**; update PLAN.md (and ACTIVE.md) to match.

## Context loading (read this first)

- At every session start, read **`ACTIVE.md` in full** — it is the current
  view (~120 lines) and the only file meant to be read whole.
- **Never read whole `SPEC.md` or `PLAN.md`.** Use the load-index table in
  `ACTIVE.md` §4: read only the cited line ranges — the **core + current-phase
  bundle always**, everything else only when a task actually touches that
  phase's code (PLAN §2–§5 are completed-phase history).
- If a cited line range looks stale (files edited since ACTIVE.md was
  refreshed), `grep -n` for the section heading, read only that section's
  lines, then refresh the range in ACTIVE.md.
- After any change to `SPEC.md` / `PLAN.md` / feature levers / current phase,
  update `ACTIVE.md` (status + line ranges) **in the same commit** — never
  land a SPEC/PLAN change without refreshing it.

## Build order

`Reliable classic RAG → Agentic retrieval → GitHub tooling → Evaluation → API/UI → optional code validation → extract reusable framework components`

1. **Never implement a later phase's functionality to unblock an earlier phase.** If Phase 1 retrieval seems weak, fix retrieval — don't paper over it with an agentic loop or a tool call.
2. **Never skip Phase 4 (Evaluation) to get to Phase 5 (API/UI) faster.** If asked to "just wire up the UI first," build it against the current phase's output but say explicitly that evaluation is still owed before the project can claim the agentic layer is worth its cost.
3. **Do not add code generation/validation (Phase 6) into the core loop** unless explicitly told the project is now at that phase. If a task description implies it, ask first.
4. **Do not build "the framework"** — reusable SDK-shaped generalization — before DocPilot itself works end-to-end and has been evaluated. Interfaces should be clean (see below) but extraction into a standalone package happens later, explicitly.

## Definition of done, per phase

- **Phase 1 (Classic RAG):** ingestion → code-aware chunking (code blocks intact) → local embeddings (BGE-small) → vector DB (Qdrant/pgvector) → retrieval → LLM (Groq) → cited answer. A debug view shows retrieved chunks and scores. This is measurable on its own before any agent logic touches it.
- **Phase 2 (Agentic retrieval):** LangGraph loop that can analyze → search → judge evidence sufficiency → reformulate/retry → answer or refuse. Simple questions must still route through the fast/cheap path — agentic looping is conditional, never mandatory for every query. If you find yourself routing everything through the agent, stop and flag it.
- **Phase 3 (GitHub tooling):** a plain GitHub REST tool behind the `Tool` interface — no MCP protocol/SDK (SPEC §5.1). The agent reaches for GitHub only when static docs are demonstrably insufficient for the question (live issue state, repo state, recent commits). If you can't construct a real example where this is true, don't wire in the tool call for its own sake.
- **Phase 4 (Evaluation):** a benchmark dataset and tracked metrics exist — retrieval quality, answer correctness, citation correctness, groundedness/hallucination rate, "I don't know" accuracy, latency, retrieval/tool call count — with a classic-RAG-vs-agentic-RAG comparison. Claims of improvement must be backed by this, not asserted.
- **Phase 5 (API/UI):** FastAPI backend, streaming responses, citations and a retrieved-context debug panel surfaced in the UI, basic session/chat history. Chainlit first (per SPEC §7); React/Next.js only if there's slack in the schedule. Frontend work never delays or reshapes the RAG/agent core.
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

## Version control

- Repo: `https://github.com/Saif-Ali-109/DocPilot.git`, branch `main`.
- Keep the repo synced with the work: commit every task and every phase as it completes (see PLAN.md §11).
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `build:`). Cut a phase-milestone commit/tag when a phase's exit criteria are met.
- Never commit secrets or `.env`. Push to `origin/main` after each meaningful commit batch.
