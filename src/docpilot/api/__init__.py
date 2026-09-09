"""Phase 5 — FastAPI + SSE + SQLite session API (SPEC §7).

Modules:
    - :mod:`docpilot.api.sse`     SSE wire-format helpers
    - :mod:`docpilot.api.store`   SQLite session/message persistence
    - :mod:`docpilot.api.service` event orchestration (the streaming ask path)
    - :mod:`docpilot.api.app`     the FastAPI application
"""