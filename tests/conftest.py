"""Root pytest configuration for DocPilot tests.

Registers custom markers so pytest emits no PytestUnknownMarkWarning.
"""


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: test requires a live PostgreSQL/pgvector instance (skips when unreachable)",
    )