import pytest

from tracing import otel


@pytest.fixture(autouse=True)
def no_export():
    """Tests never send spans to a real Langfuse, even if .env has its keys."""
    otel.disable()
    yield
    otel.reset()
