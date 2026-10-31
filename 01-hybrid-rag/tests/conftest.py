import pytest
from tracing import otel


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    """Every test writes its spans to its own temp file, never to traces/ in the project."""
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def no_export():
    """Tests never send spans to a real Langfuse, even if .env has its keys."""
    otel.disable()
    yield
    otel.reset()
