import pytest


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    """Every test writes its spans to its own temp file, never to traces/ in the project."""
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    return path
