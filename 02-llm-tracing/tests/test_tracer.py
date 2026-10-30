import contextvars
import json
import threading

import pytest

from tracing.tracer import MAX_VALUE_CHARS, current_trace_id, set_attribute, span, traced
from tracing.view import load, render


@pytest.fixture(autouse=True)
def trace_path(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    return path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@traced
def child(x: int) -> int:
    return x * 2


@traced(name="parent_step")
def parent(x: int) -> int:
    return child(x) + child(x + 1)


def test_one_span_records_name_inputs_output_and_timing(trace_path):
    assert child(4) == 8
    [s] = read(trace_path)
    assert s["name"] == "child"
    assert s["inputs"] == {"x": 4}
    assert s["output"] == 8
    assert s["status"] == "ok"
    assert s["parent_id"] is None
    assert s["duration_ms"] >= 0
    assert s["start"].endswith("+00:00")


def test_nested_spans_share_a_trace_and_point_to_their_parent(trace_path):
    parent(1)
    spans = read(trace_path)
    assert [s["name"] for s in spans] == ["child", "child", "parent_step"]  # children end first
    root = spans[-1]
    assert root["parent_id"] is None
    assert {s["trace_id"] for s in spans} == {root["trace_id"]}
    assert all(s["parent_id"] == root["span_id"] for s in spans[:2])


def test_separate_calls_are_separate_traces(trace_path):
    child(1)
    child(2)
    a, b = read(trace_path)
    assert a["trace_id"] != b["trace_id"]


def test_errors_are_recorded_and_still_raised(trace_path):
    @traced
    def boom():
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        boom()
    [s] = read(trace_path)
    assert s["status"] == "error"
    assert s["error"] == "ValueError: bad input"


def test_parent_is_restored_after_a_child_fails(trace_path):
    with span("root"):
        with pytest.raises(KeyError), span("failing"):
            raise KeyError("x")
        with span("after"):
            pass
    by_name = {s["name"]: s for s in read(trace_path)}
    assert by_name["after"]["parent_id"] == by_name["root"]["span_id"]


def test_attributes_attach_to_the_current_span(trace_path):
    @traced
    def search(tenant_id: str):
        set_attribute("tenant_id", tenant_id)
        set_attribute("hits", 3)

    search("pwani")
    set_attribute("ignored", True)  # outside any span: a no-op, not a crash
    [s] = read(trace_path)
    assert s["attributes"] == {"tenant_id": "pwani", "hits": 3}


def test_large_and_odd_values_are_truncated_and_stringified(trace_path):
    @traced
    def big(text, obj):
        return list(range(100))

    big("x" * 2000, object())
    [s] = read(trace_path)
    assert len(s["inputs"]["text"]) == MAX_VALUE_CHARS + 1  # + the ellipsis
    assert s["inputs"]["obj"].startswith("<object object")
    assert len(s["output"]) == 20


def test_methods_do_not_record_self(trace_path):
    class Retriever:
        @traced
        def search(self, q):
            return q

    Retriever().search("rent")
    [s] = read(trace_path)
    assert s["inputs"] == {"q": "rent"}


def test_threads_start_a_new_trace_unless_the_context_is_copied(trace_path):
    ids = {}
    with span("request"):
        ids["request"] = current_trace_id()
        plain = threading.Thread(target=lambda: ids.setdefault("plain", current_trace_id()))
        ctx = contextvars.copy_context()
        copied = threading.Thread(
            target=ctx.run, args=(lambda: ids.setdefault("copied", current_trace_id()),)
        )
        for t in (plain, copied):
            t.start()
            t.join()
    assert ids["plain"] is None
    assert ids["copied"] == ids["request"]


def test_render_draws_the_tree(trace_path):
    parent(1)
    [spans] = load(trace_path).values()
    lines = render(spans)
    assert lines[0].startswith("parent_step")
    assert lines[1].startswith("├── child")
    assert lines[2].startswith("└── child")
