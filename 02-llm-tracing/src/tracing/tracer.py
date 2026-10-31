"""A hand-rolled tracer: what Langfuse, Phoenix and OpenTelemetry do, in one small file.

A *trace* is one request (one question). A *span* is one timed step inside it (embed,
search, generate). Spans nest: a span started while another is running becomes its child,
so a trace is a tree:

    answer                      trace 7f3a…, 812 ms
    ├── embed_query                         41 ms
    ├── vector_search                       23 ms
    └── generate                           748 ms

Each finished span is appended to a JSONL file as one line. Children finish first, so they
are written before their parent; the viewer rebuilds the tree from `parent_id`.

The "current span" lives in a ContextVar, which follows the call stack (and asyncio tasks)
but is *not* copied into new threads. Code that hands work to a thread must copy the context
(`contextvars.copy_context().run(...)`) or the thread's spans start a new trace.
"""

import contextvars
import functools
import inspect
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tracing import otel

DEFAULT_TRACE_FILE = Path("traces") / "traces.jsonl"
MAX_VALUE_CHARS = 500  # inputs/outputs are truncated: a trace is for debugging, not storage


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: str  # wall-clock time, ISO 8601 UTC, for humans and dashboards
    duration_ms: float = 0.0  # measured with perf_counter, which is monotonic and precise
    status: str = "ok"
    error: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)


_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("span", default=None)
_write_lock = threading.Lock()  # the FastAPI app answers from several threads at once


def trace_file() -> Path:
    return Path(os.environ.get("TRACE_FILE", DEFAULT_TRACE_FILE))


def mode() -> str:
    """TRACING=on (default): JSONL, plus OTLP export if configured (see otel.py).
    TRACING=otlp: export only, no file. TRACING=off: spans still get ids and timing, nothing
    is written or sent."""
    value = os.environ.get("TRACING", "on").lower()
    return "off" if value in {"0", "off", "false", "no"} else value


def enabled() -> bool:
    return mode() != "off"


def to_jsonable(value: Any) -> Any:
    """Keep JSON-native values; anything else becomes a truncated repr."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_VALUE_CHARS else value[:MAX_VALUE_CHARS] + "…"
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in list(value.items())[:20]}
    return to_jsonable(repr(value))


def _write(span: Span) -> None:
    if mode() in {"off", "otlp"}:
        return
    path = trace_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(span), ensure_ascii=False)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@contextmanager
def span(name: str, **inputs: Any) -> Iterator[Span]:
    """Time a block as a span: `with span("search", k=5) as s: ...`."""
    parent = _current.get()
    exported = otel.start(name, parent.span_id if parent else None) if enabled() else None
    if exported is not None:
        trace_id, span_id = otel.ids(exported)  # one id everywhere: JSONL, trace-view, Langfuse
        otel.register(span_id, exported)
    else:
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
    s = Span(
        name=name,
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent.span_id if parent else None,
        start=datetime.now(UTC).isoformat(timespec="milliseconds"),
        inputs={k: to_jsonable(v) for k, v in inputs.items()},
    )
    if parent is not None and "tenant_id" in parent.attributes:
        s.attributes["tenant_id"] = parent.attributes["tenant_id"]  # filter any step by tenant
    token = _current.set(s)
    started = time.perf_counter()
    try:
        yield s
    except BaseException as e:
        s.status, s.error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        s.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        _current.reset(token)
        if exported is not None:
            otel.finish(exported, s)
        _write(s)


def set_attribute(key: str, value: Any) -> None:
    """Tag the current span (tenant_id, model, tokens…). Does nothing outside a span."""
    if (s := _current.get()) is not None:
        s.attributes[key] = to_jsonable(value)


def current_trace_id() -> str | None:
    s = _current.get()
    return s.trace_id if s else None


def traced(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """Decorator: `@traced` or `@traced(name="embed_query")` records each call as a span."""

    def decorate(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs).arguments
            bound.pop("self", None)
            with span(name or func.__name__, **bound) as s:
                result = func(*args, **kwargs)
                s.output = to_jsonable(result)
                return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
