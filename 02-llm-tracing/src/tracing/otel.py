"""Send our spans to an OpenTelemetry backend: Langfuse here, but any OTLP/HTTP receiver works.

Our tracer stays the API (`span()`, `@traced`, `set_attribute`). When an exporter is
configured, each span also opens an OpenTelemetry span and takes its ids from it, so the
JSONL file, `trace-view` and Langfuse all show the same trace id. Finished spans go to a
BatchSpanProcessor: a background thread sends them in batches, so a request never waits on
the network (M1 measured ~1.5 ms per span for a synchronous file write).

Configuration, first match wins, from the environment or .env / ../.env (repo root):
- LANGFUSE_HOST + LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY -> {host}/api/public/otel/v1/traces
- OTEL_EXPORTER_OTLP_ENDPOINT (+ OTEL_EXPORTER_OTLP_HEADERS): the standard variables, e.g. Phoenix
Neither set: nothing is exported and spans only go to JSONL.
"""

import base64
import json
import os
import threading
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from tracing.tracer import Span

SERVICE_NAME = "production-ai-systems"

_lock = threading.Lock()
_state: dict[str, Any] = {"configured": False, "tracer": None, "provider": None}
_live: dict[str, trace.Span] = {}  # our span_id -> its open OTel span, so children find parents


def langfuse_target() -> tuple[str, dict[str, str]] | None:
    """(endpoint, headers) from the LANGFUSE_* settings, or None if any is missing."""
    host = os.environ.get("LANGFUSE_HOST", "").rstrip("/")
    public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not (host and public and secret):
        return None
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"}
    return f"{host}/api/public/otel/v1/traces", headers


def exporter_from_env() -> SpanExporter | None:
    load_dotenv(".env")  # never overrides real environment variables
    load_dotenv("../.env")
    if target := langfuse_target():
        endpoint, headers = target
        return OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=5)
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return OTLPSpanExporter(timeout=5)  # reads the standard OTEL_* variables itself
    return None


def configure(exporter: SpanExporter | None = None, *, batch: bool = True) -> bool:
    """Set up export once. Tests pass an in-memory exporter with batch=False."""
    with _lock:
        exporter = exporter or exporter_from_env()
        _state["configured"] = True
        if exporter is None:
            return False
        provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)  # the provider flushes it at interpreter exit
        _state.update(provider=provider, tracer=provider.get_tracer("tracing"))
        return True


def disable() -> None:
    """Export nothing, whatever .env says (tests: never send spans to a real Langfuse)."""
    reset()
    _state["configured"] = True


def reset() -> None:
    """Forget the configuration (tests)."""
    with _lock:
        if _state["provider"] is not None:
            _state["provider"].shutdown()
        _state.update(configured=False, tracer=None, provider=None)
        _live.clear()


def flush() -> None:
    if _state["provider"] is not None:
        _state["provider"].force_flush()


def start(name: str, parent_id: str | None) -> trace.Span | None:
    """Open an OTel span under our parent's, or None when nothing is configured."""
    if not _state["configured"]:
        configure()
    tracer = _state["tracer"]
    if tracer is None:
        return None
    parent = _live.get(parent_id) if parent_id else None
    # An empty Context for roots: never adopt an unrelated OTel span as the parent.
    ctx = trace.set_span_in_context(parent) if parent else otel_context.Context()
    return tracer.start_span(name, context=ctx)


def ids(otel_span: trace.Span) -> tuple[str, str]:
    """(trace_id, span_id) as hex, the same shape our own ids have: 32 and 16 characters."""
    c = otel_span.get_span_context()
    return format(c.trace_id, "032x"), format(c.span_id, "016x")


def register(span_id: str, otel_span: trace.Span) -> None:
    _live[span_id] = otel_span


def _value(value: Any) -> Any:
    """OTel attributes take primitives or lists of str; anything else is sent as JSON."""
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return json.dumps(value, ensure_ascii=False)


def to_attributes(s: "Span") -> dict[str, Any]:
    """Our span's fields as the attribute names Langfuse understands."""
    attrs: dict[str, Any] = {"langfuse.observation.input": json.dumps(s.inputs, ensure_ascii=False)}
    if s.output is not None:
        attrs["langfuse.observation.output"] = json.dumps(s.output, ensure_ascii=False)
    if s.parent_id is None:
        attrs["langfuse.trace.name"] = s.name
    a = s.attributes
    # Always explicit: left unset, Langfuse calls any span with a model name a generation,
    # and `answer` (which records the model it used) would double-count as an LLM call.
    is_llm_call = "tokens_in" in a or "tokens_out" in a
    attrs["langfuse.observation.type"] = "generation" if is_llm_call else "span"
    for key, value in a.items():
        if key == "model" and not is_llm_call:
            attrs["langfuse.observation.metadata.model"] = value
            continue
        if key == "tenant_id":
            # On every span, not only the root: Langfuse filters reliably only then.
            attrs["langfuse.trace.metadata.tenant_id"] = value
            attrs["langfuse.trace.tags"] = [f"tenant:{value}"]
            attrs["langfuse.observation.metadata.tenant_id"] = value
        elif key == "model":
            attrs["gen_ai.request.model"] = value
        elif key == "tokens_in":
            attrs["gen_ai.usage.input_tokens"] = value
        elif key == "tokens_out":
            attrs["gen_ai.usage.output_tokens"] = value
        elif key == "cost_usd":
            attrs["langfuse.observation.cost_details"] = json.dumps({"total": value})
        else:
            attrs[f"langfuse.observation.metadata.{key}"] = _value(value)
    if s.status == "error":
        attrs["langfuse.observation.level"] = "ERROR"
        attrs["langfuse.observation.status_message"] = s.error or ""
    return attrs


def finish(otel_span: trace.Span, s: "Span") -> None:
    otel_span.set_attributes(to_attributes(s))
    if s.status == "error":
        otel_span.set_status(Status(StatusCode.ERROR, s.error))
    otel_span.end()
    _live.pop(s.span_id, None)
