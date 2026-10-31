import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tracing import otel
from tracing.tracer import set_attribute, span


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_FILE", str(tmp_path / "traces.jsonl"))
    otel.reset()
    memory = InMemorySpanExporter()
    otel.configure(memory, batch=False)  # export each span as it ends, so tests can look
    return memory


def jsonl(tmp_path):
    return [
        json.loads(line) for line in (tmp_path / "traces.jsonl").read_text("utf-8").splitlines()
    ]


def test_jsonl_and_otel_share_trace_and_span_ids(exported, tmp_path):
    with span("answer", question="Cats?"):
        set_attribute("tenant_id", "ruaka-rentals")
        with span("vector_search", k=4):
            pass
    ours = {s["name"]: s for s in jsonl(tmp_path)}
    theirs = {s.name: s for s in exported.get_finished_spans()}

    for name in ("answer", "vector_search"):
        ctx = theirs[name].get_span_context()
        assert format(ctx.trace_id, "032x") == ours[name]["trace_id"]
        assert format(ctx.span_id, "016x") == ours[name]["span_id"]
    assert theirs["vector_search"].parent.span_id == theirs["answer"].get_span_context().span_id
    assert theirs["answer"].parent is None


def test_children_inherit_the_tenant_so_every_step_is_filterable(exported, tmp_path):
    with span("answer"):
        set_attribute("tenant_id", "pwani-lettings")
        with span("embed_query"):
            pass
    child = next(s for s in exported.get_finished_spans() if s.name == "embed_query")
    assert child.attributes["langfuse.trace.metadata.tenant_id"] == "pwani-lettings"
    assert tuple(child.attributes["langfuse.trace.tags"]) == ("tenant:pwani-lettings",)


def test_llm_calls_become_generations_with_model_usage_and_cost(exported):
    with span("generate", model="llama3.2:3b") as s:
        s.output = "No pets [1]."
        s.attributes.update(model="llama3.2:3b", tokens_in=375, tokens_out=27, cost_usd=0.0)
        s.attributes["cost_usd_hosted"] = 0.00153
    [gen] = exported.get_finished_spans()
    a = gen.attributes
    assert a["langfuse.observation.type"] == "generation"
    assert a["gen_ai.request.model"] == "llama3.2:3b"
    assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == (375, 27)
    assert json.loads(a["langfuse.observation.cost_details"]) == {"total": 0.0}
    assert a["langfuse.observation.metadata.cost_usd_hosted"] == 0.00153
    assert json.loads(a["langfuse.observation.input"]) == {"model": "llama3.2:3b"}
    assert json.loads(a["langfuse.observation.output"]) == "No pets [1]."
    assert a["langfuse.trace.name"] == "generate"  # a root span names the trace


def test_errors_are_exported_with_status_and_level(exported):
    with pytest.raises(TimeoutError), span("generate"):
        raise TimeoutError("model did not answer")
    [s] = exported.get_finished_spans()
    assert s.status.status_code == StatusCode.ERROR
    assert s.attributes["langfuse.observation.level"] == "ERROR"
    assert "TimeoutError" in s.attributes["langfuse.observation.status_message"]


def test_lists_of_ids_stay_lists_and_other_values_become_json(exported):
    with span("answer") as s:
        s.attributes.update(retrieved=["a#1", "b#2"], weights=[1.0, 0.5], abstained=False)
    a = exported.get_finished_spans()[0].attributes
    assert tuple(a["langfuse.observation.metadata.retrieved"]) == ("a#1", "b#2")
    assert json.loads(a["langfuse.observation.metadata.weights"]) == [1.0, 0.5]
    assert a["langfuse.observation.metadata.abstained"] is False


def test_tracing_otlp_mode_exports_without_writing_the_file(exported, tmp_path, monkeypatch):
    monkeypatch.setenv("TRACING", "otlp")
    with span("answer"):
        pass
    assert len(exported.get_finished_spans()) == 1
    assert not (tmp_path / "traces.jsonl").exists()


def test_tracing_off_sends_nothing(exported, monkeypatch):
    monkeypatch.setenv("TRACING", "off")
    with span("answer"):
        pass
    assert exported.get_finished_spans() == ()


def test_langfuse_keys_build_the_endpoint_and_basic_auth(monkeypatch):
    monkeypatch.chdir("/")  # no .env files here
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:3000/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")
    endpoint, headers = otel.langfuse_target()
    assert endpoint == "http://127.0.0.1:3000/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic cGstbGYtMTpzay1sZi0y"  # pk-lf-1:sk-lf-2
    assert otel.exporter_from_env() is not None


def test_no_keys_means_no_exporter(monkeypatch):
    monkeypatch.chdir("/")
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert otel.exporter_from_env() is None


def test_a_span_that_only_names_a_model_stays_a_span(exported):
    with span("answer") as s:
        s.attributes["model"] = "llama3.2:3b"  # recorded for filtering, but no LLM call here
    a = exported.get_finished_spans()[0].attributes
    assert a["langfuse.observation.type"] == "span"
    assert "gen_ai.request.model" not in a
    assert a["langfuse.observation.metadata.model"] == "llama3.2:3b"
