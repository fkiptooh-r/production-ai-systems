import json

from tracing.taxonomy import candidates, evidence, first_stage, suggest, summary


def record(**overrides):
    base = {
        "id": "rr-h01",
        "kind": "numeric",
        "tenant": "ruaka-rentals",
        "question": "Late fee on the 8th?",
        "expected": "KES 500 per day",
        "answer": "KES 1,500 [1].",
        "gold": ["ruaka-rentals/tenancy-policy.md#3"],
        "retrieved": ["ruaka-rentals/tenancy-policy.md#3", "ruaka-rentals/maintenance.md#1"],
        "cited": ["ruaka-rentals/tenancy-policy.md#3"],
        "abstained": False,
        "passed": True,
        "trace_id": "t1",
    }
    return base | overrides


def span(name, span_id, parent_id=None, output=None, status="ok", error=None):
    return {
        "name": name,
        "span_id": span_id,
        "parent_id": parent_id,
        "output": output,
        "status": status,
        "error": error,
        "trace_id": "t1",
    }


RERANK_TRACE = [
    span("hybrid_search", "h", "r", output=["gold#1", "other#2", "other#3"]),
    span("rerank", "k", "r", output=["other#2"]),
    span("rerank_search", "r", "a", output=["other#2"]),
]


def test_first_stage_is_the_pool_the_reranker_saw():
    assert first_stage(RERANK_TRACE) == ["gold#1", "other#2", "other#3"]
    assert first_stage([span("vector_search", "v")]) is None


def test_gold_found_then_cut_by_the_reranker_is_a_rerank_drop():
    r = record(gold=["gold#1"], retrieved=["other#2"], cited=["other#2"])
    ev = evidence(r, RERANK_TRACE)
    assert ev["gold_in_first_stage"] and not ev["gold_retrieved"]
    assert suggest(ev) == "rerank_drop"


def test_gold_never_found_is_a_retrieval_miss():
    r = record(gold=["gold#9"], retrieved=["other#2"], cited=[])
    assert suggest(evidence(r, RERANK_TRACE)) == "retrieval_miss"
    assert suggest(evidence(r, [])) == "retrieval_miss"  # no reranker in this run


def test_pipeline_order_decides_the_label():
    refused = record(abstained=True, cited=[], answer="I don't know.")
    assert suggest(evidence(refused, [])) == "wrong_refusal"
    uncited = record(cited=[])
    assert suggest(evidence(uncited, [])) == "format_error"
    wrong_chunk = record(cited=["ruaka-rentals/maintenance.md#1"])
    assert suggest(evidence(wrong_chunk, [])) == "ignored_context"
    invented = record(expected=None, gold=[], cited=["x#1"])
    assert suggest(evidence(invented, [])) == "hallucination"
    errored = [span("generate", "g", status="error", error="LLMError: timed out")]
    assert suggest(evidence(record(), errored)) == "timeout"


def test_a_cited_grounded_answer_needs_reading():
    assert suggest(evidence(record(), [])) is None  # "KES 1,500" may still be wrong maths
    assert suggest(evidence(record(expected=None, gold=[], abstained=True), [])) is None


def test_candidates_join_runs_and_traces(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "hard-llama-vector.jsonl").write_text(json.dumps(record()) + "\n", "utf-8")
    traces = tmp_path / "traces.jsonl"
    traces.write_text("", "utf-8")
    [row] = candidates(runs, traces)
    assert row["run"] == "hard-llama-vector"
    assert row["suggested"] is None and row["evidence"]["gold_cited"] is True


def test_summary_counts_and_draws_bars():
    failures = [
        {"label": "retrieval_miss", "run": "a", "kind": "numeric"},
        {"label": "retrieval_miss", "run": "b", "kind": "negation"},
        {"label": "hallucination", "run": "a", "kind": "numeric"},
    ]
    lines = summary(failures, total=10)
    assert lines[0] == "3 bad answers out of 10"
    assert lines[2].split()[:2] == ["retrieval_miss", "2"]
    assert lines[3].split()[:2] == ["hallucination", "1"]
    assert lines[-1] == "by question kind: numeric 2, negation 1"
