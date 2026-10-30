from tracing.view import percentile, stats


def span(name, trace, ms, status="ok", **attributes):
    return {
        "name": name,
        "trace_id": trace,
        "duration_ms": ms,
        "status": status,
        "attributes": attributes,
    }


def test_percentile_uses_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0


def test_stats_per_step_and_per_tenant():
    traces = {
        "a": [
            span("answer", "a", 100, tenant_id="pwani"),
            span(
                "generate",
                "a",
                90,
                tokens_in=100,
                tokens_out=10,
                cost_usd=0.0,
                cost_usd_hosted=0.00046,
            ),
        ],
        "b": [
            span("answer", "b", 300, "error", tenant_id="ruaka"),
            span("generate", "b", 280, "error"),
        ],
    }
    lines = stats(traces)
    assert lines[1].split() == ["answer", "2", "1", "100.0", "300.0"]  # slowest step first
    assert lines[2].split() == ["generate", "2", "1", "90.0", "280.0"]
    assert lines[-1].split() == ["pwani", "100", "10", "0.00", "0.0005"]  # ruaka used no tokens
