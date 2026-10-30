"""Exercise the tracer on a fake RAG pipeline, then measure what tracing costs.

The steps sleep instead of calling Ollama or Postgres, so this runs anywhere in a second.
M2 puts the same decorator on lab 01's real functions.

    uv run trace-demo      # writes 3 traces (one fails), shows the last, prints overhead
"""

import os
import sys
import tempfile
import time

from tracing import tracer
from tracing.tracer import set_attribute, traced
from tracing.view import load, render, summary

CHUNKS = ["Rent is due on the 5th [1]", "Deposit is two months [2]", "Pets need approval [3]"]


@traced
def embed_query(question: str) -> list[float]:
    time.sleep(0.04)
    return [0.1, 0.2, 0.3]


@traced
def vector_search(vector: list[float], tenant_id: str, k: int = 3) -> list[str]:
    set_attribute("tenant_id", tenant_id)
    time.sleep(0.02)
    return CHUNKS[:k]


@traced
def generate(question: str, chunks: list[str], model: str = "llama3.2:3b") -> str:
    set_attribute("model", model)
    if "timeout" in question:
        time.sleep(0.05)
        raise TimeoutError("model did not answer in time")
    time.sleep(0.15)
    set_attribute("tokens_in", 120)
    set_attribute("tokens_out", 18)
    return "Rent is due on the 5th of each month [1]."


@traced(name="answer")
def answer(question: str, tenant_id: str) -> str:
    set_attribute("tenant_id", tenant_id)
    chunks = vector_search(embed_query(question), tenant_id)
    return generate(question, chunks)


def overhead_us(calls: int = 5_000) -> tuple[float, float]:
    """Microseconds per call of a no-op function, without and with @traced."""

    def plain(x: int) -> int:
        return x

    wrapped = traced(plain)
    start = time.perf_counter()
    for i in range(calls):
        plain(i)
    bare = (time.perf_counter() - start) / calls * 1e6
    start = time.perf_counter()
    for i in range(calls):
        wrapped(i)
    with_trace = (time.perf_counter() - start) / calls * 1e6
    return bare, with_trace


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no tree glyphs
    path = tracer.trace_file()
    for question in ["When is rent due?", "What is the deposit?", "trigger a timeout"]:
        try:
            answer(question, tenant_id="pwani")
        except TimeoutError:
            pass  # recorded on the span; a real API would return a 504

    traces = load(path)
    print(f"Traces in {path}:")
    for tid, spans in list(traces.items())[-3:]:
        print("  " + summary(tid, spans))
    last = list(traces)[-1]
    print(f"\ntrace {last}")
    print("\n".join(render(traces[last])))

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TRACE_FILE"] = os.path.join(tmp, "bench.jsonl")
        bare, with_trace = overhead_us()
        del os.environ["TRACE_FILE"]
    print(
        f"\nOverhead: {bare:.2f} µs per plain call, {with_trace:.1f} µs per traced call "
        f"(+{with_trace - bare:.1f} µs, mostly the JSONL write)."
    )
