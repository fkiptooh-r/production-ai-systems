# 02 · Failure Forensics (LLM Tracing)

**Goal:** trace every step of an LLM pipeline and be able to answer *"why did this answer go wrong?"* in under a minute.
**Time:** 2–3 weeks. Instruments project 01.
**Status:** 🟨 In progress · M1 done · tracked in the [02 milestone](https://github.com/fkiptooh-r/production-ai-systems/milestone/3) (#30–#36)

## What you'll learn
- Traces, spans and attributes (OpenTelemetry concepts)
- What to log for LLM calls: prompt, output, model, tokens, cost, latency, tenant
- Self-hosting an observability tool (Langfuse or Arize Phoenix)
- Building a failure taxonomy from real bad outputs
- Redacting personal data (PII) before it hits your logs

## Milestones

- [x] **M1 · Hand-rolled tracer.** Before using any tool, write a `@traced` decorator that records span name, parent, start/end time, inputs and outputs to a JSONL file. ~50 lines. This is how you'll understand what the tools do.
- [ ] **M2 · Instrument project 01.** Spans for `embed_query`, `vector_search`, `bm25_search`, `fuse`, `rerank`, `generate`. Attributes: `tenant_id`, `model`, `tokens_in/out`, `cost_usd`, `latency_ms`.
- [ ] **M3 · Real tool.** Self-host Langfuse (docker compose) or Phoenix and send the same traces there via its SDK or OpenTelemetry.
- [ ] **M4 · Failure taxonomy.** Run the 01 eval set and collect at least 30 bad answers. Label each with a root cause:
  `retrieval_miss` · `rerank_drop` · `hallucination` · `ignored_context` · `format_error` · `wrong_refusal` · `timeout`
- [ ] **M5 · Forensics tool.** Given a trace id, automatically suggest the likely cause using checks:
  Was the gold chunk retrieved? Did it survive reranking? Is the answer supported by the context? Optionally add an LLM classifier for the hard cases.
  *Deliverable:* `uv run forensics <trace_id>` → likely cause + evidence.
- [ ] **M6 · Dashboards.** Cost per tenant per day, p50/p95 latency per step, error rate, and failure category counts.
- [ ] **M7 · PII redaction.** Mask emails, phone numbers and ID numbers before traces are stored. Add a test for it.

## How to run

No services needed for M1 (the demo pipeline sleeps instead of calling Ollama or Postgres).

```bash
cd 02-llm-tracing
uv sync
uv run trace-demo              # 3 traces of a fake RAG request (one times out) + overhead
uv run trace-view --list       # one line per trace
uv run trace-view 3e5f         # one trace as a tree (a unique id prefix is enough)
uv run pytest
```

Traces go to `traces/traces.jsonl` (gitignored); set `TRACE_FILE` to write elsewhere.

```python
from tracing.tracer import set_attribute, traced

@traced                         # span named after the function
def vector_search(vector, tenant_id, k=5):
    set_attribute("tenant_id", tenant_id)
    ...
```

## Results

### M1 · Hand-rolled tracer (`uv run trace-demo`)

One JSON line per span: `trace_id`, `span_id`, `parent_id`, `name`, `start`, `duration_ms`,
`status`, `error`, `inputs`, `output`, `attributes`. The failing request, rebuilt as a tree:

```
answer                                           124.0 ms  ✗ TimeoutError: model did not answer in time  tenant_id=pwani
├── embed_query                                   40.7 ms
├── vector_search                                 22.4 ms  tenant_id=pwani
└── generate                                      50.4 ms  ✗ TimeoutError: model did not answer in time  model=llama3.2:3b
```

The error is recorded on `generate`, where it happened, *and* on `answer`, which it passed
through. That is how you find the failing step: the deepest span with an error.

**Overhead** (Windows laptop, no-op function, 5,000 calls):

| | per call |
|---|---|
| plain function | ~0.3 µs |
| `@traced`, total | **~1.2–1.6 ms** |
| … building the span + JSON | ~41 µs |
| … opening, appending to and closing the file | **~844 µs** |
| … writing to an already-open file instead | ~1 µs |

- **The file open is ~95% of the cost.** Around 1.5 ms × 6 spans is ~9 ms on a request that
  takes 0.5–10 s, so 0.1–2%: fine for learning, wasteful in production. Real SDKs
  (OpenTelemetry, Langfuse) put finished spans on a queue and a background thread exports
  them in batches; the request never waits for I/O. M3 uses one.
- **Children are written before their parent**, because a span is written when it *ends*.
  The viewer rebuilds the tree from `parent_id`, which is also how the real tools do it.
- **Context doesn't cross threads by itself.** The current span lives in a `ContextVar`; a new
  thread starts with an empty context, so its spans would begin a new, orphaned trace. Lab
  01's streaming endpoint answers in a worker thread, so M2 must copy the context
  (`contextvars.copy_context().run(...)`). A test pins down both behaviours.

## Done when
- You can click (or query) a bad answer and see the failing step and its category
- The dashboard shows cost per tenant
- README includes a "top failure causes" chart from your own data and what you changed because of it

## Resources
See [docs/RESOURCES.md → Tracing](../docs/RESOURCES.md#02--llm-tracing).
