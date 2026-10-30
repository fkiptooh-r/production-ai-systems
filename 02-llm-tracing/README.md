# 02 · Failure Forensics (LLM Tracing)

**Goal:** trace every step of an LLM pipeline and be able to answer *"why did this answer go wrong?"* in under a minute.
**Time:** 2–3 weeks. Instruments project 01.
**Status:** 🟨 In progress · M1–M2 done · tracked in the [02 milestone](https://github.com/fkiptooh-r/production-ai-systems/milestone/3) (#30–#36)

## What you'll learn
- Traces, spans and attributes (OpenTelemetry concepts)
- What to log for LLM calls: prompt, output, model, tokens, cost, latency, tenant
- Self-hosting an observability tool (Langfuse or Arize Phoenix)
- Building a failure taxonomy from real bad outputs
- Redacting personal data (PII) before it hits your logs

## Milestones

- [x] **M1 · Hand-rolled tracer.** Before using any tool, write a `@traced` decorator that records span name, parent, start/end time, inputs and outputs to a JSONL file. ~50 lines. This is how you'll understand what the tools do.
- [x] **M2 · Instrument project 01.** Spans for `embed_query`, `vector_search`, `keyword_search`, `fuse`, `rerank`, `generate`. Attributes: `tenant_id`, `model`, `tokens_in/out`, `cost_usd`, `latency_ms`.
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

Traces go to `traces/traces.jsonl` (gitignored); set `TRACE_FILE` to write elsewhere, or
`TRACING=off` to keep the spans' ids and timing but write nothing.

**Tracing lab 01 (M2).** Lab 01 depends on this package and traces itself, so its normal
commands now write traces to `01-hybrid-rag/traces/`. Needs Postgres + Ollama, as in lab 01:

```bash
cd 01-hybrid-rag
uv run ask --tenant ruaka-rentals "Can I keep a cat?"   # last line ends: trace <id>
uv run eval-answers --limit 20                          # each answer's trace_id is saved
cd ../02-llm-tracing
uv run trace-view --file ../01-hybrid-rag/traces/traces.jsonl <id>
uv run trace-view --file ../01-hybrid-rag/traces/traces.jsonl --stats
```

The browser chat sends the `trace_id` in its final `done` (or `error`) event.

```python
from tracing.tracer import set_attribute, traced


@traced  # span named after the function
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

### M2 · Lab 01, traced (20 answered questions + 3 retrieval runs, 177 traces)

Spans: `answer` → `vector_search` → `embed_query`, then `generate`. Hybrid and rerank add
`hybrid_search`, `keyword_search`, `fuse`, `rerank_search` and `rerank`. The web chat wraps it
all in `chat_request`. Attributes: `tenant_id`, `model`, `tokens_in`, `tokens_out`, `cost_usd`,
`cost_usd_hosted`, and on `answer` the `retrieved` and `cited` chunk ids. Latency is each
span's `duration_ms`, so it isn't repeated as an attribute.

```
answer                                         27884.6 ms  tenant_id=nyumba-homes model=llama3.2:3b
│                                                          retrieved=[tenancy-policy.md#1, …] cited=[tenancy-policy.md#1]
├── vector_search                               1024.3 ms  tenant_id=nyumba-homes
│   └── embed_query                              970.7 ms
└── generate                                   26856.7 ms  tokens_in=375 tokens_out=27 cost_usd_hosted=0.00153
```

**Where the time goes** (`trace-view --stats`, CPU laptop):

| Step | calls | p50 | p95 |
|------|------:|----:|----:|
| `answer` (whole request) | 20 | 12.2 s | 18.3 s |
| `generate` (llama3.2:3b) | 20 | **11.4 s** | 17.5 s |
| `rerank_search` (hybrid + rerank) | 52 | 1.50 s | 1.66 s |
| `hybrid_search` | 103 | 908 ms | 1113 ms |
| `vector_search` | 177 | 835 ms | 1078 ms |
| `embed_query` | 177 | **779 ms** | 1021 ms |
| `rerank` (MiniLM, 20 candidates) | 52 | 549 ms | 683 ms |
| `keyword_search` | 103 | 44 ms | 64 ms |
| `fuse` (RRF) | 103 | 0.1 ms | 0.2 ms |

Counts include one warm-up call per run (tenant `none`).

- **Generation is 93% of an answer** (11.4 of 12.2 s). Speeding up retrieval can't help much;
  a smaller or faster model, or streaming (already on), is where latency lives.
- **Embedding the question is 93% of vector search** (779 of 835 ms); the pgvector query itself
  is ~55 ms. Caching query embeddings or a smaller embedder would cut retrieval almost entirely.
- **Hybrid adds keyword search (44 ms) and fusion (0.1 ms)**: nearly free, yet it didn't
  improve quality in lab 01. Reranking adds ~550 ms, also for no gain on this data.
- **Cost:** 20 answers used 7,362 input and 452 output tokens: $0 locally, **$0.029 at
  $3/$15 per 1M tokens**, i.e. ~$0.0014 per answer, ~$1.45 per 1,000 questions. Input is 94%
  of the tokens (the four chunks), so `TOP_K` is the main cost lever.

**The first real run caught a production failure: embedding drift (#62).** Answers fell from
20/20 to **6/20** passing, mostly false "I don't know". The traces showed `generate` was
fine but `vector_search` returned unrelated chunks (a deposit question got the Kilimani area
guide first). Re-measuring: vector search MRR **0.99 → 0.09**. Cause: Ollama auto-updated
the evening after the chunks were embedded, and the same text now embeds to a different
vector (cosine between stored and fresh ≈ 0.0). Re-embedding (`uv run ingest`) restored
MRR 0.99 and the answers (3/3 abstentions, 0/17 false refusals, 17/17 grounded). No test
failed and nothing crashed, which is the point: only measuring the real system caught it.
#62 adds a startup check so it can't happen silently again.

**Two things M2 depended on M1 for:** answers report their `trace_id` (CLI, eval results,
web `done` event), and the web worker thread runs in a copy of the caller's context, so a
request's spans stay one trace. Tests cover both, plus a failing model call recorded as an
error on `generate`, `answer` and `chat_request` with the trace id in the error event.

## Done when
- You can click (or query) a bad answer and see the failing step and its category
- The dashboard shows cost per tenant
- README includes a "top failure causes" chart from your own data and what you changed because of it

## Resources
See [docs/RESOURCES.md → Tracing](../docs/RESOURCES.md#02--llm-tracing).
