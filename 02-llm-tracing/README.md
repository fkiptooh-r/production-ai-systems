# 02 · Failure Forensics (LLM Tracing)

**Goal:** trace every step of an LLM pipeline and be able to answer *"why did this answer go wrong?"* in under a minute.
**Time:** 2–3 weeks. Instruments project 01.
**Status:** 🟨 In progress · M1–M4 done · tracked in the [02 milestone](https://github.com/fkiptooh-r/production-ai-systems/milestone/3) (#30–#36)

## What you'll learn
- Traces, spans and attributes (OpenTelemetry concepts)
- What to log for LLM calls: prompt, output, model, tokens, cost, latency, tenant
- Self-hosting an observability tool (Langfuse or Arize Phoenix)
- Building a failure taxonomy from real bad outputs
- Redacting personal data (PII) before it hits your logs

## Milestones

- [x] **M1 · Hand-rolled tracer.** Before using any tool, write a `@traced` decorator that records span name, parent, start/end time, inputs and outputs to a JSONL file. ~50 lines. This is how you'll understand what the tools do.
- [x] **M2 · Instrument project 01.** Spans for `embed_query`, `vector_search`, `keyword_search`, `fuse`, `rerank`, `generate`. Attributes: `tenant_id`, `model`, `tokens_in/out`, `cost_usd`, `latency_ms`.
- [x] **M3 · Real tool.** Self-host Langfuse (docker compose) or Phoenix and send the same traces there via its SDK or OpenTelemetry.
- [x] **M4 · Failure taxonomy.** Run the 01 eval set and collect at least 30 bad answers. Label each with a root cause:
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

**Langfuse (M3).** Self-hosted, six containers, ~2.4 GB of RAM while running.

```bash
cd 02-llm-tracing
uv run langfuse-setup                 # once: random secrets -> langfuse/.env, keys -> ../.env
cd langfuse && docker compose up -d   # first start pulls ~1-2 GB and migrates: 3-5 min
curl http://127.0.0.1:3000/api/public/health        # {"status":"OK","version":"4.x"}
```

Open http://localhost:3000 and log in as `admin@langfuse.local` with
`LANGFUSE_ADMIN_PASSWORD` from `langfuse/.env` (both `.env` files are gitignored). With the
keys in the repo-root `.env`, every traced command (the demo, `ask`, `serve`,
`eval-answers`) also sends its spans to Langfuse; the trace id is the same in both places.
`TRACING=otlp` sends to Langfuse only, without the JSONL file. `docker compose stop` frees the
RAM; the data stays in Docker volumes.

The exporter speaks plain OpenTelemetry (OTLP over HTTP), so another backend is a config
change: unset the `LANGFUSE_*` keys and set `OTEL_EXPORTER_OTLP_ENDPOINT` (for example Arize
Phoenix at `http://127.0.0.1:6006`).

**Failure taxonomy (M4).** Needs Postgres + Ollama for the runs only; labelling is offline.

```bash
cd 01-hybrid-rag      # 32 harder questions, three setups, ~20 min on CPU
export TRACE_FILE=../02-llm-tracing/traces/m4.jsonl
uv run eval-answers --questions eval/hard_questions.jsonl     --out ../02-llm-tracing/data/runs/hard-llama-vector.jsonl
uv run eval-answers --questions eval/hard_questions.jsonl --model qwen2.5:1.5b     --out ../02-llm-tracing/data/runs/hard-qwen-vector.jsonl
uv run eval-answers --questions eval/hard_questions.jsonl --searcher hybrid+rerank     --out ../02-llm-tracing/data/runs/hard-llama-rerank.jsonl
cd ../02-llm-tracing
uv run failure-candidates   # evidence per answer from runs + traces -> data/candidates.jsonl
# read every answer, write data/failures.jsonl (label + one-line note per bad answer)
uv run taxonomy             # counts and chart
```

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

**The first real run caught a silent failure (#62), and I misdiagnosed it at first.**
Answers fell from 20/20 to **6/20**, mostly false "I don't know". The traces showed `generate`
was fine but `vector_search` returned unrelated chunks (a deposit question got the Kilimani
area guide first); re-measured, vector search MRR was **0.99 → 0.09**. The stored vectors no
longer matched what Ollama produces for the same text (cosine ≈ 0.0).

- **First explanation (wrong):** Ollama had auto-updated after the chunks were embedded, so
  the embedding model drifted. Re-embedding (`uv run ingest`) fixed it, which seemed to
  confirm it.
- **What actually happened:** the next day the same failure came back with *exactly* the
  same numbers and no Ollama change. Both times I had just run lab 01's tests. The database
  tests used the dev database: `test_vector.py` replaces every chunk and embeds them with a
  fake word-hash embedder. Cosine between the stored vectors and that fake embedder: **1.0000
  for all 64 chunks**.
- **Fix (#62):** the tests now create and use their own database, `ai_systems_test`, and
  refuse to run against the dev one. Proof: a full run with Postgres up (170 tests) leaves the
  dev database's embedding fingerprint unchanged, and vector MRR stays 0.99.

Lessons: a fix that works doesn't prove the diagnosis, so reproduce before explaining; and
tests that pass can still be the cause. Only measuring the real system, and tracing it,
caught this at all.

**Two things M2 depended on M1 for:** answers report their `trace_id` (CLI, eval results,
web `done` event), and the web worker thread runs in a copy of the caller's context, so a
request's spans stay one trace. Tests cover both, plus a failing model call recorded as an
error on `generate`, `answer` and `chat_request` with the trace id in the error event.

### M3 · Langfuse over OpenTelemetry

Our tracer stays the API. When an exporter is configured, each `span()` also opens an
OpenTelemetry span **and takes its ids from it**, so `trace-view 90636cb9` and the Langfuse
trace `90636cb9…` are the same request. Finished spans go to OpenTelemetry's
`BatchSpanProcessor`: a background thread sends them in batches over OTLP/HTTP to
`/api/public/otel/v1/traces`. Attributes are renamed to what Langfuse reads:

| Ours | Sent as |
|------|---------|
| `inputs`, `output` | `langfuse.observation.input`, `.output` (JSON) |
| `tokens_in`, `tokens_out` | `gen_ai.usage.input_tokens`, `.output_tokens`, and the span becomes a **generation** |
| `model` | `gen_ai.request.model` on generations, metadata elsewhere |
| `cost_usd` | `langfuse.observation.cost_details` |
| `tenant_id` | trace metadata, observation metadata and a `tenant:<id>` tag, on **every** span |
| error | span status ERROR, `langfuse.observation.level` ERROR, status message |
| anything else (`retrieved`, `cited`, `cost_usd_hosted`…) | `langfuse.observation.metadata.<key>` |

A lab 01 question, read back from Langfuse's API next to the JSONL tree. Same ids, same
nesting, same timings; token usage only on the LLM call:

```
answer         SPAN         10.73 s
vector_search  SPAN          0.49 s
embed_query    SPAN          0.46 s
generate       GENERATION   10.24 s   usage: input 268, output 25
```

**Overhead per traced call** (`uv run trace-demo`, no-op function, 5,000 calls, 3 runs):

| Exporter | per call |
|----------|---------:|
| plain function | 0.1–0.2 µs |
| JSONL file, synchronous (M1) | 690–1,370 µs |
| **OpenTelemetry, batch export (M3)** | **107–168 µs** |

**About 6–8× cheaper**, because the request thread only builds the span and queues it. What's
left (~0.15 ms) is creating the OTel span and serialising inputs and outputs to JSON.
Against a 10 s answer it is invisible.

**When Langfuse is down** (web container stopped): the demo finished normally, request
timings didn't change (~215 ms each) and the JSONL file still got every span. The exporter
retried, logged `Failed to export spans batch`, and dropped that batch; the only cost was
~2 s extra at program exit (10.6 s vs 8.8 s). Observability must never take the app down,
and here it doesn't. The trade-off: spans produced while the backend is down are lost from
Langfuse (the JSONL copy keeps them).

**Two mistakes found by reading the data back, not by tests passing:**
- **`answer` showed up as a generation.** When a span has a model name and no explicit type,
  Langfuse infers "generation", which would double-count LLM calls in cost and usage views.
  Every span now sends its type explicitly; a test pins it.
- **Langfuse v4 retired the `/api/public/traces` read API** ("events_only" mode); reads go
  through `/api/public/v2/observations`. The OTLP ingestion path is unchanged.

**Self-hosting notes.** The compose file is the official one, slimmed: no in-app agent or AI
features, only the UI (3000) and the media store (9090) published, both on 127.0.0.1;
`restart: unless-stopped` rather than `always`, so 2.4 GB doesn't come back on every boot; no
default passwords (Compose refuses to start until `langfuse-setup` has written them); headless
init creates the org, project, API keys and admin user, so there's no sign-up page.

### M4 · Failure taxonomy (96 answers, 43 bad, every one read by hand)

**Getting enough failures.** Lab 01's own eval produces ~5 bad answers, too few to learn
from. So I wrote **32 harder questions** (`01-hybrid-rag/eval/hard_questions.jsonl`, checked
by lab 01's eval-set tests like the original 60) aimed at how answers go wrong rather than
at retrieval: arithmetic (*"rent is KES 50,000, what do I pay before the keys?"*),
conditions (*"a dog on the third floor?"*), negations, another agency's rule (*"the M-Pesa
paybill?"* at an agency that only takes bank transfers), false premises (*"when is the
application fee refunded?"* where there is none) and not-in-data questions. Three setups ran
them: llama3.2:3b with vector search (lab 01 as shipped), qwen2.5:1.5b, and llama3.2:3b with
hybrid search + reranking.

**Method.** `failure-candidates` joins each answer with its trace: the gold chunk ids, whether
first-stage search found them, whether they survived reranking, whether they were cited, and
any error, plus a suggested label from that evidence alone. Then every answer was read and
labelled by hand with a one-line reason (`data/failures.jsonl`). Root cause = the first stage
of the pipeline that went wrong.

```
43 bad answers out of 96

ignored_context   15  ██████████████████████████████
format_error      15  ██████████████████████████████
wrong_refusal     10  ████████████████████
hallucination      3  ██████
retrieval_miss     0
rerank_drop        0
timeout            0
```

| Setup | Bad answers | Main cause |
|-------|------------:|------------|
| llama3.2:3b + vector (lab 01) | 10 / 32 | ignored_context 4, wrong_refusal 3 |
| llama3.2:3b + hybrid + rerank | 11 / 32 | ignored_context 5, wrong_refusal 3 |
| qwen2.5:1.5b + vector | **22 / 32** | format_error 11 (no citation) |

By question kind: conditional 14, numeric 11, cross-tenant 7, false premise 6, negation 4,
not-in-data 1.

**What the data says**
- **Retrieval is not the problem.** The chunk holding the answer was retrieved for **84 of 84**
  answerable questions in every setup, and the reranker kept it every time (28/28). All 43
  failures happened in generation. More retrieval work (hybrid, reranking, the A1 rewriting
  ideas) would not have fixed a single one of them, which matches M2's latency finding.
- **`ignored_context`: the right text was in front of the model and it got it wrong.** It
  counted 7 late days from the 1st, skipping a grace period to the 5th (KES 3,500 instead of
  1,500), *in both llama setups*; answered "Yes, you pay for water" while quoting the line
  that says it's included; said the 8am drive is "outside rush hour" from a chunk saying
  7–9am is the worst. Conditions and arithmetic are where a 3B model breaks.
- **`wrong_refusal` on false premises.** Asked *"when is the application fee refunded?"*,
  every setup said "I don't know", although the retrieved chunk says there is no fee. The
  prompt only offers two ways out, answer or "I don't know", so a premise the sources
  contradict lands in the wrong one.
- **`hallucination` is rare (3) but the worst kind:** both llama setups invented a *daily*
  late fee (10% of rent ÷ 30) because the question assumed one.
- **qwen2.5:1.5b mostly skips citations** (11 correct answers without one, and one answer that
  was only `[1]`). At half the size it's fast (p50 5.2 s vs 14.7 s per answer) but doesn't follow the
  output contract; 13 of the 15 `format_error`s were otherwise correct.

**What the automated checks missed.** Lab 01's answer eval passes an answer that cites a chunk
containing the expected phrase. **11 of the 43 bad answers passed it**, every one of them
because it cited the right chunk and then said something wrong about it. And the
evidence-only suggestion matched the hand label for **23 of 43**: the trace can show *where*
the right text was, never *what the model did with it*. That's the case for an LLM judge
that checks the claim against the cited text (project 03, and A5 in lab 01, #29), and for M5's
forensics tool needing more than retrieval checks.

**What I'd change because of it (each needs its own measurement):** let the prompt correct a
premise the sources contradict, instead of only "answer or I don't know"; spell out
conditions and arithmetic in the answer before the final figure; keep llama3.2:3b over
qwen2.5:1.5b for answering, since speed doesn't pay for a broken citation contract.

## Done when
- You can click (or query) a bad answer and see the failing step and its category
- The dashboard shows cost per tenant
- README includes a "top failure causes" chart from your own data and what you changed because of it

## Resources
See [docs/RESOURCES.md → Tracing](../docs/RESOURCES.md#02--llm-tracing).
