# 02 · Failure Forensics (LLM Tracing)

**Goal:** trace every step of an LLM pipeline and be able to answer *"why did this answer go wrong?"* in under a minute.
**Time:** 2–3 weeks. Instruments project 01.

## What you'll learn
- Traces, spans and attributes (OpenTelemetry concepts)
- What to log for LLM calls: prompt, output, model, tokens, cost, latency, tenant
- Self-hosting an observability tool (Langfuse or Arize Phoenix)
- Building a failure taxonomy from real bad outputs
- Redacting personal data (PII) before it hits your logs

## Milestones

- [ ] **M1 · Hand-rolled tracer.** Before using any tool, write a `@traced` decorator that records span name, parent, start/end time, inputs and outputs to a JSONL file. ~50 lines. This is how you'll understand what the tools do.
- [ ] **M2 · Instrument project 01.** Spans for `embed_query`, `vector_search`, `bm25_search`, `fuse`, `rerank`, `generate`. Attributes: `tenant_id`, `model`, `tokens_in/out`, `cost_usd`, `latency_ms`.
- [ ] **M3 · Real tool.** Self-host Langfuse (docker compose) or Phoenix and send the same traces there via its SDK or OpenTelemetry.
- [ ] **M4 · Failure taxonomy.** Run the 01 eval set and collect at least 30 bad answers. Label each with a root cause:
  `retrieval_miss` · `rerank_drop` · `hallucination` · `ignored_context` · `format_error` · `wrong_refusal` · `timeout`
- [ ] **M5 · Forensics tool.** Given a trace id, automatically suggest the likely cause using checks:
  Was the gold chunk retrieved? Did it survive reranking? Is the answer supported by the context? Optionally add an LLM classifier for the hard cases.
  *Deliverable:* `uv run forensics <trace_id>` → likely cause + evidence.
- [ ] **M6 · Dashboards.** Cost per tenant per day, p50/p95 latency per step, error rate, and failure category counts.
- [ ] **M7 · PII redaction.** Mask emails, phone numbers and ID numbers before traces are stored. Add a test for it.

## Done when
- You can click (or query) a bad answer and see the failing step and its category
- The dashboard shows cost per tenant
- README includes a "top failure causes" chart from your own data and what you changed because of it

## Resources
See [docs/RESOURCES.md → Tracing](../docs/RESOURCES.md#02--llm-tracing).
