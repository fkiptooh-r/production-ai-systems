# 04 · Cost Autopilot (Model Routing)

**Goal:** cut LLM spend as far as possible while keeping quality within a target, measured with the judge from project 03.
**Time:** 2–3 weeks.

## What you'll learn
- The cost / quality / latency trade-off across model tiers
- Building an LLM gateway: one interface, retries, timeouts, provider fallback
- Rule-based vs. learned routing
- Caching: exact, semantic, and provider prompt caching
- Per-tenant budgets and graceful degradation

## Milestones

- [ ] **M1 · Baseline.** Run the 01 eval set through a cheap, a mid and a strong model (cloud and/or local via Ollama). Record quality (03 judge), cost and latency for each. Plot it.
- [ ] **M2 · Gateway.** One `complete()` interface over providers (build your own, then compare with LiteLLM): retries, timeouts, and fallback to a second provider when the first fails.
- [ ] **M3 · Rule router.** Route by simple signals: query length, intent keywords, whether retrieval found strong matches, the tenant's plan.
- [ ] **M4 · Learned router.** Label which eval questions the cheap model already passes. Train a small classifier (logistic regression on embeddings is enough) to predict "cheap model is fine". Compare with M3.
- [ ] **M5 · Caching.** Exact-match cache, then a semantic cache (embedding similarity threshold). **Measure wrong cache hits**: a semantic cache that returns the wrong answer is worse than no cache. Use provider prompt caching for long system prompts.
- [ ] **M6 · Budgets.** Per-tenant monthly caps with soft limit (downgrade to cheaper models + alert) and hard limit (block or queue). Tests for both.
- [ ] **M7 · Report.** A cost vs. quality curve for every strategy, and the operating point you chose with a one-paragraph justification.

## Results to report

| Strategy | Cost / 1k queries | Judge pass rate | p95 latency |
|----------|-------------------|-----------------|-------------|
| Always strong | | | |
| Always cheap | | | |
| Rule router | | | |
| Learned router | | | |
| Learned router + cache | | | |

## Done when
- A headline you can defend: "**X% cheaper with a Y-point quality drop**", backed by the table
- Budget limits are enforced and tested

## Where this goes in your products
- **E-commerce:** AI tiers per store plan (free stores get cheaper models), per-store spending caps
- **House hunting:** cheap model for filtering, strong model for detailed neighbourhood comparisons
- **EdTech:** keep AI tutoring affordable at student prices

## Resources
See [docs/RESOURCES.md → Model Routing](../docs/RESOURCES.md#04--model-routing--cost).
