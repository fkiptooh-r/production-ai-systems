# shared/

Reusable building blocks that more than one project needs. Code moves here only once a **second** project needs it, so don't abstract early.

Expected to grow into:

| Module | Born in | Reused by |
|--------|---------|-----------|
| `llm_client` (one interface over providers, retries, token/cost counting) | 00 | all |
| `config` (loads `.env`, typed settings) | 00 | all |
| `tracing` (span decorator / OpenTelemetry setup) | 02 | 01, 03, 04, 05 |
| `retriever` (tenant-aware hybrid search) | 01 | 05 |
| `judge` (rubric-based scoring) | 03 | 04, 05 |
| `router` (model selection + budgets) | 04 | 05 |
