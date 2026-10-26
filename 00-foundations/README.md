# 00 · Foundations

**Goal:** be able to build, test, containerise and deploy a small LLM-backed API on your own.
**Time:** 4–8 weeks (skip milestones you can already do; prove it with the deliverable instead).

## What you'll learn
- Modern Python tooling: `uv`, virtual environments, `ruff`, `pytest`, type hints, `async`
- Calling LLM APIs: streaming, structured output, tool calling, retries, token and cost counting
- Building APIs with FastAPI, Postgres basics, Docker, and deploying to the cloud
- CI with GitHub Actions

## Milestones

- [x] **M1 · Tooling.** Install [`uv`](https://docs.astral.sh/uv/). Create a project, add `ruff` + `pytest`, and write one function with one test.
  *Deliverable:* `uv run pytest` passes.
- [x] **M2 · First LLM call.** A script that calls one provider (or a local model via Ollama). Add streaming, a timeout, retries with backoff, and print tokens used plus the estimated cost.
  *Deliverable:* `uv run chat "hello"` streams a reply and prints the cost.
- [x] **M3 · Structured output.** Ask the model to extract fields (e.g. from a house listing text: price, bedrooms, location) into a Pydantic model. Handle invalid JSON.
  *Deliverable:* 10 sample texts → 10 validated objects, with tests.
- [x] **M4 · Tool calling.** Give the model a tool such as `search_listings(max_price, bedrooms)` that reads a local JSON file. Run the model → tool → model loop by hand, without a framework.
  *Deliverable:* "find me a 2-bedroom under 40k" returns real results from your JSON.
- [x] **M5 · FastAPI service.** `POST /chat` with streaming (Server-Sent Events), `GET /health`, request validation, and tests using `TestClient`.
- [x] **M6 · Postgres.** Run Postgres in Docker. Store conversations and messages. Write the SQL by hand at least once (CREATE TABLE, INSERT, SELECT with JOIN).
- [ ] **M7 · Docker + deploy.** `Dockerfile` + `docker-compose.yml` (api + postgres). Deploy to Render, Fly.io or Railway, with secrets set as environment variables on the host.
- [ ] **M8 · CI.** A GitHub Actions workflow that runs `ruff` and `pytest` on every push.
- [ ] **M9 · Extract to `shared/`.** Move the LLM client and config loader into `shared/` so project 01 can reuse them.

## How to run

Requires [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) and
[Docker Desktop](https://www.docker.com/products/docker-desktop/) running locally.

```bash
ollama pull llama3.2:3b          # one-time model downloads (~2 GB + ~1 GB)
ollama pull qwen2.5:1.5b
docker compose up -d             # Postgres on 127.0.0.1:5433 (5432 may be taken by a local install)
uv sync                          # install dependencies
uv run pytest                    # tests; DB tests are skipped if Postgres isn't running
uv run chat "What is an embedding?"
uv run chat "Hi" --model qwen2.5:1.5b --retries 5
uv run extract-eval              # score listing extraction (~2 min on CPU)
uv run agent "Find me a 2-bedroom under 40k with parking"
uv run tool-eval --model qwen2.5:1.5b   # score tool choice + arguments (~1 min)
uv run serve                     # API at http://localhost:8000, try it at /docs
```

### API

| Endpoint | Does | Notes |
|----------|------|-------|
| `GET /health` | Service, Ollama and database status | 503 when either is unreachable |
| `POST /chat` | `{"prompt", "conversation_id"?}` → streamed reply | SSE: `token`… then `done` (with `conversation_id`) or `error`. Send the id back to continue the conversation |
| `GET /conversations` | Recent conversations | Message count + first question |
| `GET /conversations/{id}` | Full message history | 404 if unknown |
| `POST /agent` | `{"question"}` → answer + tool steps | Defaults to qwen2.5:1.5b (won the M4 eval) |
| `POST /extract` | `{"text"}` → validated `Listing` | |

Model failures return **502** (the upstream model failed, not the caller); invalid requests
return **422** before the model is ever called; a database outage returns **503** without
leaking SQL details.

### Storage

Hand-written SQL in [`sql/schema.sql`](sql/schema.sql) and [`db.py`](src/foundations/db.py):
`conversations` 1 → many `messages`. A question and its answer are saved in one transaction
(both or neither), and only the last 20 messages are sent back to the model. Every value is
a query parameter, never pasted into SQL; a test stores `'); DROP TABLE messages; --` as plain text.

**Windows note:** connection URLs use `127.0.0.1`, not `localhost`. On Windows, `localhost`
tries IPv6 first; the container and Ollama only listen on IPv4, so each connection waited for
the IPv6 attempt to time out. Switching took the test suite from 80s to 5s.

```bash
curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt":"hi"}'
```

Example output:
```
Retrieval-Augmented Generation (RAG) is a type of ...

---
tokens: 44 in, 85 out | 15.1s (5.6 tokens/s)
cost: $0 locally, ~$0.001407 at $3.0/$15.0 per 1M tokens
```

## Results

### M3 · Listing extraction (`uv run extract-eval`, llama3.2:3b on CPU, 10 hand-labelled listings)

| Field | v1 | v2 |
|-------|----|----|
| valid output | 10/10 | 10/10 |
| property_type | 90% | 90% |
| bedrooms | 10% | **100%** |
| price_per_month | 100% | 100% |
| location | 60% | 90% |
| parking | 80% | 90% |
| furnished | 80% | 90% |
| **overall** | **70%** | **93%** |

**v1 → v2:** the schema-constrained model was *omitting* keys rather than answering `null`.
Pydantic fields with `= None` defaults are left out of the JSON schema's `required` list, so the
constraint let the model skip them and Pydantic silently filled in `None`. Making every field
required-but-nullable fixed it; a regression test guards it.

**Remaining misses:** negatives read as "not mentioned" ("No parking" → `null`, not `false`),
"bungalow" classified as bedsitter, and "Githurai 45" shortened to "Githurai".

### M4 · Tool calling (`uv run tool-eval`, first decision on 10 questions, 2 of which need no tool)

| Metric | llama3.2:3b v1 | llama3.2:3b v2 | qwen2.5:1.5b v2 |
|--------|----------------|----------------|-----------------|
| right tool (or correctly none) | 80% | 80% | **100%** |
| right tool + right arguments | 70% | 80% | **90%** |
| called a tool when none was needed | 2/2 | 2/2 | **0/2** |

**v1 → v2 (system fix):** llama sent the *string* `"null"` for unwanted filters, which would
search for a neighbourhood called "null". A shared `ToolArgs` validator now treats
`"null"` / `"none"` / `""` as unset; regression tests guard it.

**Model limits (measured, not patched):** llama calls a tool even for "Hello!". qwen, at half
the size, never over-called, but once sent `parking: false` for "with parking", the opposite
of the request. The model choice matters as much as the code.

## Done when
- A public URL that someone else can `curl` and get a streamed answer from
- CI is green
- The README explains how to run it locally in under 5 commands

## Suggested structure
```
00-foundations/
├── pyproject.toml
├── src/foundations/
│   ├── llm.py        # client, retries, cost tracking
│   ├── extract.py    # structured output
│   ├── tools.py      # tool calling loop
│   └── api.py        # FastAPI app
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Common mistakes
- Hard-coding API keys "just for testing" (that's how they leak)
- Skipping tests because "it's just a script"
- Building with a framework (LangChain etc.) before understanding the raw API loop

## Resources
See [docs/RESOURCES.md → Foundations](../docs/RESOURCES.md#00--foundations).
