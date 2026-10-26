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
- [ ] **M3 · Structured output.** Ask the model to extract fields (e.g. from a house listing text: price, bedrooms, location) into a Pydantic model. Handle invalid JSON.
  *Deliverable:* 10 sample texts → 10 validated objects, with tests.
- [ ] **M4 · Tool calling.** Give the model a tool such as `search_listings(max_price, bedrooms)` that reads a local JSON file. Run the model → tool → model loop by hand, without a framework.
  *Deliverable:* "find me a 2-bedroom under 40k" returns real results from your JSON.
- [ ] **M5 · FastAPI service.** `POST /chat` with streaming (Server-Sent Events), `GET /health`, request validation, and tests using `TestClient`.
- [ ] **M6 · Postgres.** Run Postgres in Docker. Store conversations and messages. Write the SQL by hand at least once (CREATE TABLE, INSERT, SELECT with JOIN).
- [ ] **M7 · Docker + deploy.** `Dockerfile` + `docker-compose.yml` (api + postgres). Deploy to Render, Fly.io or Railway, with secrets set as environment variables on the host.
- [ ] **M8 · CI.** A GitHub Actions workflow that runs `ruff` and `pytest` on every push.
- [ ] **M9 · Extract to `shared/`.** Move the LLM client and config loader into `shared/` so project 01 can reuse them.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/) running locally.

```bash
ollama pull llama3.2:3b          # one-time model download (~2 GB)
uv sync                          # install dependencies
uv run pytest                    # tests (no Ollama needed, HTTP is faked)
uv run chat "What is an embedding?"
uv run chat "Hi" --model qwen2.5:1.5b --retries 5
```

Example output:
```
Retrieval-Augmented Generation (RAG) is a type of ...

---
tokens: 44 in, 85 out | 15.1s (5.6 tokens/s)
cost: $0 locally, ~$0.001407 at $3.0/$15.0 per 1M tokens
```

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
