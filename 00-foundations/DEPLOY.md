# Deploying the Foundations API

Status: **the Docker image is built and tested locally; the cloud deploy is pending.**
Follow this when a host is chosen. Tick items off in the open M7 pull request.

## What gets deployed

```
            internet
               │  HTTPS
        ┌──────▼──────┐
        │  API image  │  this repo's Dockerfile (234 MB, runs as non-root)
        └──┬───────┬──┘
           │       │
   ┌───────▼──┐ ┌──▼──────────┐
   │ Postgres │ │ LLM backend │  ← the decision below
   └──────────┘ └─────────────┘
```

## 1. Decide: where does the model run?

Ollama needs several GB of RAM and is slow without a GPU, so it doesn't fit on free or
small cloud instances. Options, cheapest first:

| Option | Cost | Trade-off |
|--------|------|-----------|
| **A. Tunnel to this laptop** (e.g. Cloudflare Tunnel) | Free | Public URL for demos; only works while the laptop is on. **Add auth first (step 2).** |
| **B. Hosted LLM API** (Anthropic, OpenAI, OpenRouter, …) | Pay per token | Fast and reliable. Needs a new provider client in `llm.py` (it only speaks Ollama today) and a **spending limit** on the provider dashboard. |
| **C. VM running Ollama** | Monthly VM cost | Full control; needs ≥ 8 GB RAM, and it's slow on CPU. |

Prices and free tiers change often, so check them on each provider's site before choosing.

## 2. Blockers: do these before anything is public

- [ ] **Authentication.** Right now anyone who finds the URL can use the model and fill the
      database, and with option B, spend your money. Add an `X-API-Key` header check
      (a FastAPI dependency, like `get_store`) and reject requests without it (401).
- [ ] **Rate limiting.** Cap requests per key per minute so one caller can't run up costs.
- [ ] **CORS.** If a browser front end will call the API, allow only its exact origin.
- [ ] **Real secrets.** A strong database password, never `postgres/postgres`. Secrets go in
      the host's environment-variable settings, never in git.
- [ ] **Scan the image.** VS Code's Docker extension flagged 1 high-severity vulnerability in
      the `python:3.12-slim` base (Oct 2026). Rebuild with the latest base
      (`docker compose --profile api build --pull`), then check with `docker scout cves
      00-foundations-api` and fix or accept each high/critical finding.
- [ ] **Turn off `reload`.** `uv run serve` uses auto-reload for development; the Dockerfile's
      `CMD` already runs without it.

## 3. Database

- [ ] Create a managed Postgres (several providers have free tiers; pick one with **pgvector**
      support, which project 01 needs).
- [ ] Apply the schema: `DATABASE_URL=<cloud url> uv run db-init`
- [ ] Make sure the database only accepts connections from the API (or requires SSL + a password).

## 4. Deploy the image

Most platforms (Render, Fly.io, Railway, Google Cloud Run, a VM with Docker) can either build
from this `Dockerfile` or run the image directly.

- [ ] Point the platform at `00-foundations/` (the Dockerfile's folder).
- [ ] Set environment variables: `DATABASE_URL`, `OLLAMA_BASE_URL` (or the provider key for
      option B), and the API key from step 2.
- [ ] Health check path: `/health` (it returns 503 when the model or database is unreachable).
- [ ] Port: the container listens on `8000`.

## 5. Smoke test after deploy

```bash
URL=https://your-app.example.com
curl -s $URL/health                                            # expect "status": "ok"
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/chat \
  -H "Content-Type: application/json" -d '{"prompt":"hi"}'     # expect 401 without a key
curl -N -X POST $URL/chat -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" -d '{"prompt":"hi"}'     # expect streamed tokens
```

- [ ] All three behave as expected
- [ ] Spending limit set on any paid provider
- [ ] Live URL added to the project README

## Run the production image locally (free, works today)

```bash
docker compose --profile api up -d --build          # API on 127.0.0.1:8000 + Postgres
API_PORT=8001 docker compose --profile api up -d    # if 8000 is busy (e.g. uv run serve)
curl http://127.0.0.1:8000/health
docker compose --profile api down                   # stop both (data is kept)
```

Inside Docker, the API reaches Ollama on Windows through `host.docker.internal` and Postgres
by its service name `postgres`; see `docker-compose.yml`.
