# shared/

Reusable building blocks that more than one project needs, installed into each project as a
local package. Code moves here only once a **second** project needs it, so don't abstract early.

## Modules

| Module | What | Born in |
|--------|------|---------|
| `shared.llm` | Streaming Ollama client: retries with backoff, JSON-schema output, tool calls, token counts | 00 (M2–M4) |
| `shared.costs` | `estimate_cost()` per million-token prices | 00 (M1) |
| `shared.config` | Typed settings from env vars / `.env` (`OLLAMA_BASE_URL`, `DATABASE_URL`) | 00 (M9) |

Planned: `tracing` (02), `retriever` (01), `judge` (03), `router` (04).

## Using it from a project

In the project's `pyproject.toml`:

```toml
[project]
dependencies = ["shared"]

[tool.uv.sources]
shared = { path = "../shared", editable = true }
```

`editable = true` means edits in `shared/` are picked up immediately, without reinstalling.

```python
from shared.config import get_settings
from shared.llm import stream_chat
```

## Develop

```bash
cd shared
uv sync
uv run pytest            # no Ollama or database needed
uv run ruff check . && uv run ruff format --check .
```
