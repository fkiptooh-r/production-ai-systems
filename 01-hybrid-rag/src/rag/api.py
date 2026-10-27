"""HTTP API and browser chat over each agency's documents.

Run:   uv run serve          then open http://127.0.0.1:8001 (chat) or /docs (API)
Needs Postgres (docker compose up -d in 00-foundations), Ollama, and `uv run ingest`.
Port 8001, so it can run next to 00-foundations' API on 8000.
"""

import json
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Annotated

import httpx
import psycopg
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from shared.config import get_settings
from shared.llm import ChatResult, LLMError, stream_chat

from rag.ask import tenants
from rag.chunking import Chunk
from rag.generate import DEFAULT_MODEL, agency_name, answer
from rag.ingest import PROJECT_DIR, ChunkStore
from rag.vector import ollama_embedder, vector_searcher

INDEX_HTML = PROJECT_DIR / "static" / "index.html"

app = FastAPI(
    title="Hybrid RAG API",
    description="Ask a letting agency's documents; answers cite their sources.",
)

Searcher = Callable[[str, str, int], list[Chunk]]


# --- Dependencies, swappable in tests via app.dependency_overrides ---


def get_store() -> ChunkStore:
    return ChunkStore()


def get_searcher() -> Searcher:
    # Vector search won M3-M6: Recall@1 0.98 at ~0.5 s. Tenant isolation is built in (M8).
    return vector_searcher(ChunkStore(), ollama_embedder())


def get_chat() -> Callable[..., ChatResult]:
    return stream_chat


def ollama_is_up() -> bool:
    try:
        return httpx.get(f"{get_settings().ollama_base_url}/api/version", timeout=2).is_success
    except httpx.HTTPError:
        return False


Store = Annotated[ChunkStore, Depends(get_store)]
Search = Annotated[Searcher, Depends(get_searcher)]
Chat = Annotated[Callable[..., ChatResult], Depends(get_chat)]
OllamaUp = Annotated[bool, Depends(ollama_is_up)]


# --- Request shapes ---


class AskRequest(BaseModel):
    # In a real product the tenant comes from the logged-in user or the subdomain, never
    # from the request body: anyone can edit a request. Fine for a local demo.
    tenant_id: str
    question: str = Field(min_length=1, max_length=1000)
    model: str = DEFAULT_MODEL

    @field_validator("tenant_id")
    @classmethod
    def known_tenant(cls, value: str) -> str:
        if value not in tenants():
            raise ValueError(f"unknown tenant; choose one of {tenants()}")
        return value


@app.exception_handler(psycopg.Error)
def database_failure(request: Request, exc: psycopg.Error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --- Endpoints ---


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health(response: Response, store: Store, ollama_up: OllamaUp) -> dict:
    database_up = store.is_up()
    if not (database_up and ollama_up):
        response.status_code = 503
    return {
        "status": "ok" if database_up and ollama_up else "degraded",
        "database": "up" if database_up else "down",
        "ollama": "up" if ollama_up else "down",
    }


@app.get("/tenants")
def list_tenants() -> list[dict]:
    return [{"id": t, "name": agency_name(t)} for t in tenants()]


@app.post("/ask")
def ask(body: AskRequest, search: Search, chat: Chat) -> StreamingResponse:
    """Stream Server-Sent Events: `sources` (the retrieved chunks), many `token`s, then
    `done` (which sources were cited, whether it said "I don't know") or `error`."""
    return StreamingResponse(ask_events(body, search, chat), media_type="text/event-stream")


# --- Streaming ---


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def source_dict(n: int, chunk: Chunk) -> dict:
    return {
        "n": n,
        "source": chunk.source,
        "section": chunk.section.split(" > ")[-1],
        "text": chunk.text,
    }


def ask_events(
    body: AskRequest, search: Searcher, chat: Callable[..., ChatResult]
) -> Iterator[str]:
    """Run retrieval + generation in a thread, yielding events as they happen.

    Retrieval runs first so the page can show the sources before the answer starts; the
    same chunks are then handed to answer(), so what's shown is exactly what the model read.
    """
    events: queue.Queue = queue.Queue()
    finished = object()

    def worker() -> None:
        try:
            chunks = search(body.question, body.tenant_id, 4)
            events.put(
                sse_event(
                    "sources", {"sources": [source_dict(n, c) for n, c in enumerate(chunks, 1)]}
                )
            )
            result = answer(
                body.question,
                body.tenant_id,
                lambda *_: chunks,
                model=body.model,
                chat=chat,
                on_token=lambda t: events.put(sse_event("token", {"text": t})),
            )
            cited = [n for n, c in enumerate(chunks, 1) if c in result.cited]
            done = {
                "cited": cited,
                "abstained": result.abstained,
                "seconds": round(result.seconds, 1),
            }
            events.put(sse_event("done", done))
        except LLMError as exc:  # headers (200) are already sent, so errors travel as events
            events.put(sse_event("error", {"message": f"model failure: {exc}"}))
        except psycopg.Error:
            events.put(sse_event("error", {"message": "database unavailable"}))
        finally:
            events.put(finished)

    threading.Thread(target=worker, daemon=True).start()
    while (item := events.get()) is not finished:
        yield item


def serve() -> None:
    uvicorn.run("rag.api:app", host="127.0.0.1", port=8001, reload=True)
