"""HTTP API over the chat, agent and extraction features.

Run:   uv run serve          then open http://localhost:8000/docs
"""

import json
import os
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Annotated

import httpx
import psycopg
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from foundations.agent import AgentError, ToolStep, run_agent
from foundations.db import ConversationNotFound, PostgresStore
from foundations.extract import ExtractionError, Listing, extract_listing
from foundations.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, ChatResult, LLMError, stream_chat

# Chosen by measurement: qwen2.5:1.5b scored 100% / 90% on the M4 tool eval vs llama's 80% / 80%.
AGENT_MODEL = "qwen2.5:1.5b"

app = FastAPI(
    title="Foundations API",
    description="Chat, a rental-search agent, and listing extraction over local LLMs.",
)


# --- Dependencies: things endpoints need, swappable in tests via app.dependency_overrides ---


def get_chat() -> Callable[..., ChatResult]:
    return stream_chat


def ollama_is_up() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    try:
        return httpx.get(f"{base_url}/api/version", timeout=2).is_success
    except httpx.HTTPError:
        return False


def get_store() -> PostgresStore:
    return PostgresStore()


Chat = Annotated[Callable[..., ChatResult], Depends(get_chat)]
OllamaUp = Annotated[bool, Depends(ollama_is_up)]
Store = Annotated[PostgresStore, Depends(get_store)]


# --- Request and response shapes (FastAPI validates these and shows them in /docs) ---


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    model: str = DEFAULT_MODEL
    # Leave out to start a new conversation; send back the id from `done` to continue one.
    conversation_id: int | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str = AGENT_MODEL


class AgentResponse(BaseModel):
    answer: str
    steps: list[dict]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model: str = DEFAULT_MODEL


# --- Errors: the model failing is not the caller's fault, so it's a 502 (bad upstream) ---


@app.exception_handler(LLMError)
@app.exception_handler(AgentError)
@app.exception_handler(ExtractionError)
def model_failure(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": f"model failure: {exc}"})


@app.exception_handler(ConversationNotFound)
def conversation_not_found(request: Request, exc: ConversationNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"no conversation {exc}"})


@app.exception_handler(psycopg.Error)
def database_failure(request: Request, exc: psycopg.Error) -> JSONResponse:
    # Don't echo database errors to callers: they can reveal table names and queries.
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --- Endpoints ---


@app.get("/health")
def health(response: Response, ollama_up: OllamaUp, store: Store) -> dict:
    """Can the service reach Ollama and the database? 503 if not, so load balancers notice."""
    database_up = store.is_up()
    if not (ollama_up and database_up):
        response.status_code = 503
    return {
        "status": "ok" if ollama_up and database_up else "degraded",
        "ollama": "up" if ollama_up else "down",
        "database": "up" if database_up else "down",
    }


@app.post("/chat")
def chat(body: ChatRequest, chat: Chat, store: Store) -> StreamingResponse:
    """Stream the reply as Server-Sent Events: `token` events, then `done` (or `error`).

    The model sees earlier messages from the same conversation, so follow-up
    questions like "and the cheaper one?" make sense.
    """
    if body.conversation_id is None:
        conversation_id = store.create_conversation()
        history = []
    else:
        conversation_id = body.conversation_id
        history = store.get_messages(conversation_id)  # 404 if it doesn't exist

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": body.prompt})

    def save(result: ChatResult) -> dict:
        store.add_exchange(
            conversation_id,
            body.prompt,
            result.text,
            model=body.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return {"conversation_id": conversation_id}

    return StreamingResponse(
        sse_chat_events(messages, body.model, chat, on_complete=save),
        media_type="text/event-stream",
    )


@app.get("/conversations")
def list_conversations(store: Store) -> list[dict]:
    """Most recent conversations, with message counts."""
    return store.list_conversations()


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, store: Store) -> dict:
    return {"id": conversation_id, "messages": store.get_messages(conversation_id, limit=1000)}


@app.post("/agent")
def agent(body: AgentRequest, chat: Chat) -> AgentResponse:
    """Run the rental assistant; returns its answer and every tool call it made."""
    result = run_agent(body.question, model=body.model, chat=chat)
    return AgentResponse(answer=result.answer, steps=[_step_dict(s) for s in result.steps])


@app.post("/extract")
def extract(body: ExtractRequest, chat: Chat) -> Listing:
    """Turn free-text listing into structured, validated fields."""
    return extract_listing(body.text, model=body.model, chat=chat)


# --- Helpers ---


def sse_event(event: str, data: dict) -> str:
    """One Server-Sent Event: a name line, a data line, then a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_chat_events(
    prompt: str | list[dict],
    model: str,
    chat: Callable[..., ChatResult],
    on_complete: Callable[[ChatResult], dict] = lambda _: {},
) -> Iterator[str]:
    """Bridge a callback-style chat (on_token) to a generator FastAPI can stream.

    The chat runs in a background thread and drops each token into a queue; this
    generator takes them out and yields them as SSE as soon as they arrive.
    on_complete runs after a successful reply (e.g. to save it); what it returns
    is added to the `done` event.
    """
    events: queue.Queue = queue.Queue()
    finished = object()

    def worker() -> None:
        try:
            result = chat(
                prompt, model=model, on_token=lambda t: events.put(sse_event("token", {"text": t}))
            )
            done = {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "seconds": round(result.seconds, 2),
                **on_complete(result),
            }
            events.put(sse_event("done", done))
        except LLMError as exc:
            # Headers (status 200) were already sent, so errors travel as an event.
            events.put(sse_event("error", {"message": str(exc)}))
        except psycopg.Error:
            events.put(sse_event("error", {"message": "reply could not be saved"}))
        finally:
            events.put(finished)

    threading.Thread(target=worker, daemon=True).start()
    while (item := events.get()) is not finished:
        yield item


def _step_dict(step: ToolStep) -> dict:
    return {"tool": step.name, "arguments": step.arguments, "output": json.loads(step.output)}


def serve() -> None:
    uvicorn.run("foundations.api:app", host="127.0.0.1", port=8000, reload=True)
