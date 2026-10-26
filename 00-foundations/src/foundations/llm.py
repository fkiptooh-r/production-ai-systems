"""Minimal streaming client for Ollama's /api/chat endpoint.

Ollama streams newline-delimited JSON (NDJSON): one object per piece of text,
then a final object with "done": true that carries the token counts.
"""

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Worth retrying: the server is overloaded or briefly broken.
# Not worth retrying: 4xx errors such as 404 (model not downloaded).
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """The call failed and retrying will not help."""


class _RetryableError(Exception):
    """The call failed in a way that might succeed on another attempt."""


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


def stream_chat(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    on_token: Callable[[str], None] = lambda _: None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatResult:
    """Send one prompt, call on_token for each piece of the reply, return the result.

    timeout is the longest wait for the *next* piece of data, not for the whole reply.
    Failures are retried with exponential backoff: wait 1s, 2s, 4s, ...
    client and sleep can be replaced in tests so nothing touches the network or waits.
    """
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)

    try:
        for attempt in range(max_retries + 1):
            try:
                return _stream_once(client, f"{base_url}/api/chat", payload, on_token)
            except _RetryableError as exc:
                if attempt == max_retries:
                    raise LLMError(f"gave up after {attempt + 1} attempts: {exc}") from exc
                delay = 2**attempt
                logger.warning("attempt %d failed (%s), retrying in %ds", attempt + 1, exc, delay)
                sleep(delay)
        raise AssertionError("unreachable")
    finally:
        if owns_client:
            client.close()


def _stream_once(
    client: httpx.Client,
    url: str,
    payload: dict,
    on_token: Callable[[str], None],
) -> ChatResult:
    started = time.perf_counter()
    parts: list[str] = []
    try:
        with client.stream("POST", url, json=payload) as response:
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise _RetryableError(f"HTTP {response.status_code}")
            if response.is_error:
                response.read()
                raise LLMError(f"HTTP {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(chunk["error"])

                token = chunk.get("message", {}).get("content", "")
                if token:
                    parts.append(token)
                    on_token(token)

                if chunk.get("done"):
                    return ChatResult(
                        text="".join(parts),
                        input_tokens=chunk.get("prompt_eval_count", 0),
                        output_tokens=chunk.get("eval_count", 0),
                        seconds=time.perf_counter() - started,
                    )
    except httpx.TransportError as exc:  # connection refused, timeout, dropped connection
        if parts:
            # Part of the reply was already shown; retrying would show it twice.
            raise LLMError(f"connection lost mid-reply: {exc}") from exc
        raise _RetryableError(str(exc) or type(exc).__name__) from exc

    raise LLMError("stream ended without a final 'done' message")
