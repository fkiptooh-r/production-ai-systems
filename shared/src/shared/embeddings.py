"""Turn text into embedding vectors with Ollama's /api/embed endpoint.

An embedding is a list of numbers (768 for nomic-embed-text) that places a text in a
"meaning space": texts that say similar things end up close together, even when they share
no words. Search then becomes "find the stored vectors closest to the question's vector".
"""

import httpx

from shared.config import get_settings
from shared.llm import LLMError

DEFAULT_EMBED_MODEL = "nomic-embed-text"


def embed(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    """One vector per text, in the same order. Send texts in batches: one request each."""
    if not texts:
        return []
    base_url = base_url or get_settings().ollama_base_url
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.post(f"{base_url}/api/embed", json={"model": model, "input": texts})
    except httpx.TransportError as exc:
        raise LLMError(f"Ollama not reachable at {base_url}: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    if response.status_code == 404:
        raise LLMError(f"model {model!r} not found; run: ollama pull {model}")
    if response.is_error:
        raise LLMError(f"HTTP {response.status_code}: {response.text}")

    vectors = response.json().get("embeddings", [])
    if len(vectors) != len(texts):
        raise LLMError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors
