"""Tests for the embedding client using a fake HTTP transport, so no server is needed."""

import json

import httpx
import pytest

from shared.embeddings import embed
from shared.llm import LLMError


def fake_client(response: httpx.Response | Exception, sent: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(json.loads(request.content))
        if isinstance(response, Exception):
            raise response
        return response

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_one_vector_per_text_in_one_request():
    sent = []
    reply = httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    vectors = embed(["a", "b"], model="m", client=fake_client(reply, sent))

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert sent == [{"model": "m", "input": ["a", "b"]}]


def test_no_texts_means_no_request():
    assert embed([], client=fake_client(RuntimeError("should not be called"))) == []


def test_missing_model_says_how_to_fix_it():
    with pytest.raises(LLMError, match="ollama pull nomic-embed-text"):
        embed(["a"], client=fake_client(httpx.Response(404, json={"error": "not found"})))


def test_server_down_is_an_llm_error():
    with pytest.raises(LLMError, match="not reachable"):
        embed(["a"], client=fake_client(httpx.ConnectError("refused")))


def test_wrong_number_of_vectors_is_caught():
    reply = httpx.Response(200, json={"embeddings": [[0.1]]})
    with pytest.raises(LLMError, match="asked for 2"):
        embed(["a", "b"], client=fake_client(reply))
