"""API tests: real HTTP requests through FastAPI's TestClient, with fake models."""

import json

import pytest
from fastapi.testclient import TestClient

from foundations.api import app, get_chat, ollama_is_up
from foundations.llm import ChatResult, LLMError

VALID_LISTING = {
    "property_type": "apartment",
    "bedrooms": 2,
    "price_per_month": 45000,
    "location": "Kilimani",
    "parking": True,
    "furnished": None,
    "amenities": [],
}


class FakeChat:
    """Stands in for stream_chat: streams scripted replies through on_token."""

    def __init__(self, *replies: ChatResult | Exception) -> None:
        self.replies = list(replies)

    def __call__(self, prompt, *, on_token=lambda _: None, **kwargs) -> ChatResult:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for word in reply.text.split(" "):
            on_token(word)
        return reply


def text_reply(text: str, *tool_calls: dict) -> ChatResult:
    return ChatResult(
        text, input_tokens=7, output_tokens=3, seconds=0.5, tool_calls=list(tool_calls)
    )


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_model(*replies: ChatResult | Exception) -> None:
    app.dependency_overrides[get_chat] = lambda: FakeChat(*replies)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        name_line, data_line = block.split("\n")
        events.append((name_line.removeprefix("event: "), json.loads(data_line[len("data: ") :])))
    return events


def test_health_ok_when_ollama_is_up(client):
    app.dependency_overrides[ollama_is_up] = lambda: True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": "up"}


def test_health_is_503_when_ollama_is_down(client):
    app.dependency_overrides[ollama_is_up] = lambda: False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["ollama"] == "down"


def test_chat_streams_tokens_then_done(client):
    use_model(text_reply("Hello there"))

    response = client.post("/chat", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[:2] == [("token", {"text": "Hello"}), ("token", {"text": "there"})]
    assert events[-1] == ("done", {"input_tokens": 7, "output_tokens": 3, "seconds": 0.5})


def test_chat_failure_arrives_as_an_error_event(client):
    use_model(LLMError("gave up after 4 attempts"))

    events = parse_sse(client.post("/chat", json={"prompt": "hi"}).text)

    assert events == [("error", {"message": "gave up after 4 attempts"})]


def test_empty_prompt_is_rejected_before_reaching_the_model(client):
    use_model()  # no replies: the test fails if the model is called
    response = client.post("/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_agent_returns_answer_and_tool_steps(client):
    search = {"function": {"name": "get_listing", "arguments": {"listing_id": "L007"}}}
    use_model(text_reply("", search), text_reply("L007 is a 3 bed bungalow in Ngong for 40,000."))

    response = client.post("/agent", json={"question": "Tell me about L007"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("L007")
    assert body["steps"][0]["tool"] == "get_listing"
    assert body["steps"][0]["output"]["location"] == "Ngong"


def test_extract_returns_a_validated_listing(client):
    use_model(text_reply(json.dumps(VALID_LISTING)))

    response = client.post("/extract", json={"text": "2BR in Kilimani, 45k, parking"})

    assert response.status_code == 200
    assert response.json() == VALID_LISTING


def test_model_failure_is_a_502_not_a_crash(client):
    use_model(text_reply("not json"), text_reply("not json"), text_reply("not json"))

    response = client.post("/extract", json={"text": "gibberish"})

    assert response.status_code == 502
    assert "model failure" in response.json()["detail"]
