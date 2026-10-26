"""API tests: real HTTP requests through FastAPI's TestClient, with a fake model and store."""

import json

import pytest
from fastapi.testclient import TestClient

from foundations.api import app, get_chat, get_store, ollama_is_up
from foundations.db import ConversationNotFound
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
        self.prompts: list = []

    def __call__(self, prompt, *, on_token=lambda _: None, **kwargs) -> ChatResult:
        self.prompts.append(prompt)
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


class InMemoryStore:
    """Stands in for PostgresStore: same methods, data kept in a dict."""

    def __init__(self) -> None:
        self.up = True
        self.conversations: dict[int, list[dict]] = {}

    def is_up(self) -> bool:
        return self.up

    def create_conversation(self) -> int:
        conversation_id = len(self.conversations) + 1
        self.conversations[conversation_id] = []
        return conversation_id

    def add_exchange(self, conversation_id, user_text, assistant_text, **details) -> None:
        self.conversations[conversation_id] += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    def get_messages(self, conversation_id: int, limit: int = 20) -> list[dict]:
        if conversation_id not in self.conversations:
            raise ConversationNotFound(conversation_id)
        return self.conversations[conversation_id][-limit:]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def client(store):
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_model(*replies: ChatResult | Exception) -> FakeChat:
    model = FakeChat(*replies)
    app.dependency_overrides[get_chat] = lambda: model
    return model


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        name_line, data_line = block.split("\n")
        events.append((name_line.removeprefix("event: "), json.loads(data_line[len("data: ") :])))
    return events


def test_health_ok_when_everything_is_up(client):
    app.dependency_overrides[ollama_is_up] = lambda: True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": "up", "database": "up"}


def test_health_is_503_when_ollama_is_down(client):
    app.dependency_overrides[ollama_is_up] = lambda: False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["ollama"] == "down"


def test_health_is_503_when_database_is_down(client, store):
    app.dependency_overrides[ollama_is_up] = lambda: True
    store.up = False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "down"


def test_chat_streams_tokens_then_done(client):
    use_model(text_reply("Hello there"))

    response = client.post("/chat", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[:2] == [("token", {"text": "Hello"}), ("token", {"text": "there"})]
    assert events[-1] == (
        "done",
        {"input_tokens": 7, "output_tokens": 3, "seconds": 0.5, "conversation_id": 1},
    )


def test_follow_up_sees_earlier_messages_and_everything_is_saved(client, store):
    model = use_model(text_reply("L010 is 35k."), text_reply("It is in Ruaka."))

    first = parse_sse(client.post("/chat", json={"prompt": "cheapest 2BR?"}).text)
    conversation_id = first[-1][1]["conversation_id"]
    client.post("/chat", json={"prompt": "where is it?", "conversation_id": conversation_id})

    assert model.prompts[1] == [
        {"role": "user", "content": "cheapest 2BR?"},
        {"role": "assistant", "content": "L010 is 35k."},
        {"role": "user", "content": "where is it?"},
    ]
    saved = client.get(f"/conversations/{conversation_id}").json()["messages"]
    assert [m["content"] for m in saved] == [
        "cheapest 2BR?",
        "L010 is 35k.",
        "where is it?",
        "It is in Ruaka.",
    ]


def test_unknown_conversation_is_404(client):
    use_model()
    response = client.post("/chat", json={"prompt": "hi", "conversation_id": 999})
    assert response.status_code == 404


def test_chat_failure_arrives_as_an_error_event_and_nothing_is_saved(client, store):
    use_model(LLMError("gave up after 4 attempts"))

    events = parse_sse(client.post("/chat", json={"prompt": "hi"}).text)

    assert events == [("error", {"message": "gave up after 4 attempts"})]
    assert store.conversations[1] == []


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
