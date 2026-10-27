"""API tests with a fake search and a fake model: no Ollama, no database."""

import json

import pytest
from fastapi.testclient import TestClient
from shared.llm import ChatResult, LLMError

from rag.api import app, get_chat, get_searcher, get_store, ollama_is_up
from rag.chunking import Chunk

PETS = Chunk(
    "ruaka-rentals",
    "tenancy-policy.md",
    "Ruaka Rentals Tenancy Terms > Pets",
    5,
    "No pets are allowed in any Ruaka Rentals unit.",
)
RENT = Chunk(
    "ruaka-rentals",
    "tenancy-policy.md",
    "Ruaka Rentals Tenancy Terms > Rent payment",
    2,
    "Rent is due on the 1st of every month.",
)


def fake_chat(reply: str):
    def chat(messages, **kwargs):
        for token in reply.split(" "):
            kwargs["on_token"](token + " ")
        return ChatResult(text=reply, input_tokens=50, output_tokens=8, seconds=0.4)

    return chat


def broken_chat(messages, **kwargs):
    raise LLMError("Ollama not reachable")


class FakeStore:
    def __init__(self, up: bool) -> None:
        self.up = up

    def is_up(self) -> bool:
        return self.up


@pytest.fixture
def client():
    searched = []

    def search(question, tenant_id, k):
        searched.append((question, tenant_id, k))
        return [PETS, RENT][:k]

    app.dependency_overrides[get_searcher] = lambda: search
    app.dependency_overrides[get_chat] = lambda: fake_chat("No pets are allowed [1].")
    with TestClient(app) as test_client:
        test_client.searched = searched
        yield test_client
    app.dependency_overrides.clear()


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Agency Docs Chat</title>" in response.text


def test_tenants_lists_every_agency_with_a_display_name(client):
    assert client.get("/tenants").json() == [
        {"id": "nyumba-homes", "name": "Nyumba Homes"},
        {"id": "pwani-lettings", "name": "Pwani Lettings"},
        {"id": "ruaka-rentals", "name": "Ruaka Rentals"},
    ]


def test_ask_streams_sources_then_tokens_then_done(client):
    response = client.post("/ask", json={"tenant_id": "ruaka-rentals", "question": "Cats?"})
    stream = events(response)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert stream[0][0] == "sources"
    assert [s["section"] for s in stream[0][1]["sources"]] == ["Pets", "Rent payment"]
    assert {name for name, _ in stream[1:-1]} == {"token"}
    assert "".join(d["text"] for n, d in stream if n == "token").strip() == (
        "No pets are allowed [1]."
    )
    assert stream[-1] == ("done", {"cited": [1], "abstained": False, "seconds": 0.4})
    assert client.searched == [("Cats?", "ruaka-rentals", 4)]  # searched once, as that tenant


def test_i_dont_know_is_flagged(client):
    app.dependency_overrides[get_chat] = lambda: fake_chat("I don't know.")
    stream = events(client.post("/ask", json={"tenant_id": "ruaka-rentals", "question": "Gym?"}))
    assert stream[-1][1]["abstained"] is True
    assert stream[-1][1]["cited"] == []


def test_model_failure_arrives_as_an_error_event(client):
    app.dependency_overrides[get_chat] = lambda: broken_chat
    stream = events(client.post("/ask", json={"tenant_id": "ruaka-rentals", "question": "Cats?"}))
    assert stream[-1][0] == "error"
    assert "Ollama not reachable" in stream[-1][1]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {"tenant_id": "someone-else", "question": "Cats?"},  # unknown tenant
        {"tenant_id": "ruaka-rentals", "question": ""},  # empty question
        {"tenant_id": "ruaka-rentals", "question": "x" * 1001},  # too long
        {"question": "Cats?"},  # no tenant at all
    ],
)
def test_bad_requests_are_rejected_before_any_search(client, body):
    assert client.post("/ask", json=body).status_code == 422
    assert client.searched == []


@pytest.mark.parametrize(
    ("db", "ollama", "status"), [(True, True, 200), (False, True, 503), (True, False, 503)]
)
def test_health_reports_each_dependency(client, db, ollama, status):
    app.dependency_overrides[get_store] = lambda: FakeStore(db)
    app.dependency_overrides[ollama_is_up] = lambda: ollama
    response = client.get("/health")
    assert response.status_code == status
    assert response.json()["database"] == ("up" if db else "down")
