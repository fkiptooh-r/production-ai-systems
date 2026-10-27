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


def test_home_page_links_to_every_agency(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Agency Docs Chat</title>" in response.text
    for tenant_id in ("nyumba-homes", "pwani-lettings", "ruaka-rentals"):
        assert f'href="http://testserver/t/{tenant_id}"' in response.text


def test_agency_page_is_rendered_in_its_brand(client):
    html = client.get("/t/ruaka-rentals").text

    assert "<title>Ruaka Rentals · Ask the agency</title>" in html
    assert "Affordable bedsitters and family units in Ruaka" in html
    assert "--brand: #a8501c; --brand-dark: #f0a46c;" in html
    assert 'data-tenant="ruaka-rentals"' in html
    assert 'data-question="Can I keep a cat?"' in html  # Ruaka's own example questions
    assert 'href="http://testserver/t/ruaka-rentals"\n         aria-current="page"' in html


def test_unknown_agency_page_is_a_404_that_still_offers_the_real_ones(client):
    response = client.get("/t/someone-else")
    assert response.status_code == 404
    assert "No such agency." in response.text
    assert "Ruaka Rentals" in response.text


def test_static_files_are_served(client):
    assert "--brand" in client.get("/static/style.css").text
    assert "document.body.dataset" in client.get("/static/chat.js").text


def test_agency_from_the_url_is_the_one_searched(client):
    stream = events(client.post("/t/pwani-lettings/ask", json={"question": "Dogs?"}))
    assert stream[-1][0] == "done"
    assert client.searched == [("Dogs?", "pwani-lettings", 4)]


def test_ask_for_an_unknown_agency_in_the_url_is_404_before_any_search(client):
    response = client.post("/t/someone-else/ask", json={"question": "Dogs?"})
    assert response.status_code == 404
    assert client.searched == []


def test_every_tenant_folder_has_a_valid_brand():
    from rag.ask import tenants
    from rag.brand import brands

    assert list(brands()) == tenants()


def test_template_values_are_escaped(client, monkeypatch):
    from rag import api
    from rag.brand import Brand

    evil = Brand(
        tenant_id="evil",
        name="<script>alert(1)</script>",
        tagline="x",
        accent="#000000",
        accent_dark="#ffffff",
        examples=['"><img src=x>'],
    )
    monkeypatch.setattr(api, "brands", lambda: {"evil": evil})
    html = client.get("/t/evil").text

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert '"><img src=x>' not in html


@pytest.mark.parametrize("colour", ["red", "#12345", "#000000; } body { display: none"])
def test_brand_colours_must_be_plain_hex_because_css_is_not_escaped(colour):
    from pydantic import ValidationError

    from rag.brand import Brand

    with pytest.raises(ValidationError):
        Brand(tenant_id="t", name="T", tagline="", accent=colour, accent_dark="#ffffff")


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
