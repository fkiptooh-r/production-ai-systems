"""Integration tests: the real SQL against the real Postgres from docker-compose.yml.

Skipped locally when the database isn't running, so `uv run pytest` still works without
Docker (start it with: docker compose up -d). In CI they never skip: a database that
failed to start must fail the build, not hide behind a green tick.
"""

import os

import pytest

from foundations.db import ConversationNotFound, PostgresStore

store = PostgresStore()
pytestmark = pytest.mark.skipif(
    not store.is_up() and not os.getenv("CI"),
    reason="Postgres not running (docker compose up -d)",
)


@pytest.fixture
def conversation_id():
    """A fresh conversation, deleted afterwards so tests don't leave data behind."""
    store.init_schema()
    conversation_id = store.create_conversation()
    yield conversation_id
    with store._connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))


def save(conversation_id: int, question: str, answer: str) -> None:
    store.add_exchange(
        conversation_id, question, answer, model="test", input_tokens=1, output_tokens=2
    )


def test_exchange_round_trip_in_order(conversation_id):
    save(conversation_id, "first question", "first answer")
    save(conversation_id, "second question", "second answer")

    messages = store.get_messages(conversation_id)

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]


def test_history_limit_keeps_the_most_recent_messages_oldest_first(conversation_id):
    for i in range(3):
        save(conversation_id, f"q{i}", f"a{i}")

    messages = store.get_messages(conversation_id, limit=3)

    assert [m["content"] for m in messages] == ["a1", "q2", "a2"]


def test_sql_injection_attempt_is_stored_as_plain_text(conversation_id):
    attack = "'); DROP TABLE messages; --"
    save(conversation_id, attack, "nice try")

    assert store.get_messages(conversation_id)[0]["content"] == attack


def test_unknown_conversation_raises():
    with pytest.raises(ConversationNotFound):
        store.get_messages(-1)


def test_list_counts_messages_and_shows_first_question(conversation_id):
    save(conversation_id, "hello there", "hi")

    listed = {c["id"]: c for c in store.list_conversations(limit=50)}

    assert listed[conversation_id]["message_count"] == 2
    assert listed[conversation_id]["first_message"] == "hello there"
