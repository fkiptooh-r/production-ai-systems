"""Conversation storage in Postgres, with hand-written SQL.

Values are always passed as parameters (%s), never pasted into the SQL string.
Pasting user text into SQL is how SQL injection happens.

Usage: uv run db-init     (re-applies sql/schema.sql; safe to run repeatedly)
"""

from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from shared.config import get_settings

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

# Only the most recent messages are sent back to the model, so long conversations
# don't grow past what a small model can read (its context window).
HISTORY_LIMIT = 20


class ConversationNotFound(Exception):
    pass


class PostgresStore:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or get_settings().database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, row_factory=dict_row, connect_timeout=3)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA_FILE.read_text())

    def is_up(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def create_conversation(self) -> int:
        with self._connect() as conn:
            row = conn.execute("INSERT INTO conversations DEFAULT VALUES RETURNING id").fetchone()
            return row["id"]

    def add_exchange(
        self,
        conversation_id: int,
        user_text: str,
        assistant_text: str,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Save a question and its answer together: both are stored, or neither is."""
        # `with connect()` is one transaction: committed at the end, rolled back on error.
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                (conversation_id, user_text),
            )
            conn.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, model, input_tokens, output_tokens)
                VALUES (%s, 'assistant', %s, %s, %s, %s)
                """,
                (conversation_id, assistant_text, model, input_tokens, output_tokens),
            )

    def get_messages(self, conversation_id: int, limit: int = HISTORY_LIMIT) -> list[dict]:
        """The last `limit` messages, oldest first."""
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s", (conversation_id,)
            ).fetchone()
            if not exists:
                raise ConversationNotFound(conversation_id)
            rows = conn.execute(
                """
                SELECT role, content, model, created_at
                FROM (
                    SELECT * FROM messages
                    WHERE conversation_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) AS latest
                ORDER BY id
                """,
                (conversation_id, limit),
            ).fetchall()
            return rows

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """Most recent conversations with their message count and first question."""
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT c.id,
                       c.created_at,
                       COUNT(m.id) AS message_count,
                       (SELECT content FROM messages
                        WHERE conversation_id = c.id
                        ORDER BY id LIMIT 1) AS first_message
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id  -- LEFT: keep empty ones too
                GROUP BY c.id
                ORDER BY c.id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()


def init_db() -> None:
    store = PostgresStore()
    store.init_schema()
    print(f"schema applied to {store.url.rsplit('@', 1)[-1]}")
