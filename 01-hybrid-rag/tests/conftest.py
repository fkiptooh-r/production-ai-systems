"""Shared test setup. Runs before any test module is imported.

Tests that touch Postgres run against their own database, never the one holding real data.
Before this, test_vector.py and friends replaced the real chunks and embeddings in the dev
database with fakes, which silently broke retrieval (MRR 0.99 -> 0.09) until the next
`uv run ingest` (#62).
"""

import os

import psycopg
import pytest
from psycopg import conninfo, sql
from shared.config import get_settings
from tracing import otel

TEST_DB_NAME = "ai_systems_test"


def pick_test_database_url(dev_url: str) -> str:
    """TEST_DATABASE_URL if set, else the dev URL with the database renamed. Never the dev DB."""
    url = os.environ.get("TEST_DATABASE_URL") or conninfo.make_conninfo(
        dev_url, dbname=TEST_DB_NAME
    )
    test, dev = conninfo.conninfo_to_dict(url), conninfo.conninfo_to_dict(dev_url)
    same = [test.get(k) for k in ("host", "port", "dbname")] == [
        dev.get(k) for k in ("host", "port", "dbname")
    ]
    if same:
        raise pytest.UsageError(
            f"Refusing to run tests against the dev database {dev.get('dbname')!r}: they replace "
            "its chunks and embeddings. Point TEST_DATABASE_URL at another database."
        )
    return url


def ensure_database(url: str) -> None:
    """Create the test database if it's missing. Postgres down: the DB tests skip themselves."""
    name = conninfo.conninfo_to_dict(url)["dbname"]
    try:
        admin = conninfo.make_conninfo(url, dbname="postgres")
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.OperationalError:
        pass


DEV_DATABASE_URL = get_settings().database_url
TEST_DATABASE_URL = pick_test_database_url(DEV_DATABASE_URL)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL  # every ChunkStore() in the tests now uses it
get_settings.cache_clear()
ensure_database(TEST_DATABASE_URL)


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    """Every test writes its spans to its own temp file, never to traces/ in the project."""
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def no_export():
    """Tests never send spans to a real Langfuse, even if .env has its keys."""
    otel.disable()
    yield
    otel.reset()
