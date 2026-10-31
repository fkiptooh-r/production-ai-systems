"""#62: the test suite must never write to the database that holds the real data."""

import pytest
from conftest import DEV_DATABASE_URL, TEST_DB_NAME, pick_test_database_url
from psycopg import conninfo

from rag.ingest import ChunkStore

DEV = "postgresql://postgres:postgres@127.0.0.1:5433/ai_systems"


def dbname(url: str) -> str:
    return conninfo.conninfo_to_dict(url)["dbname"]


def test_every_store_in_the_tests_points_at_the_test_database():
    assert dbname(ChunkStore().url) == TEST_DB_NAME
    assert dbname(ChunkStore().url) != dbname(DEV_DATABASE_URL)


def test_test_database_is_the_dev_server_with_another_name(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    test = conninfo.conninfo_to_dict(pick_test_database_url(DEV))
    assert (test["host"], test["port"], test["dbname"]) == ("127.0.0.1", "5433", TEST_DB_NAME)


def test_pointing_the_tests_at_the_dev_database_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", DEV)
    with pytest.raises(pytest.UsageError, match="Refusing"):
        pick_test_database_url(DEV)


def test_another_database_can_be_chosen(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://u:p@db.example:5432/scratch")
    assert dbname(pick_test_database_url(DEV)) == "scratch"
