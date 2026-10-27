"""Integration tests against the real Postgres (docker compose up -d in 00-foundations).

Skipped locally when the database isn't running; in CI they fail instead of skipping.
"""

import os

import pytest

from rag.ingest import ChunkStore, build_chunks, load_documents

store = ChunkStore()
pytestmark = pytest.mark.skipif(
    not store.is_up() and not os.getenv("CI"),
    reason="Postgres not running (cd 00-foundations && docker compose up -d)",
)


@pytest.fixture
def ingested():
    store.init_schema()
    chunks = build_chunks(load_documents(), "headings", size=120, overlap=20)
    store.replace_all(chunks, "headings")
    return chunks


def test_every_chunk_is_stored_per_tenant(ingested):
    stats = {row["tenant_id"]: row for row in store.stats()}

    assert set(stats) == {"nyumba-homes", "pwani-lettings", "ruaka-rentals"}
    assert sum(row["chunks"] for row in stats.values()) == len(ingested)
    assert all(row["documents"] == 4 for row in stats.values())


def test_reingesting_replaces_instead_of_duplicating(ingested):
    before = sum(row["chunks"] for row in store.stats())
    store.replace_all(ingested, "headings")
    assert sum(row["chunks"] for row in store.stats()) == before


def test_switching_strategy_replaces_the_whole_corpus(ingested):
    fixed = build_chunks(load_documents(), "fixed", size=60, overlap=10)
    store.replace_all(fixed, "fixed")

    assert sum(row["chunks"] for row in store.stats()) == len(fixed)
    with store.connect() as conn:
        strategies = conn.execute("SELECT DISTINCT strategy FROM rag.chunks").fetchall()
    assert strategies == [{"strategy": "fixed"}]
