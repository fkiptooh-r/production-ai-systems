"""M8: prove one tenant can never read another tenant's chunks.

Two layers are tested separately:
1. The search functions: every eval question, asked of every tenant, through vector,
   keyword and hybrid search, returns only that tenant's chunks.
2. The database alone (Row-Level Security): even SQL with NO tenant filter at all, run on a
   tenant connection, sees only that tenant's rows. So a future bug in a query can't leak.
"""

import os

import psycopg
import pytest

from rag.hybrid import hybrid_searcher
from rag.ingest import ChunkStore, build_chunks, load_documents
from rag.keyword import keyword_searcher
from rag.vector import embed_chunks, vector_searcher

store = ChunkStore()
pytestmark = pytest.mark.skipif(
    not store.is_up() and not os.getenv("CI"),
    reason="Postgres not running (cd 00-foundations && docker compose up -d)",
)

TENANTS = ["nyumba-homes", "pwani-lettings", "ruaka-rentals"]


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for Ollama: letter counts. Isolation doesn't depend on quality."""
    vectors = []
    for text in texts:
        vector = [0.0] * 768
        for ch in text.lower():
            vector[ord(ch) % 768] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture(scope="module", autouse=True)
def corpus():
    store.init_schema()
    chunks = build_chunks(load_documents(), "headings", 120, 20)
    store.replace_all(chunks, "headings")
    embed_chunks(store, fake_embedder)
    return chunks


def count_visible(tenant_id: str, sql: str = "SELECT COUNT(*) AS n FROM rag.chunks") -> int:
    with store.tenant_connection(tenant_id) as conn:
        return conn.execute(sql).fetchone()["n"]


# --- Layer 1: the search functions ---


@pytest.mark.parametrize("tenant", TENANTS)
def test_no_search_ever_returns_another_tenants_chunks(tenant):
    from rag.evaluate import load_questions

    vector = vector_searcher(store, fake_embedder)
    keyword = keyword_searcher(store)
    searches = {"vector": vector, "keyword": keyword, "hybrid": hybrid_searcher([vector, keyword])}

    for question in load_questions():  # all 60, asked of every tenant, not just their own
        for name, search in searches.items():
            leaked = [
                c.chunk_id for c in search(question.question, tenant, 20) if c.tenant_id != tenant
            ]
            assert leaked == [], f"{name} leaked {leaked} into {tenant} for {question.id}"


def test_other_tenants_policy_words_find_nothing_from_them():
    # "holding fee" exists only in Pwani's documents; asked as Ruaka, keyword search must
    # come back empty rather than borrow Pwani's chunk.
    results = keyword_searcher(store)("holding fee KES 5,000", "ruaka-rentals", 5)
    assert all(c.tenant_id == "ruaka-rentals" for c in results)
    assert not any("holding fee" in c.text.lower() for c in results)


# --- Layer 2: the database on its own ---


def test_unfiltered_sql_on_a_tenant_connection_sees_only_that_tenant(corpus):
    for tenant in TENANTS:
        expected = sum(c.tenant_id == tenant for c in corpus)
        assert count_visible(tenant) == expected  # no WHERE clause at all

        with store.tenant_connection(tenant) as conn:
            seen = {r["tenant_id"] for r in conn.execute("SELECT tenant_id FROM rag.chunks")}
        assert seen == {tenant}


def test_owner_connection_sees_everything_which_is_why_searches_must_not_use_it(corpus):
    # The table owner bypasses RLS. Documented here so nobody "simplifies" the searchers
    # back to store.connect().
    with store.connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM rag.chunks").fetchone()["n"]
    assert total == len(corpus)


def test_no_tenant_set_means_no_rows():
    # Fails closed: a connection that switches role but forgets the tenant sees nothing.
    with store.connect() as conn, conn.transaction():
        conn.execute("SET LOCAL ROLE rag_reader")
        assert conn.execute("SELECT COUNT(*) AS n FROM rag.chunks").fetchone()["n"] == 0


@pytest.mark.parametrize("tenant", ["", "unknown-agency", "NYUMBA-HOMES", "x' OR '1'='1"])
def test_unknown_misspelled_or_malicious_tenant_ids_see_nothing(tenant):
    assert count_visible(tenant) == 0


def test_known_limit_sql_inside_the_connection_can_change_the_tenant():
    # rag_reader could call set_config itself; the guarantee is that the *app* sets it once
    # per request. This test pins down that a later set_config does change what's visible,
    # so the app must never pass user input to set_config except as the tenant parameter.
    with store.tenant_connection("ruaka-rentals") as conn:
        conn.execute("SELECT set_config('app.tenant_id', 'nyumba-homes', true)")
        seen = {r["tenant_id"] for r in conn.execute("SELECT tenant_id FROM rag.chunks")}
    assert seen == {"nyumba-homes"}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM rag.chunks",
        "UPDATE rag.chunks SET content = 'hacked'",
        "INSERT INTO rag.chunks (chunk_id, tenant_id, source, section, chunk_index, content,"
        " word_count, strategy) VALUES ('x', 'ruaka-rentals', 's', 's', 0, 'c', 1, 'h')",
    ],
)
def test_tenant_connection_is_read_only(sql):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with store.tenant_connection("ruaka-rentals") as conn:
            conn.execute(sql)


def test_role_settings_do_not_outlive_the_transaction():
    with store.connect() as conn:
        with conn.transaction():
            conn.execute("SET LOCAL ROLE rag_reader")
            conn.execute("SELECT set_config('app.tenant_id', 'ruaka-rentals', true)")
        row = conn.execute(
            "SELECT current_user AS role, current_setting('app.tenant_id', true) AS tenant"
        ).fetchone()
    assert row["role"] != "rag_reader"
    assert not row["tenant"]  # '' or NULL once the transaction is over
