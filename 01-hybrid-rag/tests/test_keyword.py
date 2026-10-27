"""Keyword search against the real Postgres. No Ollama needed."""

import os

import pytest

from rag.ingest import ChunkStore, build_chunks, load_documents
from rag.keyword import keyword_searcher, show_query

store = ChunkStore()
pytestmark = pytest.mark.skipif(
    not store.is_up() and not os.getenv("CI"),
    reason="Postgres not running (cd 00-foundations && docker compose up -d)",
)


@pytest.fixture(autouse=True)
def ingested():
    store.init_schema()
    store.replace_all(build_chunks(load_documents(), "headings", 120, 20), "headings")


def test_search_column_is_filled_automatically():
    with store.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS missing FROM rag.chunks WHERE search IS NULL")
        assert row.fetchone()["missing"] == 0


def test_questions_become_stems_without_stopwords():
    assert show_query(store, "Are pets allowed?") == "'pet' & 'allow'"


def test_stemming_matches_other_word_forms():
    # "viewing" in the question, "Viewings" in the heading
    [top] = keyword_searcher(store)("viewing", "pwani-lettings", 1)
    assert top.source == "viewings-and-applications.md"


def test_exact_numbers_are_searchable():
    [top] = keyword_searcher(store)("522100", "ruaka-rentals", 1)
    assert "522100" in top.text


def test_all_words_mode_is_too_strict_for_natural_questions():
    question = "Can I keep a cat in my bedsitter?"  # no single chunk has keep + cat + bedsitter
    assert keyword_searcher(store, mode="all")(question, "ruaka-rentals", 5) == []
    assert keyword_searcher(store, mode="any")(question, "ruaka-rentals", 5) != []


def test_tenant_filter_only_returns_that_tenants_chunks():
    results = keyword_searcher(store)("deposit rent month", "nyumba-homes", 5)
    assert results and {c.tenant_id for c in results} == {"nyumba-homes"}


def test_without_the_filter_other_agencies_leak_in():
    results = keyword_searcher(store, tenant_filter=False)("deposit", "nyumba-homes", 5)
    assert len({c.tenant_id for c in results}) > 1


def test_no_matching_words_means_no_results():
    assert keyword_searcher(store)("xylophone", "nyumba-homes", 5) == []


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        keyword_searcher(store, mode="some")
