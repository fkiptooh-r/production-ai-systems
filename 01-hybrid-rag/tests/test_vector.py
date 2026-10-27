"""Vector search against the real Postgres + pgvector, with a fake embedder instead of Ollama.

The fake turns text into word counts hashed into 768 buckets: crude, but deterministic and
"similar words → nearby vectors", which is enough to test the plumbing. Real quality numbers
come from `uv run evaluate --searcher vector` with nomic-embed-text.
"""

import os
import zlib

import pytest

from rag.ingest import ChunkStore, build_chunks, load_documents
from rag.vector import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    embed_chunks,
    to_pgvector,
    vector_searcher,
)

store = ChunkStore()
pytestmark = pytest.mark.skipif(
    not store.is_up() and not os.getenv("CI"),
    reason="Postgres not running (cd 00-foundations && docker compose up -d)",
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 768
            for word in text.lower().split():
                vector[zlib.crc32(word.strip(".,?").encode()) % 768] += 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def embedder():
    store.init_schema()
    store.replace_all(build_chunks(load_documents(), "headings", 120, 20), "headings")
    fake = FakeEmbedder()
    embed_chunks(store, fake, batch_size=10)
    return fake


def test_pgvector_text_format():
    assert to_pgvector([0.5, -1.0, 2.0]) == "[0.5,-1.0,2.0]"


def test_every_chunk_is_embedded_once(embedder):
    corpus = store.corpus()
    assert corpus["embedded"] == corpus["chunks"] > 0
    assert sum(len(batch) for batch in embedder.calls) == corpus["chunks"]
    assert all(len(batch) <= 10 for batch in embedder.calls)  # batched, not one giant request

    assert embed_chunks(store, embedder) == 0  # nothing left to do


def test_documents_are_embedded_with_prefix_and_heading_trail(embedder):
    first = embedder.calls[0][0]
    assert first.startswith(DOCUMENT_PREFIX)
    assert " > " in first or "\n" in first  # the heading trail came along with the body


def test_question_uses_the_query_prefix(embedder):
    vector_searcher(store, embedder)("Are pets allowed?", "ruaka-rentals", 3)
    assert embedder.calls[-1] == [QUERY_PREFIX + "Are pets allowed?"]


def test_tenant_filter_only_returns_that_tenants_chunks(embedder):
    results = vector_searcher(store, embedder)("How much is the deposit?", "ruaka-rentals", 5)

    assert len(results) == 5
    assert {c.tenant_id for c in results} == {"ruaka-rentals"}


def test_closest_chunk_comes_first(embedder):
    question = "Subletting is not allowed. Only the people named on the tenancy agreement"
    [top] = vector_searcher(store, embedder)(question, "ruaka-rentals", 1)
    assert top.source == "tenancy-policy.md"
    assert top.section.endswith("Subletting")
