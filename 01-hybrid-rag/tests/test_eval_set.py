"""Checks on the eval set itself. A wrong label makes every later score meaningless."""

from collections import Counter

import pytest

from rag.evaluate import KINDS, is_relevant, load_questions, normalize
from rag.ingest import build_chunks, load_documents

QUESTIONS = load_questions()
DOCUMENTS = load_documents()
DOCS = {(d.tenant_id, d.source): normalize(d.text) for d in DOCUMENTS}
TENANTS = {d.tenant_id for d in DOCUMENTS}


def test_ids_are_unique_and_fields_are_valid():
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert q.tenant in TENANTS, q.id
        assert q.kind in KINDS, q.id
        if q.kind == "not_in_data":
            assert q.source is None and q.answer is None and q.absent, q.id
        else:
            assert q.source and q.answer and q.absent is None, q.id


def test_every_tenant_has_a_balanced_share():
    per_tenant = Counter(q.tenant for q in QUESTIONS)
    assert len(QUESTIONS) >= 50
    assert all(n >= 15 for n in per_tenant.values()), per_tenant

    not_in_data = Counter(q.tenant for q in QUESTIONS if not q.answerable)
    assert set(not_in_data) == TENANTS  # every tenant gets some "I don't know" questions


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q.answerable], ids=lambda q: q.id)
def test_answer_phrase_is_in_the_source_document(q):
    assert normalize(q.answer) in DOCS[(q.tenant, q.source)]


@pytest.mark.parametrize("q", [q for q in QUESTIONS if not q.answerable], ids=lambda q: q.id)
def test_not_in_data_term_is_missing_from_every_tenant_document(q):
    for (tenant, source), text in DOCS.items():
        if tenant == q.tenant:
            assert normalize(q.absent) not in text, f"{q.id}: '{q.absent}' found in {source}"


@pytest.mark.parametrize(
    ("strategy", "size", "overlap"), [("headings", 120, 20), ("fixed", 80, 15), ("fixed", 50, 10)]
)
def test_every_answer_survives_chunking(strategy, size, overlap):
    # If an answer phrase were split across two chunks, no retriever could ever find it and
    # the question would count as a miss for reasons that have nothing to do with search.
    chunks = build_chunks(DOCUMENTS, strategy, size, overlap)
    unreachable = [
        q.id for q in QUESTIONS if q.answerable and not any(is_relevant(c, q) for c in chunks)
    ]
    assert unreachable == []
