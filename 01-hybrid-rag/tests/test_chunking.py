import pytest

from rag.chunking import chunk_document, clean, split_sections, split_words
from rag.ingest import build_chunks, load_documents

DOC = """# Policy

Intro line.

## Deposit

Two months'   rent.

### Refunds

Within 30 days.

## Pets

Cats allowed.
"""


def test_sections_carry_the_full_heading_trail():
    sections = split_sections(DOC)
    assert [s.path for s in sections] == [
        "Policy",
        "Policy > Deposit",
        "Policy > Deposit > Refunds",
        "Policy > Pets",  # back up a level: "Deposit > Refunds" is closed
    ]
    assert sections[1].text == "Two months' rent."  # whitespace cleaned


def test_split_words_overlaps_and_covers_every_word():
    words = [f"w{i}" for i in range(25)]
    windows = split_words(" ".join(words), size=10, overlap=3)

    assert [len(w.split()) for w in windows] == [10, 10, 10, 4]
    assert windows[0].split()[-3:] == windows[1].split()[:3]  # 3 shared words
    assert {w for window in windows for w in window.split()} == set(words)


def test_short_text_is_one_chunk_and_empty_text_is_none():
    assert split_words("just a few words", size=10, overlap=2) == ["just a few words"]
    assert split_words("   ", size=10, overlap=2) == []


@pytest.mark.parametrize("overlap", [10, 11, -1])
def test_overlap_must_be_smaller_than_size(overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size=10, overlap=overlap)


def test_headings_strategy_keeps_section_and_ids():
    chunks = chunk_document(DOC, tenant_id="t1", source="policy.md")

    assert [c.section for c in chunks][1] == "Policy > Deposit"
    assert [c.chunk_id for c in chunks][:2] == ["t1/policy.md#0", "t1/policy.md#1"]


def test_fixed_strategy_strips_heading_markers_but_keeps_heading_words():
    chunks = chunk_document(DOC, tenant_id="t1", source="policy.md", strategy="fixed", size=50)

    assert len(chunks) == 1
    assert "#" not in chunks[0].text
    assert chunks[0].text.startswith("Policy Intro line. Deposit")
    assert chunks[0].section == "Policy"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        chunk_document(DOC, tenant_id="t1", source="x.md", strategy="sentences")


# --- Properties checked against the real corpus ---

DOCUMENTS = load_documents()


def test_corpus_has_three_tenants_with_four_documents_each():
    tenants = {}
    for doc in DOCUMENTS:
        tenants.setdefault(doc.tenant_id, []).append(doc.source)
    assert {t: len(s) for t, s in tenants.items()} == {
        "nyumba-homes": 4,
        "pwani-lettings": 4,
        "ruaka-rentals": 4,
    }


def test_full_text_puts_the_heading_trail_before_the_body():
    chunk = chunk_document(DOC, tenant_id="t1", source="policy.md")[1]
    assert chunk.full_text == "Policy > Deposit\nTwo months' rent."


@pytest.mark.parametrize("strategy", ["headings", "fixed"])
def test_no_word_is_lost_and_no_chunk_is_too_big(strategy):
    # Regression: the first version dropped heading words ("Getting around", agency names)
    # from searchable text under the headings strategy; full_text keeps them.
    chunks = build_chunks(DOCUMENTS, strategy, size=60, overlap=10)

    for doc in DOCUMENTS:
        doc_chunks = [c for c in chunks if c.source == doc.source and c.tenant_id == doc.tenant_id]
        chunk_words = {w for c in doc_chunks for w in c.full_text.split()}
        body_words = set(clean(doc.text.replace("#", " ")).split())
        assert body_words <= chunk_words, f"{strategy} lost words in {doc.tenant_id}/{doc.source}"

    assert all(0 < c.word_count <= 60 for c in chunks)  # size limits the body only
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # ids are unique
