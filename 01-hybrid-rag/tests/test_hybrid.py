"""Reciprocal Rank Fusion is pure arithmetic on ranks, so no database or model is needed."""

import pytest

from rag.chunking import Chunk
from rag.hybrid import hybrid_searcher, rrf


def chunk(name: str) -> Chunk:
    return Chunk(tenant_id="t1", source="doc.md", section=name, index=ord(name), text=name)


A, B, C, D = (chunk(n) for n in "ABCD")


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.section for c in chunks]


def test_a_chunk_ranked_well_by_both_beats_a_chunk_first_in_only_one():
    vector = [A, C, B]
    keyword = [D, C, B]
    # C is 2nd in both lists: 2/62 = 0.032 beats A or D at 1st in one list: 1/61 = 0.016
    assert ids(rrf([vector, keyword]))[0] == "C"


def test_scores_add_up_across_lists():
    fused = rrf([[A, B], [B, A]], rrf_k=0)  # both get 1/1 + 1/2: a tie
    assert set(ids(fused)) == {"A", "B"}
    assert ids(rrf([[A, B], [B]], rrf_k=0)) == ["B", "A"]  # B: 1/2 + 1/1 beats A: 1/1


def test_chunks_found_by_only_one_searcher_are_kept():
    assert set(ids(rrf([[A], [B], [C]]))) == {"A", "B", "C"}


def test_empty_lists_are_fine():
    assert rrf([]) == []
    assert ids(rrf([[], [A]])) == ["A"]  # keyword search often finds nothing


def test_same_chunk_is_never_returned_twice():
    # A: 1/61 + 1/61 + 1/62 beats B: 1/62 + 1/62 + 1/61, and each appears once
    assert ids(rrf([[A, B], [A, B], [B, A]])) == ["A", "B"]


def test_weights_let_one_list_dominate():
    vector, keyword = [A, B], [B, A]
    assert ids(rrf([vector, keyword], weights=[1.0, 0.5])) == ["A", "B"]
    assert ids(rrf([vector, keyword], weights=[0.5, 1.0])) == ["B", "A"]


def test_weights_must_match_the_number_of_lists():
    with pytest.raises(ValueError, match="2 rankings but 1 weights"):
        rrf([[A], [B]], weights=[1.0])


@pytest.mark.parametrize("k", [1, 2])
def test_hybrid_asks_each_searcher_for_candidates_then_cuts_to_k(k):
    asked = []

    def fake(results):
        def search(question, tenant_id, n):
            asked.append(n)
            return results[:n]

        return search

    search = hybrid_searcher([fake([A, B, C]), fake([C, D])], candidates=20)
    results = search("q", "t1", k)

    assert asked == [20, 20]  # deep lists, so fusion has material to work with
    assert len(results) == k
    assert results[0] == C  # in both lists
