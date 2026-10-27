"""Reranking logic with a fake scorer, so no model download is needed (CI stays fast)."""

from rag.chunking import Chunk
from rag.rerank import reranking_searcher


def chunk(name: str, text: str) -> Chunk:
    return Chunk(tenant_id="t1", source="doc.md", section=name, index=0, text=text)


PETS = chunk("Pets", "No pets are allowed. This includes cats.")
RENT = chunk("Rent", "Rent is due on the 1st.")
NOTICE = chunk("Notice", "Give one month's notice.")


def word_overlap(question: str, texts: list[str]) -> list[float]:
    """Fake cross-encoder: counts question words that appear in the text."""
    words = set(question.lower().replace("?", "").split())
    return [float(len(words & set(t.lower().replace(".", "").split()))) for t in texts]


def first_stage(results: list[Chunk], asked: list[int] | None = None):
    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        if asked is not None:
            asked.append(k)
        return results[:k]

    return search


def test_reranker_moves_the_best_match_to_the_top():
    search = reranking_searcher(first_stage([RENT, NOTICE, PETS]), word_overlap)
    assert search("are cats allowed?", "t1", 1) == [PETS]


def test_reranker_asks_for_candidates_then_cuts_to_k():
    asked = []
    search = reranking_searcher(
        first_stage([RENT, NOTICE, PETS], asked), word_overlap, candidates=20
    )
    assert len(search("are cats allowed?", "t1", 2)) == 2
    assert asked == [20]


def test_scorer_sees_the_heading_trail():
    seen = []

    def recording(question, texts):
        seen.extend(texts)
        return [0.0] * len(texts)

    reranking_searcher(first_stage([PETS]), recording)("q", "t1", 1)
    assert seen == ["Pets\nNo pets are allowed. This includes cats."]


def test_ties_keep_the_first_stage_order():
    search = reranking_searcher(first_stage([RENT, NOTICE]), lambda q, texts: [1.0, 1.0])
    assert search("q", "t1", 2) == [RENT, NOTICE]


def test_empty_first_stage_returns_nothing():
    calls = []
    search = reranking_searcher(first_stage([]), lambda q, t: calls.append(t) or [])
    assert search("q", "t1", 5) == []
