"""Hybrid search: run vector and keyword search, then merge the two ranked lists with RRF.

Reciprocal Rank Fusion gives each chunk 1 / (rrf_k + rank) from every list it appears in
and sorts by the sum. It only looks at *ranks*, never at the raw scores, which matters
because the scores aren't comparable: cosine distance (0 to 2, lower is better) and ts_rank
(0 to ~1, higher is better) live on different scales. Ranks are always 1, 2, 3...

rrf_k = 60 comes from the original paper (Cormack et al., 2009). It dampens the gap between
rank 1 and rank 2, so a chunk ranked well by *both* methods beats one ranked first by only
one of them:
    rank 1 in one list only:    1/61            = 0.0164
    rank 3 in both lists:       1/63 + 1/63     = 0.0317
"""

from collections.abc import Callable

from tracing.tracer import span

from rag.chunking import Chunk

Searcher = Callable[[str, str, int], list[Chunk]]


def rrf(
    rankings: list[list[Chunk]], rrf_k: int = 60, weights: list[float] | None = None
) -> list[Chunk]:
    """Fuse ranked lists (best first) into one, best first. Ties keep first-seen order.

    weights scale each list's vote (default: all 1.0). A weaker retriever can be given less
    say, e.g. [1.0, 0.5] so keyword search breaks ties rather than overruling vector search.
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (rrf_k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks[chunk_id] for chunk_id in order]


def hybrid_searcher(
    searchers: list[Searcher],
    *,
    candidates: int = 20,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> Searcher:
    """search(question, tenant_id, k): take the top `candidates` from each searcher, fuse, cut to k.

    Asking each searcher for more than k matters: a chunk at rank 8 in both lists can fuse
    into the top 5, but only if both lists went deep enough to include it.
    """

    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        with span("hybrid_search", question=question, k=k, candidates=candidates) as hs:
            rankings = [s(question, tenant_id, max(k, candidates)) for s in searchers]
            with span("fuse", rrf_k=rrf_k, weights=weights) as fs:
                fused = rrf(rankings, rrf_k, weights)[:k]
                fs.output = [c.chunk_id for c in fused]
            hs.output = fs.output
            return fused

    return search
