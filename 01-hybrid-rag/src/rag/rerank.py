"""Second-stage reranking with a cross-encoder.

Vector search embeds the question and each chunk *separately* and compares the vectors:
fast, so it can scan everything, but the two texts never "see" each other. A cross-encoder
reads question + chunk *together* through one transformer and outputs a relevance score:
much more precise, but one model run per chunk, so it only re-orders a short candidate list.

    question -> first-stage search (top 20, fast) -> cross-encoder scores 20 pairs -> top k

The model (ms-marco-MiniLM-L-6-v2, ~80 MB) runs on CPU with ONNX Runtime via fastembed,
so there's no PyTorch. It's downloaded on first use into models/ (gitignored).
"""

from collections.abc import Callable

from tracing.tracer import span

from rag.chunking import Chunk
from rag.ingest import PROJECT_DIR

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODELS_DIR = PROJECT_DIR / "models"

Searcher = Callable[[str, str, int], list[Chunk]]
# (question, texts) -> one relevance score per text, higher = more relevant. Swappable in tests.
Scorer = Callable[[str, list[str]], list[float]]


def cross_encoder_scorer(model: str = DEFAULT_RERANK_MODEL) -> Scorer:
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # slow import, so only here

    encoder = TextCrossEncoder(model_name=model, cache_dir=str(MODELS_DIR))
    return lambda question, texts: list(encoder.rerank(question, texts)) if texts else []


def reranking_searcher(first_stage: Searcher, scorer: Scorer, *, candidates: int = 20) -> Searcher:
    """search(question, tenant_id, k): take `candidates` from first_stage, rescore, keep k."""

    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        with span("rerank_search", question=question, k=k, candidates=candidates) as rs:
            pool = first_stage(question, tenant_id, max(k, candidates))
            with span("rerank", candidates=len(pool)) as s:
                # Score full_text: the heading trail ("... > Pets") is often the strongest clue.
                scores = scorer(question, [chunk.full_text for chunk in pool])
                order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)  # stable
                kept = [pool[i] for i in order[:k]]
                # The first stage's output is the "before": M5 asks "did gold survive reranking?"
                s.output = [c.chunk_id for c in kept]
            rs.output = s.output
            return kept

    return search
