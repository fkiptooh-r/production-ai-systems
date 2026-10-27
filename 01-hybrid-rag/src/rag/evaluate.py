"""Score a retriever against the eval set in eval/questions.jsonl.

A retrieved chunk is *relevant* to a question when it comes from the question's tenant and
source document AND contains the answer phrase. Labels are phrases, not chunk ids, so the
same eval set works for every chunking strategy and size.

Metrics (over answerable questions only; not-in-data questions are for M7's "I don't know"):
- Recall@k: share of questions with at least one relevant chunk in the top k.
  With one answer phrase per question this equals the "hit rate".
- MRR: mean of 1/rank of the first relevant chunk (0 if none in the top k).
  Rewards putting the answer at #1 rather than #5.

Usage:
  uv run evaluate --searcher random [--strategy headings|fixed] [--size 120] [--overlap 20]
  uv run evaluate --searcher vector [--all-tenants]   # uses whatever `uv run ingest` stored
"""

import argparse
import json
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Chunk, clean
from rag.ingest import PROJECT_DIR, ChunkStore, build_chunks, load_documents

QUESTIONS_FILE = PROJECT_DIR / "eval" / "questions.jsonl"
KINDS = ("direct", "paraphrase", "exact", "not_in_data")

# search(question, tenant_id, k) -> the k best chunks, best first
Searcher = Callable[[str, str, int], list[Chunk]]


@dataclass(frozen=True)
class Question:
    id: str
    tenant: str
    kind: str  # one of KINDS
    question: str
    source: str | None  # document holding the answer; None if the answer isn't in the data
    answer: str | None  # phrase a relevant chunk must contain
    absent: str | None  # not-in-data only: a term that must NOT appear in the tenant's docs

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    lines = path.read_text("utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def normalize(text: str) -> str:
    return clean(text).lower()


def is_relevant(chunk: Chunk, question: Question) -> bool:
    return (
        question.answerable
        and chunk.tenant_id == question.tenant
        and chunk.source == question.source
        and normalize(question.answer) in normalize(chunk.full_text)
    )


def first_relevant_rank(results: list[Chunk], question: Question) -> int | None:
    """1-based rank of the first relevant chunk, or None if there isn't one."""
    for rank, chunk in enumerate(results, start=1):
        if is_relevant(chunk, question):
            return rank
    return None


@dataclass
class Report:
    k: int
    questions: int
    recall: float
    mrr: float
    p50_ms: float = 0.0  # median time for one search call
    by_kind: dict[str, tuple[int, float, float]] = field(default_factory=dict)  # n, recall, mrr
    misses: list[str] = field(default_factory=list)  # ids with nothing relevant in the top k


def _score(ranks: list[int | None]) -> tuple[float, float]:
    if not ranks:
        return 0.0, 0.0
    recall = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return recall, mrr


def evaluate(search: Searcher, questions: list[Question], k: int = 5) -> Report:
    answerable = [q for q in questions if q.answerable]
    ranks: dict[str, int | None] = {}
    timings: list[float] = []
    for q in answerable:
        started = time.perf_counter()
        results = search(q.question, q.tenant, k)[:k]
        timings.append(time.perf_counter() - started)
        ranks[q.id] = first_relevant_rank(results, q)

    recall, mrr = _score(list(ranks.values()))
    p50_ms = statistics.median(timings) * 1000 if timings else 0.0
    report = Report(k=k, questions=len(answerable), recall=recall, mrr=mrr, p50_ms=p50_ms)
    for kind in KINDS:
        kind_ranks = [ranks[q.id] for q in answerable if q.kind == kind]
        if kind_ranks:
            report.by_kind[kind] = (len(kind_ranks), *_score(kind_ranks))
    report.misses = [qid for qid, rank in ranks.items() if rank is None]
    return report


def random_searcher(chunks: list[Chunk], seed: int = 0) -> Searcher:
    """The floor: k chunks picked at random from all tenants. Any real retriever must beat it."""
    rng = random.Random(seed)

    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        return rng.sample(chunks, min(k, len(chunks)))

    return search


def print_report(name: str, report: Report) -> None:
    print(
        f"{name}: Recall@{report.k} {report.recall:.2f} | MRR {report.mrr:.2f}"
        f" | p50 {report.p50_ms:.0f} ms ({report.questions} answerable questions)\n"
    )
    print(f"{'kind':<12} {'n':>3} {'recall':>7} {'mrr':>5}")
    for kind, (n, recall, mrr) in report.by_kind.items():
        print(f"{kind:<12} {n:>3} {recall:>7.2f} {mrr:>5.2f}")
    print(f"\nmissed: {', '.join(report.misses) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a retriever against the eval set.")
    parser.add_argument("--searcher", choices=["random", "vector"], default="random")
    parser.add_argument("--k", type=int, default=5)
    random_args = parser.add_argument_group("random only (vector uses the ingested chunks)")
    random_args.add_argument("--strategy", choices=["headings", "fixed"], default="headings")
    random_args.add_argument("--size", type=int, default=120)
    random_args.add_argument("--overlap", type=int, default=20)
    vector_args = parser.add_argument_group("vector only")
    vector_args.add_argument(
        "--all-tenants", action="store_true", help="search every agency, not just the asker's"
    )
    args = parser.parse_args()

    if args.searcher == "random":
        chunks = build_chunks(load_documents(), args.strategy, args.size, args.overlap)
        search = random_searcher(chunks)
        name = f"random ({args.strategy}, {len(chunks)} chunks)"
    else:
        from shared.llm import LLMError

        from rag.vector import ollama_embedder, vector_searcher

        store = ChunkStore()
        corpus = store.corpus()
        if not corpus["embedded"]:
            raise SystemExit("No embedded chunks yet. Run: uv run ingest")
        search = vector_searcher(store, ollama_embedder(), tenant_filter=not args.all_tenants)
        try:
            search("warm-up: load the model before timing", "none", 1)
        except LLMError as exc:
            raise SystemExit(f"Embedding failed: {exc}") from exc
        scope = "all tenants" if args.all_tenants else "tenant filter"
        name = f"vector ({corpus['strategy']}, {corpus['embedded']} chunks, {scope})"

    print_report(name, evaluate(search, load_questions(), args.k))
