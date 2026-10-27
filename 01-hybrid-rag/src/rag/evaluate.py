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
  uv run evaluate --searcher vector                   # uses whatever `uv run ingest` stored
  uv run evaluate --searcher keyword [--mode any|all]
  uv run evaluate --searcher hybrid [--rrf-k 60] [--candidates 20]
  uv run evaluate --searcher vector+rerank            # also keyword+rerank, hybrid+rerank
  uv run compare [--searchers vector keyword hybrid]  # question by question
"""

import argparse
import functools
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
SEARCHERS = ["random", "vector", "keyword", "hybrid"]
SEARCHERS += [f"{s}+rerank" for s in SEARCHERS[1:]]  # second stage on top of any real search

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
    ranks: dict[str, int | None] = field(default_factory=dict)  # id -> first relevant rank


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
    report.ranks = ranks
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


@functools.cache
def _scorer(model: str):
    """The cross-encoder, loaded once per run even when several searchers rerank."""
    from rag.rerank import cross_encoder_scorer

    return cross_encoder_scorer(model)


def make_searcher(args: argparse.Namespace, searcher: str) -> tuple[str, Searcher]:
    """Build a searcher from command-line options. Returns (display name, search function)."""
    if searcher == "random":
        chunks = build_chunks(load_documents(), args.strategy, args.size, args.overlap)
        return f"random ({args.strategy}, {len(chunks)} chunks)", random_searcher(chunks)

    if searcher.endswith("+rerank"):
        from rag.rerank import reranking_searcher

        name, first_stage = make_searcher(args, searcher.removesuffix("+rerank"))
        scorer = _scorer(args.rerank_model)
        search = reranking_searcher(first_stage, scorer, candidates=args.candidates)
        search("warm-up: load the reranker before timing", "none", 1)
        model = args.rerank_model.split("/")[-1]
        return f"{name} + {model} top {args.candidates}", search

    store = ChunkStore()
    corpus = store.corpus()
    if not corpus["chunks"]:
        raise SystemExit("No chunks stored yet. Run: uv run ingest")
    details = f"({corpus['strategy']}, {corpus['chunks']} chunks)"

    def keyword() -> Searcher:
        from rag.keyword import keyword_searcher

        return keyword_searcher(store, mode=args.mode)

    def vector() -> Searcher:
        from shared.llm import LLMError

        from rag.vector import ollama_embedder, vector_searcher

        if not corpus["embedded"]:
            raise SystemExit("No embedded chunks yet. Run: uv run ingest")
        search = vector_searcher(store, ollama_embedder())
        try:
            search("warm-up: load the model before timing", "none", 1)
        except LLMError as exc:
            raise SystemExit(f"Embedding failed: {exc}") from exc
        return search

    if searcher == "keyword":
        return f"keyword-{args.mode} {details}", keyword()
    if searcher == "vector":
        return f"vector {details}", vector()

    from rag.hybrid import hybrid_searcher

    search = hybrid_searcher(
        [vector(), keyword()],
        candidates=args.candidates,
        rrf_k=args.rrf_k,
        weights=[1.0, args.keyword_weight],
    )
    return f"hybrid rrf_k={args.rrf_k} kw={args.keyword_weight} {details}", search


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["any", "all"], default="any", help="keyword: match any or all words"
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="hybrid: RRF damping constant")
    parser.add_argument(
        "--candidates", type=int, default=20, help="hybrid: results taken from each searcher"
    )
    parser.add_argument(
        "--keyword-weight", type=float, default=1.0, help="hybrid: keyword's vote (vector = 1)"
    )
    parser.add_argument(
        "--rerank-model",
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        help="+rerank: any fastembed cross-encoder, e.g. jinaai/jina-reranker-v1-turbo-en",
    )
    random_args = parser.add_argument_group("random only (others use the ingested chunks)")
    random_args.add_argument("--strategy", choices=["headings", "fixed"], default="headings")
    random_args.add_argument("--size", type=int, default=120)
    random_args.add_argument("--overlap", type=int, default=20)
    return parser


def main() -> None:
    parser = _parser("Score a retriever against the eval set.")
    parser.add_argument("--searcher", choices=SEARCHERS, default="random")
    args = parser.parse_args()

    name, search = make_searcher(args, args.searcher)
    print_report(name, evaluate(search, load_questions(), args.k))


def compare() -> None:
    """uv run compare: several searchers side by side, listing the questions they disagree on."""
    parser = _parser("Compare searchers question by question.")
    parser.add_argument(
        "--searchers", nargs="+", choices=SEARCHERS, default=["vector", "keyword", "hybrid"]
    )
    args = parser.parse_args()
    questions = load_questions()

    reports = {}
    for searcher in args.searchers:
        name, search = make_searcher(args, searcher)
        reports[searcher] = evaluate(search, questions, args.k)
        print(
            f"{name}: Recall@{args.k} {reports[searcher].recall:.2f}"
            f" | MRR {reports[searcher].mrr:.2f} | p50 {reports[searcher].p50_ms:.0f} ms"
        )

    def show(rank: int | None) -> str:
        return str(rank) if rank else "-"

    print(f"\nQuestions where the ranks differ (first right chunk; - = not in top {args.k})")
    print(f"{'id':<6} {'kind':<11}" + "".join(f"{s:>9}" for s in args.searchers) + "  question")
    for q in questions:
        if not q.answerable:
            continue
        ranks = [reports[s].ranks[q.id] for s in args.searchers]
        if len(set(ranks)) > 1:
            print(
                f"{q.id:<6} {q.kind:<11}"
                + "".join(f"{show(r):>9}" for r in ranks)
                + f"  {q.question}"
            )
