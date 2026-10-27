"""Score a retriever against the eval set in eval/questions.jsonl.

A retrieved chunk is *relevant* to a question when it comes from the question's tenant and
source document AND contains the answer phrase. Labels are phrases, not chunk ids, so the
same eval set works for every chunking strategy and size.

Metrics (over answerable questions only; not-in-data questions are for M7's "I don't know"):
- Recall@k: share of questions with at least one relevant chunk in the top k.
  With one answer phrase per question this equals the "hit rate".
- MRR: mean of 1/rank of the first relevant chunk (0 if none in the top k).
  Rewards putting the answer at #1 rather than #5.

Usage: uv run evaluate [--searcher random] [--strategy headings|fixed] [--k 5]
"""

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Chunk, clean
from rag.ingest import PROJECT_DIR, build_chunks, load_documents

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
    ranks = {q.id: first_relevant_rank(search(q.question, q.tenant, k)[:k], q) for q in answerable}

    recall, mrr = _score(list(ranks.values()))
    report = Report(k=k, questions=len(answerable), recall=recall, mrr=mrr)
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
        f" ({report.questions} answerable questions)\n"
    )
    print(f"{'kind':<12} {'n':>3} {'recall':>7} {'mrr':>5}")
    for kind, (n, recall, mrr) in report.by_kind.items():
        print(f"{kind:<12} {n:>3} {recall:>7.2f} {mrr:>5.2f}")
    print(f"\nmissed: {', '.join(report.misses) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a retriever against the eval set.")
    parser.add_argument("--searcher", choices=["random"], default="random")
    parser.add_argument("--strategy", choices=["headings", "fixed"], default="headings")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    chunks = build_chunks(load_documents(), args.strategy, args.size, args.overlap)
    search = random_searcher(chunks)
    print_report(
        f"{args.searcher} ({args.strategy}, {len(chunks)} chunks)",
        evaluate(search, load_questions(), args.k),
    )
