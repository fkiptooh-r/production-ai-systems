"""Score generated answers on the full eval set (all 60 questions, including not-in-data).

Metrics:
- Abstention accuracy: on not-in-data questions, did it say "I don't know"? (The other
  choice is a made-up answer, often borrowed from another agency's policy.)
- False "I don't know": on answerable questions, did it refuse when the answer was there?
- Grounded citation: did it cite at least one chunk that really contains the answer
  (right tenant, right document, contains the answer phrase)?
- Phrase in answer: does the answer text contain the exact answer phrase? A strict lower
  bound: "2 months" for "two months' rent" counts as a miss, so read the failures file.

Every answer is saved to eval/results/answers.jsonl for reading by hand.

Usage: uv run eval-answers [--model llama3.2:3b] [--limit 10]
Slow on CPU: roughly 10-15 s per question, so ~12-15 minutes for all 60.

For failure analysis (02 M4), any question file, searcher and top-k, saved elsewhere:
  uv run eval-answers --questions eval/hard_questions.jsonl --searcher hybrid+rerank
      --top-k 4 --out ../02-llm-tracing/data/runs/hard-rerank.jsonl
"""

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from shared.llm import LLMError
from tracing.tracer import span

from rag.chunking import Chunk
from rag.evaluate import (
    QUESTIONS_FILE,
    SEARCHERS,
    Question,
    _parser,
    is_relevant,
    load_questions,
    make_searcher,
    normalize,
)
from rag.generate import DEFAULT_MODEL, TOP_K, Answer, answer
from rag.ingest import PROJECT_DIR, ChunkStore
from rag.vector import row_to_chunk

RESULTS_FILE = PROJECT_DIR / "eval" / "results" / "answers.jsonl"


@dataclass
class Judged:
    question: Question
    answer: Answer

    @property
    def correct_abstention(self) -> bool:
        return not self.question.answerable and self.answer.abstained

    @property
    def false_abstention(self) -> bool:
        return self.question.answerable and self.answer.abstained

    @property
    def grounded(self) -> bool:
        return any(is_relevant(chunk, self.question) for chunk in self.answer.cited)

    @property
    def phrase_in_answer(self) -> bool:
        return self.question.answerable and normalize(self.question.answer) in normalize(
            self.answer.text
        )

    @property
    def passed(self) -> bool:
        if not self.question.answerable:
            return self.correct_abstention
        return self.grounded and not self.false_abstention


@dataclass
class AnswerReport:
    answerable: int = 0
    unanswerable: int = 0
    correct_abstentions: int = 0
    false_abstentions: int = 0
    grounded: int = 0
    phrase_in_answer: int = 0
    p50_seconds: float = 0.0
    failures: list[Judged] = field(default_factory=list)


def summarize(judged: list[Judged]) -> AnswerReport:
    answerable = [j for j in judged if j.question.answerable]
    report = AnswerReport(
        answerable=len(answerable),
        unanswerable=len(judged) - len(answerable),
        correct_abstentions=sum(j.correct_abstention for j in judged),
        false_abstentions=sum(j.false_abstention for j in answerable),
        grounded=sum(j.grounded for j in answerable),
        phrase_in_answer=sum(j.phrase_in_answer for j in answerable),
        failures=[j for j in judged if not j.passed],
    )
    if judged:
        report.p50_seconds = statistics.median(j.answer.seconds for j in judged)
    return report


def stored_chunks() -> list[Chunk]:
    """Every chunk currently in the database, i.e. exactly what retrieval searches."""
    with ChunkStore().connect() as conn:
        rows = conn.execute(
            "SELECT tenant_id, source, section, chunk_index, content FROM rag.chunks"
        ).fetchall()
    return [row_to_chunk(row) for row in rows]


def gold_ids(question: Question, chunks: list[Chunk]) -> list[str]:
    """Chunks that contain the answer: what retrieval should have found. Empty if unanswerable."""
    return [c.chunk_id for c in chunks if is_relevant(c, question)]


def to_record(j: Judged, gold: list[str] | None = None) -> dict:
    return {
        "id": j.question.id,
        "kind": j.question.kind,
        "tenant": j.question.tenant,
        "question": j.question.question,
        "expected": j.question.answer,
        "answer": j.answer.text,
        "gold": gold or [],  # for failure analysis: was it retrieved? cited? (02 M4, M5)
        "retrieved": [c.chunk_id for c in j.answer.retrieved],
        "cited": [c.chunk_id for c in j.answer.cited],
        "abstained": j.answer.abstained,
        "grounded": j.grounded,
        "passed": j.passed,
        "seconds": round(j.answer.seconds, 1),
        "trace_id": j.answer.trace_id,  # links each scored answer to its trace (M4, M5)
    }


def main() -> None:
    parser = _parser("Score generated answers.")  # shares the searcher options with evaluate
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N questions (quick check)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--searcher", choices=SEARCHERS[1:], default="vector")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="chunks shown to the model")
    parser.add_argument("--out", type=Path, default=RESULTS_FILE, help="where to save answers")
    args = parser.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    name, search = make_searcher(args, args.searcher)
    print(f"{args.model} · {name} · top {args.top_k} · {len(questions)} questions")
    judged = []
    started = time.perf_counter()
    for n, q in enumerate(questions, start=1):
        # One trace per question, so a model error or timeout still has a trace to look at.
        with span("eval_question", question_id=q.id, kind=q.kind) as s:
            try:
                result = answer(q.question, q.tenant, search, model=args.model, top_k=args.top_k)
            except LLMError as exc:  # a failure to record, not a reason to stop the run
                s.status, s.error = "error", f"LLMError: {exc}"
                result = Answer(question=q.question, text=f"[error] {exc}", retrieved=[])
            result.trace_id = s.trace_id
        judged.append(Judged(q, result))
        mark = "ok  " if judged[-1].passed else "FAIL"
        print(f"[{n:>2}/{len(questions)}] {mark} {q.id} ({result.seconds:.0f}s) {result.text[:70]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks = stored_chunks()
    records = [to_record(j, gold_ids(j.question, chunks)) for j in judged]
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), "utf-8")

    r = summarize(judged)
    print(f"\n{args.model}, {len(judged)} questions in {time.perf_counter() - started:.0f}s")
    if r.unanswerable:
        print(f"Abstention accuracy:   {r.correct_abstentions}/{r.unanswerable} not-in-data")
    if r.answerable:
        print(f"False 'I don't know':  {r.false_abstentions}/{r.answerable} answerable")
        print(f"Grounded citation:     {r.grounded}/{r.answerable}")
        print(f"Phrase in answer:      {r.phrase_in_answer}/{r.answerable} (strict lower bound)")
    print(f"p50 answer time:       {r.p50_seconds:.1f}s")
    print(f"Saved every answer to {args.out}")
    for j in r.failures:
        print(f"\nFAIL {j.question.id} [{j.question.kind}] {j.question.question}")
        print(f"  expected: {j.question.answer or I_DONT_KNOW_HINT}")
        print(f"  got:      {j.answer.text}")


I_DONT_KNOW_HINT = "I don't know (not in this agency's documents)"
