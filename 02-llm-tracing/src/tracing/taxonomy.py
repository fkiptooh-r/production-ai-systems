"""M4 · Failure taxonomy: every bad answer, the evidence from its trace, and a root cause.

    uv run failure-candidates   # eval runs + traces -> data/candidates.jsonl, evidence per answer
    uv run taxonomy             # data/failures.jsonl (labelled by hand) -> counts and a chart

The workflow is Hamel Husain's "look at your data": the evidence and a suggested label are
computed, but every answer is read and the final label is a human decision, recorded with a
one-line note. Suggested labels come from the trace, so they can be wrong in one direction
only: they never see *what the answer says*, which is why numeric and conditional answers
are always read in full.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tracing.view import load

DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
FAILURES_FILE = DATA_DIR / "failures.jsonl"
TRACE_FILE = Path("traces") / "m4.jsonl"

# Root causes, in pipeline order: the first stage that went wrong gets the blame.
LABELS = {
    "retrieval_miss": "the chunk holding the answer was never retrieved",
    "rerank_drop": "first-stage search found it; the reranker cut it from the top k",
    "wrong_refusal": 'said "I don\'t know" although the answer was in the retrieved chunks',
    "ignored_context": "the answer was retrieved, but the reply contradicts or misses it",
    "hallucination": "states something no retrieved chunk supports",
    "format_error": "right content, broken contract: missing or wrong citation",
    "timeout": "the model errored or didn't answer in time",
}


def first_stage(spans: list[dict]) -> list[str] | None:
    """Candidate ids before reranking, from the trace; None when there was no reranker."""
    rerank = next((s for s in spans if s["name"] == "rerank_search"), None)
    if rerank is None:
        return None
    pool = next(
        (s for s in spans if s["parent_id"] == rerank["span_id"] and s["name"] != "rerank"), None
    )
    return pool["output"] if pool else None


def evidence(record: dict, spans: list[dict]) -> dict:
    gold = set(record.get("gold", []))
    pool = first_stage(spans)
    errors = [s["error"] for s in spans if s["status"] == "error"]
    return {
        "answerable": record["expected"] is not None,
        "gold": sorted(gold),
        "gold_in_first_stage": bool(gold & set(pool)) if pool is not None else None,
        "gold_retrieved": bool(gold & set(record["retrieved"])),
        "gold_cited": bool(gold & set(record["cited"])),
        "cited_anything": bool(record["cited"]),
        "abstained": record["abstained"],
        "error": errors[0] if errors else None,
    }


def suggest(ev: dict) -> str | None:
    """A first guess from the evidence alone. None: can't tell without reading the answer."""
    if ev["error"]:
        return "timeout"
    if not ev["answerable"]:
        return None if ev["abstained"] else "hallucination"
    if not ev["gold_retrieved"]:
        return "rerank_drop" if ev["gold_in_first_stage"] else "retrieval_miss"
    if ev["abstained"]:
        return "wrong_refusal"
    if not ev["cited_anything"]:
        return "format_error"
    if not ev["gold_cited"]:
        return "ignored_context"
    return None  # retrieved, cited, answered: right or wrong depends on what it says


def candidates(runs_dir: Path, trace_file: Path) -> list[dict]:
    traces = load(trace_file)
    out = []
    for run_file in sorted(runs_dir.glob("*.jsonl")):
        for line in run_file.read_text("utf-8").splitlines():
            record = json.loads(line)
            ev = evidence(record, traces.get(record["trace_id"], []))
            out.append(
                {
                    "run": run_file.stem,
                    **{k: record[k] for k in ("id", "kind", "tenant", "question", "expected")},
                    "answer": record["answer"],
                    "auto_passed": record["passed"],
                    "suggested": suggest(ev),
                    "evidence": ev,
                    "trace_id": record["trace_id"],
                }
            )
    return out


def bar(n: int, top: int, width: int = 30) -> str:
    return "█" * max(1, round(n / top * width)) if n else ""


def summary(failures: list[dict], total: int | None = None) -> list[str]:
    counts = Counter(f["label"] for f in failures)
    top = max(counts.values(), default=1)
    lines = [f"{len(failures)} bad answers" + (f" out of {total}" if total else "")]
    lines.append("")
    for label, n in counts.most_common():
        lines.append(f"{label:<16}{n:>4}  {bar(n, top)}")
    per_run: dict[str, Counter] = defaultdict(Counter)
    for f in failures:
        per_run[f["run"]][f["label"]] += 1
    lines.append("")
    for run, c in sorted(per_run.items()):
        lines.append(
            f"{run:<22}{sum(c.values()):>4}  " + ", ".join(f"{k} {v}" for k, v in c.most_common())
        )
    per_kind = Counter(f["kind"] for f in failures)
    lines.append("")
    lines.append("by question kind: " + ", ".join(f"{k} {v}" for k, v in per_kind.most_common()))
    return lines


def main_candidates() -> None:
    parser = argparse.ArgumentParser(description="Evidence for every answer in the eval runs.")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--traces", type=Path, default=TRACE_FILE)
    parser.add_argument("--out", type=Path, default=CANDIDATES_FILE)
    args = parser.parse_args()
    rows = candidates(args.runs, args.traces)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    flagged = sum(1 for r in rows if r["suggested"] or not r["auto_passed"])
    print(f"{len(rows)} answers, {flagged} flagged by the evidence. Read all of them: {args.out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Count labelled failures by root cause.")
    parser.add_argument("--file", type=Path, default=FAILURES_FILE)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_FILE)
    args = parser.parse_args()
    failures = [json.loads(line) for line in args.file.read_text("utf-8").splitlines() if line]
    unknown = {f["label"] for f in failures} - set(LABELS)
    if unknown:
        sys.exit(f"Unknown labels {unknown}; use one of {list(LABELS)}")
    total = None
    if args.candidates.exists():
        total = sum(1 for line in args.candidates.read_text("utf-8").splitlines() if line)
    print("\n".join(summary(failures, total)))
