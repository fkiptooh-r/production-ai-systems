"""Score listing extraction against hand-labelled examples.

Usage: uv run extract-eval [--model llama3.2:3b] [--file eval/listings.jsonl]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from foundations.extract import ExtractionError, Listing, extract_listing
from foundations.llm import DEFAULT_MODEL, LLMError

# amenities is free text, so it isn't scored exactly. Improving that is a good exercise.
SCORED_FIELDS = ["property_type", "bedrooms", "price_per_month", "location", "parking", "furnished"]

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "eval" / "listings.jsonl"


def field_matches(expected: object, actual: object) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def score(expected: dict, listing: Listing | None) -> dict[str, bool]:
    """Per-field correctness; a failed extraction counts as wrong on every field."""
    actual = listing.model_dump() if listing else {}
    return {f: listing is not None and field_matches(expected[f], actual[f]) for f in SCORED_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate listing extraction accuracy.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    examples = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    correct = {f: 0 for f in SCORED_FIELDS}
    valid = 0
    started = time.perf_counter()

    for i, example in enumerate(examples, 1):
        try:
            listing = extract_listing(example["text"], model=args.model)
            valid += 1
        except (ExtractionError, LLMError) as exc:
            listing = None
            print(f"[{i}] FAILED: {exc}")

        results = score(example["expected"], listing)
        for field, ok in results.items():
            correct[field] += ok
        misses = [f for f, ok in results.items() if not ok]
        if listing and misses:
            got = listing.model_dump()
            details = ", ".join(
                f"{f}: expected {example['expected'][f]!r} got {got[f]!r}" for f in misses
            )
            print(f"[{i}] {details}")
        elif listing:
            print(f"[{i}] all fields correct")

    n = len(examples)
    total_correct = sum(correct.values())
    print(f"\n--- {args.model} on {n} listings ({time.perf_counter() - started:.0f}s) ---")
    print(f"{'valid output':<16} {valid}/{n}")
    for field in SCORED_FIELDS:
        print(f"{field:<16} {correct[field]}/{n}  {correct[field] / n:.0%}")
    print(f"{'overall':<16} {total_correct / (n * len(SCORED_FIELDS)):.0%}")
