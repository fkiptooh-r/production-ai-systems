"""Score the model's FIRST decision: which tool (if any), with which arguments.

Only the first step is scored. It's where most mistakes happen, and it keeps the
eval fast on a CPU.

Usage: uv run tool-eval [--model llama3.2:3b]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError
from shared.llm import DEFAULT_MODEL, LLMError, stream_chat

from foundations.agent import SYSTEM_PROMPT
from foundations.tools import TOOLS, tool_schemas

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "eval" / "tool_calls.jsonl"


def normalise_args(tool: str, arguments: dict) -> dict | None:
    """Validated, comparable arguments: unset filters dropped, text lowercased.

    Returns None if the arguments are invalid for the tool.
    """
    try:
        args = TOOLS[tool].args_model.model_validate(arguments)
    except ValidationError:
        return None
    return {
        key: value.strip().lower() if isinstance(value, str) else value
        for key, value in args.model_dump(exclude_none=True).items()
    }


def grade(expected: dict, tool_calls: list[dict]) -> tuple[bool, bool]:
    """(right tool chosen, right arguments) for the model's first call."""
    if expected["tool"] is None:
        no_call = not tool_calls
        return no_call, no_call
    if not tool_calls:
        return False, False

    call = tool_calls[0]["function"]
    if call["name"] != expected["tool"]:
        return False, False
    got = normalise_args(call["name"], call.get("arguments") or {})
    return True, got == normalise_args(expected["tool"], expected["arguments"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool selection and arguments.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    examples = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    right_tool = right_args = 0
    started = time.perf_counter()

    for i, example in enumerate(examples, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        try:
            result = stream_chat(
                messages, model=args.model, tools=tool_schemas(), options={"temperature": 0}
            )
            calls = result.tool_calls
        except LLMError as exc:
            print(f"[{i}] FAILED: {exc}")
            continue

        tool_ok, args_ok = grade(example, calls)
        right_tool += tool_ok
        right_args += args_ok
        got = [(c["function"]["name"], c["function"].get("arguments")) for c in calls] or "no tool"
        mark = "ok" if args_ok else "WRONG"
        print(f"[{i}] {mark:<5} {example['question']}\n      got: {got}")
        if not args_ok:
            print(f"      expected: {example['tool'] or 'no tool'} {example['arguments'] or ''}")

    n = len(examples)
    print(f"\n--- {args.model} on {n} questions ({time.perf_counter() - started:.0f}s) ---")
    print(f"right tool (or correctly no tool)  {right_tool}/{n}  {right_tool / n:.0%}")
    print(f"right tool AND right arguments     {right_args}/{n}  {right_args / n:.0%}")
