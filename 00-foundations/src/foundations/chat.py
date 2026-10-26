"""Command line chat with a local Ollama model.

Usage: uv run chat "your question" [--model llama3.2:3b]
"""

import argparse
import logging
import sys

from shared.costs import estimate_cost
from shared.llm import DEFAULT_MODEL, LLMError, stream_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a reply from a local Ollama model.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0, help="max seconds between chunks")
    parser.add_argument("--retries", type=int, default=3)
    # Local models are free. These example prices show what a paid API would charge.
    parser.add_argument("--input-price", type=float, default=3.0, help="USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, default=15.0, help="USD per 1M output tokens")
    args = parser.parse_args()

    # Windows terminals may not default to UTF-8; model output can contain any character.
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="[retry] %(message)s")

    try:
        result = stream_chat(
            args.prompt,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.retries,
            on_token=lambda token: print(token, end="", flush=True),
        )
    except LLMError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    cost = estimate_cost(
        result.input_tokens, result.output_tokens, args.input_price, args.output_price
    )
    speed = result.output_tokens / result.seconds if result.seconds else 0.0
    print(
        f"\n\n---\n"
        f"tokens: {result.input_tokens} in, {result.output_tokens} out"
        f" | {result.seconds:.1f}s ({speed:.1f} tokens/s)\n"
        f"cost: $0 locally, ~${cost:.6f} at ${args.input_price}/${args.output_price} per 1M tokens"
    )
