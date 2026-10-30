"""Chat with an agency's documents in the terminal.

Usage:
  uv run ask --tenant ruaka-rentals                      # interactive chat, Ctrl+C to quit
  uv run ask --tenant ruaka-rentals "Can I keep a cat?"  # one question, then exit

Needs Postgres (docker compose up -d in 00-foundations), Ollama running, and `uv run ingest`.
Each question is answered on its own: there's no memory of earlier turns yet.
"""

import argparse
import sys

from shared.llm import LLMError

from rag.generate import DEFAULT_MODEL, Answer, agency_name, answer
from rag.ingest import TENANTS_DIR, ChunkStore
from rag.vector import ollama_embedder, vector_searcher


def tenants() -> list[str]:
    return sorted(path.name for path in TENANTS_DIR.iterdir() if path.is_dir())


def print_sources(result: Answer) -> None:
    if result.abstained:
        return
    if not result.cited:
        print("  (no sources cited: treat this answer with suspicion)")
        return
    for n, chunk in enumerate(result.retrieved, start=1):
        if chunk in result.cited:
            print(f"  [{n}] {chunk.source} > {chunk.section.split(' > ')[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an agency's documents a question.")
    parser.add_argument("question", nargs="?", help="omit to start an interactive chat")
    parser.add_argument("--tenant", choices=tenants(), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    args = parser.parse_args()

    store = ChunkStore()
    if not store.is_up():
        raise SystemExit("Postgres isn't running: cd ../00-foundations && docker compose up -d")
    if not store.corpus()["embedded"]:
        raise SystemExit("No embedded chunks yet. Run: uv run ingest")
    search = vector_searcher(store, ollama_embedder())  # M3-M6 winner: vector + tenant filter
    agency = agency_name(args.tenant)

    def ask(question: str) -> None:
        print(f"{agency}: ", end="", flush=True)
        try:
            result = answer(
                question,
                args.tenant,
                search,
                model=args.model,
                on_token=lambda token: print(token, end="", flush=True),
            )
        except LLMError as exc:
            print(f"\n[error] {exc}\nIs Ollama running? (ollama serve)")
            return
        print()
        print_sources(result)
        if args.show_context:
            for n, chunk in enumerate(result.retrieved, start=1):
                print(f"  ({n}) {chunk.chunk_id}: {chunk.text[:100]}...")
        print(f"  ({result.seconds:.1f}s, {result.output_tokens} tokens, trace {result.trace_id})")

    if args.question:
        ask(args.question)
        return

    print(f"Chatting with {agency}'s documents. Ctrl+C or an empty line to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        ask(question)
        print()
    sys.exit(0)
