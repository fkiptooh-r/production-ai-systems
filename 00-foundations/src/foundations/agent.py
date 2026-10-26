"""A hand-written agent loop: the model calls tools until it can answer.

    question -> model -> tool call? -> run tool -> result back to model -> ... -> answer

Every agent framework wraps this same loop. Writing it once makes the frameworks
easy to understand later.

Usage: uv run agent "Find me a 2-bedroom under 40k with parking"
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from shared.llm import DEFAULT_MODEL, ChatResult, LLMError, stream_chat

from foundations.tools import execute_tool, load_listings, tool_schemas

SYSTEM_PROMPT = """You are a helpful rental assistant.
Use the tools to look up listings; never invent listings, prices or ids.
Only pass the filters the user actually asked for.
If a question needs no data (e.g. "what is a bedsitter?"), answer directly without tools.
After getting tool results, answer briefly and mention each listing's id and price."""


class AgentError(Exception):
    """The agent could not finish."""


@dataclass
class ToolStep:
    name: str
    arguments: dict
    output: str


@dataclass
class AgentResult:
    answer: str
    steps: list[ToolStep] = field(default_factory=list)


def run_agent(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = 5,
    listings: list[dict] | None = None,
    chat: Callable[..., ChatResult] = stream_chat,
    on_step: Callable[[ToolStep], None] = lambda _: None,
) -> AgentResult:
    """Answer a question, letting the model call tools up to max_steps times."""
    listings = load_listings() if listings is None else listings
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []

    for _ in range(max_steps):
        result = chat(messages, model=model, tools=tool_schemas(), options={"temperature": 0})
        if not result.tool_calls:
            return AgentResult(answer=result.text, steps=steps)

        messages.append(
            {"role": "assistant", "content": result.text, "tool_calls": result.tool_calls}
        )
        for call in result.tool_calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or {}
            output = execute_tool(name, arguments, listings)
            step = ToolStep(name, arguments, output)
            steps.append(step)
            on_step(step)
            messages.append({"role": "tool", "tool_name": name, "content": output})

    # A confused model can keep calling tools forever; stop it and say so.
    raise AgentError(f"no answer after {max_steps} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the rental assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    def show(step: ToolStep) -> None:
        print(f"[tool] {step.name}({json.dumps(step.arguments)})")
        print(f"       -> {step.output[:200]}{'...' if len(step.output) > 200 else ''}")

    try:
        result = run_agent(args.question, model=args.model, max_steps=args.max_steps, on_step=show)
    except (AgentError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{result.answer}")
