"""Grounded answers: retrieve chunks, then have the model answer ONLY from them, with citations.

The prompt numbers each retrieved chunk [1], [2], ... and asks the model to cite those
numbers after each fact, or to say "I don't know" when the sources don't contain the answer.
Citations are then mapped back to chunks, so every claim can be traced to a document.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from shared.costs import estimate_cost
from shared.llm import ChatResult, stream_chat
from tracing.tracer import set_attribute, span

from rag.chunking import Chunk

DEFAULT_MODEL = "llama3.2:3b"
TOP_K = 4  # chunks shown to the model: enough to cover the answer, few enough to stay focused
I_DONT_KNOW = "I don't know"
# Ollama is free, so cost_usd is 0. cost_usd_hosted prices the same tokens at a mid-tier hosted
# model (USD per 1M input/output tokens, the reference used in 00 M2): what this would cost.
HOSTED_PRICES = (3.0, 15.0)

SYSTEM_PROMPT = """You answer questions from tenants of {agency}, a letting agency.

Rules:
- Use ONLY the numbered sources below. Never use outside knowledge or guess.
- After each fact, cite the source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: I don't know.
- Answer in 1-3 short sentences."""

CITATION = re.compile(r"\[(\d+)\]")

Searcher = Callable[[str, str, int], list[Chunk]]
Chat = Callable[..., ChatResult]  # same signature as shared.llm.stream_chat


def agency_name(tenant_id: str) -> str:
    """ "ruaka-rentals" -> "Ruaka Rentals"."""
    return tenant_id.replace("-", " ").title()


def build_messages(question: str, tenant_id: str, chunks: list[Chunk]) -> list[dict]:
    sources = "\n\n".join(
        f"[{n}] ({chunk.source} > {chunk.section.split(' > ')[-1]})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(agency=agency_name(tenant_id))},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]


def cited_numbers(text: str, available: int) -> list[int]:
    """Source numbers cited in the answer, in order of first use, ignoring invalid ones."""
    seen: list[int] = []
    for match in CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= available and n not in seen:
            seen.append(n)
    return seen


def is_abstention(text: str) -> bool:
    return "don't know" in text.lower().replace("’", "'")


@dataclass
class Answer:
    question: str
    text: str
    retrieved: list[Chunk]
    cited: list[Chunk] = field(default_factory=list)
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    trace_id: str | None = None  # look it up with `uv run trace-view <id>` in 02-llm-tracing

    @property
    def abstained(self) -> bool:
        return is_abstention(self.text)


def answer(
    question: str,
    tenant_id: str,
    search: Searcher,
    *,
    model: str = DEFAULT_MODEL,
    top_k: int = TOP_K,
    chat: Chat = stream_chat,
    on_token: Callable[[str], None] = lambda _: None,
) -> Answer:
    with span("answer", question=question, top_k=top_k) as root:
        set_attribute("tenant_id", tenant_id)
        set_attribute("model", model)
        chunks = search(question, tenant_id, top_k)
        with span("generate", model=model) as gen:
            result = chat(
                build_messages(question, tenant_id, chunks),
                model=model,
                options={"temperature": 0},  # same question, same answer: needed for a fair eval
                on_token=on_token,
            )
            gen.output = result.text
            hosted = estimate_cost(result.input_tokens, result.output_tokens, *HOSTED_PRICES)
            gen.attributes.update(
                model=model,
                tokens_in=result.input_tokens,
                tokens_out=result.output_tokens,
                cost_usd=0.0,
                cost_usd_hosted=round(hosted, 6),
            )
        numbers = cited_numbers(result.text, len(chunks))
        cited = [chunks[n - 1] for n in numbers]
        root.output = result.text.strip()
        root.attributes.update(
            retrieved=[c.chunk_id for c in chunks],
            cited=[c.chunk_id for c in cited],
            abstained=is_abstention(result.text),
        )
        return Answer(
            question=question,
            text=result.text.strip(),
            retrieved=chunks,
            cited=cited,
            seconds=result.seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            trace_id=root.trace_id,
        )
