"""Grounded answers: retrieve chunks, then have the model answer ONLY from them, with citations.

The prompt numbers each retrieved chunk [1], [2], ... and asks the model to cite those
numbers after each fact, or to say "I don't know" when the sources don't contain the answer.
Citations are then mapped back to chunks, so every claim can be traced to a document.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from shared.llm import ChatResult, stream_chat

from rag.chunking import Chunk

DEFAULT_MODEL = "llama3.2:3b"
TOP_K = 4  # chunks shown to the model: enough to cover the answer, few enough to stay focused
I_DONT_KNOW = "I don't know"

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
    chunks = search(question, tenant_id, top_k)
    result = chat(
        build_messages(question, tenant_id, chunks),
        model=model,
        options={"temperature": 0},  # same question, same answer: needed for a fair eval
        on_token=on_token,
    )
    numbers = cited_numbers(result.text, len(chunks))
    return Answer(
        question=question,
        text=result.text.strip(),
        retrieved=chunks,
        cited=[chunks[n - 1] for n in numbers],
        seconds=result.seconds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
