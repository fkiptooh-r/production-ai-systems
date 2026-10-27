"""Split documents into chunks small enough to retrieve precisely.

Two strategies, so later milestones can measure which one retrieves better:

- "fixed":    every `size` words, overlapping by `overlap` words so a fact that falls on a
              boundary still appears whole in at least one chunk.
- "headings": one chunk per markdown section, which keeps related sentences together;
              sections longer than `size` words are split further with the fixed strategy.

Sizes are in words, not model tokens: simple, and close enough (1 word ≈ 1.3 tokens).
"""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Section:
    path: str  # heading trail, e.g. "Nyumba Homes Tenancy Policy > Deposit"
    text: str


@dataclass(frozen=True)
class Chunk:
    tenant_id: str
    source: str  # file name, e.g. "tenancy-policy.md"
    section: str
    index: int  # position within the document
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.tenant_id}/{self.source}#{self.index}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def full_text(self) -> str:
        """Heading trail + body: what gets searched and what the model sees.

        The body alone often lacks key context. "The deposit is one month's rent" never
        says which agency it's about; the heading trail "Ruaka Rentals Tenancy Terms >
        Deposit" does.
        """
        return f"{self.section}\n{self.text}" if self.section else self.text


def clean(text: str) -> str:
    """Collapse all whitespace (newlines, double spaces) to single spaces."""
    return " ".join(text.split())


def split_sections(markdown: str) -> list[Section]:
    """One Section per heading, carrying the full heading trail for context."""
    sections: list[Section] = []
    trail: list[tuple[int, str]] = []  # (heading level, title) from the top level down
    lines: list[str] = []

    def flush() -> None:
        text = clean("\n".join(lines))
        if text:
            sections.append(Section(" > ".join(title for _, title in trail), text))
        lines.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, match.group(2)))
        else:
            lines.append(line)
    flush()
    return sections


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Windows of `size` words, each starting `size - overlap` words after the previous one."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got overlap={overlap} size={size}")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []

    step = size - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return windows


def document_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            return match.group(2)
    return ""


def chunk_document(
    markdown: str,
    *,
    tenant_id: str,
    source: str,
    strategy: str = "headings",
    size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    if strategy == "fixed":
        plain = "\n".join(HEADING.sub(r"\2", line) for line in markdown.splitlines())
        title = document_title(markdown)
        pieces = [(title, text) for text in split_words(clean(plain), size, overlap)]
    elif strategy == "headings":
        pieces = [
            (section.path, text)
            for section in split_sections(markdown)
            for text in split_words(section.text, size, overlap)
        ]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'fixed' or 'headings'")

    return [
        Chunk(tenant_id=tenant_id, source=source, section=section, index=i, text=text)
        for i, (section, text) in enumerate(pieces)
    ]
