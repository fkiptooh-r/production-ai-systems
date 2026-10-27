"""Load each tenant's documents, chunk them, and store the chunks in Postgres.

Folder layout = ownership: data/tenants/<tenant_id>/<document>.md

Usage: uv run ingest [--strategy headings|fixed] [--size 120] [--overlap 20] [--skip-embed]
Re-running replaces the previous chunks, so it's safe to run as often as you like.
Embedding each chunk (M3) needs Ollama with nomic-embed-text; --skip-embed stores text only.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from shared.config import get_settings

from rag.chunking import Chunk, chunk_document

PROJECT_DIR = Path(__file__).resolve().parents[2]
TENANTS_DIR = PROJECT_DIR / "data" / "tenants"
SCHEMA_FILE = PROJECT_DIR / "sql" / "schema.sql"


@dataclass(frozen=True)
class Document:
    tenant_id: str
    source: str
    text: str


def load_documents(tenants_dir: Path = TENANTS_DIR) -> list[Document]:
    return [
        Document(tenant_id=path.parent.name, source=path.name, text=path.read_text("utf-8"))
        for path in sorted(tenants_dir.glob("*/*.md"))
    ]


class ChunkStore:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or get_settings().database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, row_factory=dict_row, connect_timeout=3)

    def is_up(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA_FILE.read_text())

    def replace_all(self, chunks: list[Chunk], strategy: str) -> None:
        """Swap the whole corpus in one transaction: readers see the old set or the new one,
        never a half-written mix."""
        with self.connect() as conn:
            conn.execute("DELETE FROM rag.chunks")
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO rag.chunks (chunk_id, tenant_id, source, section,
                                            chunk_index, content, word_count, strategy)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            c.chunk_id,
                            c.tenant_id,
                            c.source,
                            c.section,
                            c.index,
                            c.text,
                            c.word_count,
                            strategy,
                        )
                        for c in chunks
                    ],
                )

    def corpus(self) -> dict:
        """What's stored right now: {"strategy", "chunks", "embedded"}."""
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT MIN(strategy) AS strategy, COUNT(*) AS chunks,
                       COUNT(embedding) AS embedded
                FROM rag.chunks
                """
            ).fetchone()

    def stats(self) -> list[dict]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT tenant_id,
                       COUNT(DISTINCT source) AS documents,
                       COUNT(*) AS chunks,
                       ROUND(AVG(word_count)) AS avg_words,
                       MAX(word_count) AS max_words
                FROM rag.chunks
                GROUP BY tenant_id
                ORDER BY tenant_id
                """
            ).fetchall()


def build_chunks(documents: list[Document], strategy: str, size: int, overlap: int) -> list[Chunk]:
    return [
        chunk
        for doc in documents
        for chunk in chunk_document(
            doc.text,
            tenant_id=doc.tenant_id,
            source=doc.source,
            strategy=strategy,
            size=size,
            overlap=overlap,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk tenant documents into Postgres.")
    parser.add_argument("--strategy", choices=["headings", "fixed"], default="headings")
    parser.add_argument("--size", type=int, default=120, help="max words per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="words shared by neighbours")
    parser.add_argument("--skip-embed", action="store_true", help="store chunks without vectors")
    args = parser.parse_args()

    documents = load_documents()
    chunks = build_chunks(documents, args.strategy, args.size, args.overlap)

    store = ChunkStore()
    store.init_schema()
    store.replace_all(chunks, args.strategy)

    print(f"strategy={args.strategy} size={args.size} overlap={args.overlap}\n")
    print(f"{'tenant':<16} {'docs':>4} {'chunks':>6} {'avg words':>9} {'max':>4}")
    for row in store.stats():
        print(
            f"{row['tenant_id']:<16} {row['documents']:>4} {row['chunks']:>6}"
            f" {row['avg_words']:>9} {row['max_words']:>4}"
        )

    if args.skip_embed:
        return
    from shared.llm import LLMError

    from rag.vector import embed_chunks, ollama_embedder  # imported here: vector imports ingest

    started = time.perf_counter()
    try:
        count = embed_chunks(store, ollama_embedder())
    except LLMError as exc:
        raise SystemExit(f"\nEmbedding failed: {exc}\nIs Ollama running? (ollama serve)") from exc
    print(f"\nembedded {count} chunks in {time.perf_counter() - started:.1f}s")
