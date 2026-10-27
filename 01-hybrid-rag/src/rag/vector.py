"""Semantic search: embed every chunk once, then return the chunks closest to the question.

"Closest" is cosine distance (pgvector's `<=>` operator): 0 = same direction (same meaning),
2 = opposite. Embedding happens at ingest time for chunks and at query time for questions.
"""

from collections.abc import Callable

from shared.embeddings import DEFAULT_EMBED_MODEL, embed

from rag.chunking import Chunk
from rag.ingest import ChunkStore

# nomic-embed-text was trained with task prefixes that tell it whether a text is something
# to be found or something doing the finding. Leaving them out quietly hurts retrieval.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# texts -> one vector each. Swappable so tests run without Ollama.
Embedder = Callable[[list[str]], list[list[float]]]


def ollama_embedder(model: str = DEFAULT_EMBED_MODEL) -> Embedder:
    return lambda texts: embed(texts, model=model)


def to_pgvector(vector: list[float]) -> str:
    """pgvector's text format, e.g. '[0.1,0.2]'. Avoids needing numpy or an extra driver."""
    return "[" + ",".join(str(x) for x in vector) + "]"


def row_to_chunk(row: dict) -> Chunk:
    return Chunk(
        tenant_id=row["tenant_id"],
        source=row["source"],
        section=row["section"],
        index=row["chunk_index"],
        text=row["content"],
    )


def embed_chunks(store: ChunkStore, embedder: Embedder, batch_size: int = 16) -> int:
    """Embed every stored chunk that has no embedding yet. Returns how many were embedded."""
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, tenant_id, source, section, chunk_index, content
            FROM rag.chunks WHERE embedding IS NULL ORDER BY chunk_id
            """
        ).fetchall()

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # Embed full_text (heading trail + body), the same text the eval checks against.
        vectors = embedder([DOCUMENT_PREFIX + row_to_chunk(row).full_text for row in batch])
        with store.connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE rag.chunks SET embedding = %s::vector WHERE chunk_id = %s",
                [(to_pgvector(v), row["chunk_id"]) for v, row in zip(vectors, batch, strict=True)],
            )
    return len(rows)


def vector_searcher(
    store: ChunkStore, embedder: Embedder, *, tenant_filter: bool = True
) -> Callable[[str, str, int], list[Chunk]]:
    """search(question, tenant_id, k) -> the k nearest chunks, nearest first.

    tenant_filter=False searches every agency's chunks: useful once, to measure how often
    another tenant's near-identical policy outranks the right one.
    """
    where = "embedding IS NOT NULL" + (" AND tenant_id = %(tenant)s" if tenant_filter else "")

    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        [vector] = embedder([QUERY_PREFIX + question])
        with store.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tenant_id, source, section, chunk_index, content
                FROM rag.chunks WHERE {where}
                ORDER BY embedding <=> %(vector)s::vector
                LIMIT %(k)s
                """,
                {"tenant": tenant_id, "vector": to_pgvector(vector), "k": k},
            ).fetchall()
        return [row_to_chunk(row) for row in rows]

    return search
