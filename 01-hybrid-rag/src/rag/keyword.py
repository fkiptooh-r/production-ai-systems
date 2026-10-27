"""Keyword search with Postgres full-text search: match the question's words, not its meaning.

Both sides are reduced to stems with stopwords removed, so "Are viewings allowed on
Sundays?" becomes 'view' & 'allow' & 'sunday'. Two ways to combine the words:

- "all": a chunk must contain every word (`&`). Precise, but natural questions rarely have
  all their words in one chunk, so most return nothing.
- "any": a chunk needs at least one word (`|`), ranked by how many it matches and how
  rare they are (ts_rank). This is how keyword search is normally used for retrieval.

No model and no embedding call: one indexed SQL query, a few milliseconds.
"""

from collections.abc import Callable

from rag.chunking import Chunk
from rag.ingest import ChunkStore
from rag.vector import row_to_chunk

# plainto_tsquery joins the stems with & ("all"); swapping & for | gives "any".
QUERIES = {
    "all": "plainto_tsquery('english', %(question)s)",
    "any": "replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery",
}


def keyword_searcher(
    store: ChunkStore, *, mode: str = "any"
) -> Callable[[str, str, int], list[Chunk]]:
    """search(question, tenant_id, k) -> up to k of that tenant's chunks, best match first.

    Tenant isolation: WHERE clause + Row-Level Security on the tenant connection (M8).
    """
    if mode not in QUERIES:
        raise ValueError(f"unknown mode {mode!r}; use 'all' or 'any'")
    sql = f"""
        SELECT tenant_id, source, section, chunk_index, content
        FROM rag.chunks, (SELECT {QUERIES[mode]} AS query) AS q
        WHERE search @@ q.query AND tenant_id = %(tenant)s
        ORDER BY ts_rank(search, q.query) DESC, chunk_id
        LIMIT %(k)s
    """

    def search(question: str, tenant_id: str, k: int) -> list[Chunk]:
        with store.tenant_connection(tenant_id) as conn:
            rows = conn.execute(sql, {"question": question, "tenant": tenant_id, "k": k})
            return [row_to_chunk(row) for row in rows.fetchall()]

    return search


def show_query(store: ChunkStore, question: str) -> str:
    """What Postgres actually searches for, e.g. "'pet' & 'allow'". Handy for debugging."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT plainto_tsquery('english', %s)::text AS query", (question,)
        ).fetchone()
    return row["query"]
