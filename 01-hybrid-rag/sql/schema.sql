-- Project 01 tables live in their own Postgres schema ("rag"), separate from
-- 00-foundations' tables in the same database. Safe to run repeatedly.

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.chunks (
    chunk_id     TEXT PRIMARY KEY,          -- "tenant/source#index"
    tenant_id    TEXT NOT NULL,             -- which letting agency owns this text
    source       TEXT NOT NULL,             -- file name
    section      TEXT NOT NULL,             -- heading trail, e.g. "Tenancy Policy > Deposit"
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    word_count   INTEGER NOT NULL,
    strategy     TEXT NOT NULL,             -- "headings" or "fixed", for comparing later
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every search filters by tenant first (M8 makes that mandatory), so index it.
CREATE INDEX IF NOT EXISTS chunks_by_tenant ON rag.chunks (tenant_id);

-- M3: semantic search with pgvector. 768 = nomic-embed-text's vector size.
-- NULL until `uv run ingest` embeds the chunk. No vector index (HNSW) yet: with ~100 rows an
-- exact scan takes well under a millisecond; an index starts paying off around 10k+ rows.
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE rag.chunks ADD COLUMN IF NOT EXISTS embedding vector(768);

-- M4: keyword search with Postgres full-text search. The tsvector holds each chunk's word
-- stems ("viewings" -> "view") minus stopwords ("the", "is"). GENERATED ... STORED means
-- Postgres keeps it in sync with section + content automatically, so ingest never touches it.
ALTER TABLE rag.chunks ADD COLUMN IF NOT EXISTS search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', section || ' ' || content)) STORED;
-- GIN: an inverted index (stem -> rows), the classic search-engine structure.
CREATE INDEX IF NOT EXISTS chunks_search ON rag.chunks USING GIN (search);
