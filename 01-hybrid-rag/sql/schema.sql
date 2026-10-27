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

-- M8: tenant isolation enforced by the database, not just by the app's WHERE clause.
-- Searches run as rag_reader, which may only SELECT, and Row-Level Security (RLS) hides
-- every row whose tenant_id isn't the one set for the transaction in app.tenant_id.
-- Why a separate role: superusers and the table owner (postgres here) BYPASS RLS, so
-- ingest keeps working as postgres while every search is filtered.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_reader') THEN
        CREATE ROLE rag_reader NOLOGIN;  -- no password: only reachable via SET ROLE
    END IF;
END $$;
GRANT USAGE ON SCHEMA rag TO rag_reader;
GRANT SELECT ON rag.chunks TO rag_reader;  -- read only: no INSERT, UPDATE or DELETE

ALTER TABLE rag.chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON rag.chunks;
-- current_setting(..., true) returns NULL when unset, and `tenant_id = NULL` is never true:
-- forgetting to set the tenant shows zero rows (fails closed) instead of all of them.
CREATE POLICY tenant_isolation ON rag.chunks FOR SELECT TO rag_reader
    USING (tenant_id = current_setting('app.tenant_id', true));
