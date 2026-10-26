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
