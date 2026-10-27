-- Queries for exploring the RAG tables by hand (DBeaver, psql, any SQL client).
-- Connect to: host 127.0.0.1, port 5433, database ai_systems, user/password postgres/postgres
-- Run `uv run ingest` first. In DBeaver: put the cursor in one query and press Ctrl+Enter.

-- 1. What's stored: chunks and embeddings per agency
SELECT tenant_id, strategy, COUNT(*) AS chunks, COUNT(embedding) AS embedded,
       ROUND(AVG(word_count)) AS avg_words
FROM rag.chunks
GROUP BY tenant_id, strategy
ORDER BY tenant_id;

-- 2. The chunks themselves (the `search` and `embedding` columns are explained below)
SELECT chunk_id, section, word_count, content
FROM rag.chunks
WHERE tenant_id = 'ruaka-rentals'
ORDER BY source, chunk_index;

-- 3. What an embedding looks like: 768 numbers per chunk. Here are the first 8.
SELECT chunk_id, vector_dims(embedding) AS dimensions,
       (embedding::real[])[1:8] AS first_8_numbers
FROM rag.chunks
ORDER BY chunk_id
LIMIT 5;

-- 4. Semantic neighbours: the chunks whose meaning is closest to one chosen chunk.
--    Cosine distance (<=>): 0 = same meaning, bigger = less related. The OTHER agencies'
--    deposit sections come out 4th and 5th, closer than most of Ruaka's own documents:
--    that's the cross-tenant risk the tenant filter (and M8) exists for.
SELECT other.chunk_id, other.section,
       ROUND((other.embedding <=> chosen.embedding)::numeric, 3) AS distance
FROM rag.chunks AS chosen, rag.chunks AS other
WHERE chosen.section = 'Ruaka Rentals Tenancy Terms > Deposit'
  AND other.chunk_id <> chosen.chunk_id
ORDER BY distance
LIMIT 10;

-- 5. What keyword search sees: each chunk as word stems with positions, stopwords removed
SELECT chunk_id, section, search
FROM rag.chunks
WHERE tenant_id = 'pwani-lettings'
LIMIT 5;

-- 6. How a question is turned into a keyword query (try your own sentences)
SELECT plainto_tsquery('english', 'Are viewings allowed on Sundays?') AS all_words,
       replace(plainto_tsquery('english', 'Are viewings allowed on Sundays?')::text,
               '&', '|')::tsquery AS any_word;

-- 7. Keyword search exactly as rag/keyword.py runs it ("any" mode, tenant filter on)
SELECT chunk_id, section, ROUND(ts_rank(search, q.query)::numeric, 4) AS rank, content
FROM rag.chunks,
     (SELECT replace(plainto_tsquery('english', 'Can I keep a cat?')::text,
                     '&', '|')::tsquery AS query) AS q
WHERE search @@ q.query AND tenant_id = 'ruaka-rentals'
ORDER BY rank DESC
LIMIT 5;

-- 8. Indexes (the GIN index is what makes keyword search fast)
SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'rag';

-- 9. Project 00's tables live in the `public` schema of the same database
SELECT c.id, c.created_at, COUNT(m.*) AS messages
FROM public.conversations c LEFT JOIN public.messages m ON m.conversation_id = c.id
GROUP BY c.id, c.created_at
ORDER BY c.created_at DESC
LIMIT 10;
