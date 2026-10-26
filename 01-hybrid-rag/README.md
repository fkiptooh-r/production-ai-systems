# 01 · Hybrid RAG Search

**Goal:** a tenant-aware question-answering system over documents that combines keyword and semantic search, reranks results, cites its sources, and **proves** each improvement with numbers.
**Time:** 3–4 weeks.

## What you'll learn
- Document loading, cleaning and chunking strategies
- Embeddings and vector search (pgvector)
- Keyword search (BM25 / Postgres full-text)
- Fusing result lists with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded generation with citations and "I don't know" answers
- Retrieval metrics: Recall@k, MRR
- Multi-tenant data isolation

## Dataset
Pick **one** domain and create data for **2–3 fake tenants**:
- House listings and area guides (synthetic, shaped like your house-hunting app's data)
- Product catalogues and store FAQs (synthetic, one catalogue per store/tenant)
- Course notes / past papers (public or your own)

Never use real customer data. Generating realistic synthetic data with an LLM is itself a good exercise.

## Milestones

- [ ] **M1 · Ingestion.** Load PDF/Markdown/JSON, clean the text, and chunk it (start with fixed size + overlap, then try splitting by headings). Every chunk stores `tenant_id`, `source`, `page`/`section`, `chunk_id`.
- [ ] **M2 · Eval set FIRST.** Write 50 questions with their correct answer and the `chunk_id`(s) that contain it. Include some questions whose answers are *not* in the data.
  *Deliverable:* `eval/questions.jsonl`
- [ ] **M3 · Vector search.** Local embeddings with `sentence-transformers` (e.g. a small BGE or MiniLM model), stored in Postgres + pgvector. Measure Recall@5 and MRR.
- [ ] **M4 · Keyword search.** BM25 (`rank_bm25`) or Postgres full-text search. Measure. Notice which questions keyword search wins (names, codes, exact numbers).
- [ ] **M5 · Hybrid.** Combine both with RRF. Measure.
- [ ] **M6 · Rerank.** Retrieve the top 20–50, then rerank with a cross-encoder down to the top 5. Measure. Also measure latency, since reranking costs time.
- [ ] **M7 · Generation with citations.** Answer only from the retrieved chunks, cite `[source, page]`, and say "I don't know" when the context doesn't contain the answer. Check the not-in-data questions.
- [ ] **M8 · Tenant isolation.** Enforce the tenant filter in the query layer (bonus: Postgres Row-Level Security). Write a test proving tenant A can never retrieve tenant B's chunks.
- [ ] **M9 · API + demo.** `POST /ask` in FastAPI, plus a minimal UI (Streamlit or a single HTML page). Record a 2-minute demo.

## Results table (fill this in as you go)

| Retrieval setup | Recall@5 | MRR | p50 latency |
|-----------------|----------|-----|-------------|
| Vector only | | | |
| BM25 only | | | |
| Hybrid (RRF) | | | |
| Hybrid + rerank | | | |

## Done when
- The results table is filled in and the README explains *why* the winner won
- The tenant isolation test passes
- The demo video is linked

## Stretch goals
- Contextual retrieval (prepend a short LLM-written context to each chunk before embedding)
- Query rewriting / multi-query
- Extracting filters from natural language ("under 40k, near a school" → structured filters + semantic query)
- Multilingual queries, if your users mix languages

## Where this goes in your products
- **E-commerce:** product search that understands both exact SKUs and "comfy running shoes"; a per-store FAQ bot
- **House hunting:** natural-language listing search with hard filters
- **EdTech:** "ask your notes" with page citations

## Resources
See [docs/RESOURCES.md → Hybrid RAG](../docs/RESOURCES.md#01--hybrid-rag).
