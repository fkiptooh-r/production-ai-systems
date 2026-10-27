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
**3 fictional letting agencies (tenants)**, 4 markdown documents each, in
[`data/tenants/<tenant_id>/`](data/tenants/): tenancy policy, viewings & applications,
maintenance, and an area guide. **All of it is synthetic**: agencies, phone numbers,
fees and rules are invented.

| Tenant | Area | Example of a deliberately different answer |
|--------|------|---------------------------------------------|
| `nyumba-homes` | Kilimani / Westlands | Deposit: 2 months · Pets: cats yes, dogs case by case |
| `ruaka-rentals` | Ruaka | Deposit: 1 month · Pets: none allowed |
| `pwani-lettings` | Nyali (coast) | Deposit: 3 months furnished / 2 unfurnished · Pets: landlord approval |

The same question has a **different correct answer per tenant**, so retrieving another
agency's chunk produces a confidently wrong answer. That's what makes tenant isolation (M8)
a real test.

## Milestones

- [x] **M1 · Ingestion.** Load PDF/Markdown/JSON, clean the text, and chunk it (start with fixed size + overlap, then try splitting by headings). Every chunk stores `tenant_id`, `source`, `page`/`section`, `chunk_id`.
- [ ] **M2 · Eval set FIRST.** Write 50 questions with their correct answer and the `chunk_id`(s) that contain it. Include some questions whose answers are *not* in the data.
  *Deliverable:* `eval/questions.jsonl`
  *Planned change:* label each question with its tenant, source and an **answer phrase**
  instead of chunk ids, because chunk ids change whenever the chunking strategy or size
  changes. A retrieved chunk counts as relevant if it's from the right tenant and source
  and contains the phrase.
- [ ] **M3 · Vector search.** Local embeddings with `sentence-transformers` (e.g. a small BGE or MiniLM model), stored in Postgres + pgvector. Measure Recall@5 and MRR.
  *Planned change:* embeddings from **Ollama** (e.g. `nomic-embed-text`, ~270 MB) instead
  of sentence-transformers, which would pull in PyTorch (1 GB+) on a CPU-only laptop and
  duplicate the model runtime we already have.
- [ ] **M4 · Keyword search.** BM25 (`rank_bm25`) or Postgres full-text search. Measure. Notice which questions keyword search wins (names, codes, exact numbers).
- [ ] **M5 · Hybrid.** Combine both with RRF. Measure.
- [ ] **M6 · Rerank.** Retrieve the top 20–50, then rerank with a cross-encoder down to the top 5. Measure. Also measure latency, since reranking costs time.
- [ ] **M7 · Generation with citations.** Answer only from the retrieved chunks, cite `[source, page]`, and say "I don't know" when the context doesn't contain the answer. Check the not-in-data questions.
- [ ] **M8 · Tenant isolation.** Enforce the tenant filter in the query layer (bonus: Postgres Row-Level Security). Write a test proving tenant A can never retrieve tenant B's chunks.
- [ ] **M9 · API + demo.** `POST /ask` in FastAPI, plus a minimal UI (Streamlit or a single HTML page). Record a 2-minute demo.

## How to run

Uses the Postgres from 00-foundations (tables go in a separate `rag` schema).

```bash
cd ../00-foundations && docker compose up -d && cd ../01-hybrid-rag
uv sync
uv run pytest                                  # DB tests skip if Postgres isn't running
uv run ingest                                  # headings strategy (default)
uv run ingest --strategy fixed --size 80 --overlap 15
```

## Results

### M1 · Chunking (12 documents, 3 tenants)

| Strategy | Chunks | Avg words | Max words |
|----------|--------|-----------|-----------|
| fixed (80 words, 15 overlap) | 36 | ~70 | 80 |
| headings (sections, split above 120) | 64 | ~31 | 77 |

**Headings** gives one topic per chunk ("Ruaka Rentals Tenancy Terms > Deposit"); **fixed**
can mix two topics in one chunk. Which retrieves better is measured from M3 on.

**Design note:** each chunk has a `full_text` = heading trail + body, and that's what gets
searched. The coverage test found that, without it, heading words were lost, including the
agency name, which the body of "The deposit is one month's rent" never mentions.

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
