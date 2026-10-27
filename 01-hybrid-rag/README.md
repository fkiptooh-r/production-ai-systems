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
- [x] **M2 · Eval set FIRST.** Write 50 questions with their correct answer and the `chunk_id`(s) that contain it. Include some questions whose answers are *not* in the data.
  *Deliverable:* `eval/questions.jsonl`
  *Change made:* each question is labelled with its tenant, source and an **answer phrase**
  instead of chunk ids, because chunk ids change whenever the chunking strategy or size
  changes. A retrieved chunk counts as relevant if it's from the right tenant and source
  and contains the phrase.
- [x] **M3 · Vector search.** Local embeddings with `sentence-transformers` (e.g. a small BGE or MiniLM model), stored in Postgres + pgvector. Measure Recall@5 and MRR.
  *Change made:* embeddings from **Ollama** (`nomic-embed-text`, 137M params, ~270 MB)
  instead of sentence-transformers, which would pull in PyTorch (1 GB+) on a CPU-only laptop
  and duplicate the model runtime we already have.
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
ollama pull nomic-embed-text                   # embedding model, ~270 MB (Ollama must be running)
uv sync
uv run pytest                                  # DB tests skip if Postgres isn't running
uv run ingest                                  # headings strategy (default) + embeddings
uv run ingest --strategy fixed --size 80 --overlap 15
uv run evaluate                                # random baseline
uv run evaluate --searcher vector              # scores whatever `ingest` stored last
uv run evaluate --searcher vector --all-tenants --k 1
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

### M2 · Eval set (60 questions)

[`eval/questions.jsonl`](eval/questions.jsonl), 20 per tenant, one JSON object per line:

```json
{"id": "rr-07", "tenant": "ruaka-rentals", "kind": "paraphrase",
 "question": "Can I keep a cat in my bedsitter?",
 "source": "tenancy-policy.md", "answer": "cats, dogs and birds", "absent": null}
```

| Kind | n | What it tests | Example |
|------|---|---------------|---------|
| `direct` | 17 | Question reuses the document's words | "What are the viewing hours?" |
| `paraphrase` | 24 | Same meaning, different words (favours vector search) | "Can I rent my flat out on Airbnb?" → "holiday letting" |
| `exact` | 10 | Numbers, codes, emails (favours keyword search) | "What paybill number do I pay rent to?" → "522100" |
| `not_in_data` | 9 | No answer exists for this tenant; the right reply is "I don't know" (M7) | "Is there a holding fee?" asked of Ruaka |

**Traps on purpose:** the same question is asked of several agencies with different answers
("Can I bring my dog?"), and several not-in-data questions *are* answered by another
agency (Ruaka has no holding fee; Pwani does). A retriever that ignores the tenant scores 0
on those.

**Metrics** (answerable questions only, `src/rag/evaluate.py`):
- **Recall@5:** share of questions with a relevant chunk in the top 5. *Did we find it?*
- **MRR:** mean of 1/rank of the first relevant chunk. *Did we put it near the top?*
  Rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2, not found → 0.

**The exam is tested too** (`tests/test_eval_set.py`): every answer phrase must appear in
its source document *and* survive chunking whole (headings 120/20, fixed 80/15 and 50/10),
and every not-in-data term must be absent from that tenant's documents.

## Results table (fill this in as you go)

### M3 · Vector search (nomic-embed-text + pgvector)

| Setup (tenant filter on unless stated) | Recall@5 | Recall@1 | MRR | p50 latency |
|----------------------------------------|----------|----------|-----|-------------|
| Vector, headings (64 chunks) | 1.00 | 0.98 | 0.99 | ~0.4–1 s |
| Vector, fixed 80/15 (36 chunks) | 1.00 | | 0.97 | ~0.45 s |
| Vector, headings, **all tenants** | 1.00 | | 0.77 | ~0.7 s |

- **Paraphrases work:** "Can I rent my flat out on Airbnb?" finds "holiday letting" with no
  shared words. That's what embeddings add over keyword search.
- **Recall@5 = 1.00 is a ceiling, not a triumph.** With the tenant filter each agency has
  only 10–22 chunks, so the top 5 covers a quarter to a half of them. **MRR and Recall@1 are
  the honest metrics here**, and headings (0.99) edges out fixed (0.97): one topic per chunk
  ranks more cleanly.
- **Only miss at k=1:** `rr-09` "Do I need to book before coming to see a house?" vs "No
  appointment is needed". The question's meaning is about *booking*, the answer phrase about
  *appointments*; a keyword or reranking step should help (M4–M6).
- **Without the tenant filter MRR drops 0.99 → 0.77:** the three agencies' deposit, pets and
  notice sections are worded alike, so another agency's chunk often outranks the right one.
  Retrieval still "finds" it in the top 5, but a model reading the #1 chunk would answer
  with the wrong agency's policy. That's what M8 makes impossible.
- **Latency is the embedding call, not the database:** turning the question into a vector on
  CPU takes ~0.4–1 s; the pgvector lookup over 64 rows is under a millisecond (no index
  needed until ~10k+ rows).

## Results table (fill this in as you go)

| Retrieval setup | Recall@5 | MRR | p50 latency |
|-----------------|----------|-----|-------------|
| Random, the floor (headings / fixed) | 0.08 / 0.14 | 0.03 / 0.06 | |
| Vector only (headings / fixed) | 1.00 / 1.00 | 0.99 / 0.97 | ~0.4–1 s |
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
