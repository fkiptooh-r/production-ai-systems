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
- [x] **M4 · Keyword search.** BM25 (`rank_bm25`) or Postgres full-text search. Measure. Notice which questions keyword search wins (names, codes, exact numbers).
- [x] **M5 · Hybrid.** Combine both with RRF. Measure.
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
uv run evaluate --searcher keyword [--mode all]  # no Ollama needed
uv run evaluate --searcher hybrid [--rrf-k 60] [--keyword-weight 1.0]
uv run compare                                 # vector vs keyword vs hybrid, per question
```

**Look at the data yourself:** connect DBeaver (or any SQL client) to `127.0.0.1:5433`,
database `ai_systems`, user/password `postgres`/`postgres`, then open
[`sql/explore.sql`](sql/explore.sql): chunks, embeddings, nearest neighbours, word stems.

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

### M4 · Keyword search (Postgres full-text, headings chunks)

| Setup (tenant filter on unless stated) | Recall@5 | Recall@1 | MRR | p50 latency |
|----------------------------------------|----------|----------|-----|-------------|
| Keyword, **all** words must match | 0.22 | | 0.22 | ~30 ms |
| Keyword, **any** word, ranked | 0.90 | 0.78 | 0.83 | ~25 ms |
| Keyword, any word, all tenants | 0.86 | | 0.67 | ~25 ms |
| *Vector, for comparison* | *1.00* | *0.98* | *0.99* | *~500 ms* |

By kind (any mode): direct 1.00 / MRR 0.97 · exact 1.00 / 1.00 · **paraphrase 0.79 / 0.66**.

- **"All words" is useless for questions:** 0.22, and **0.00 on paraphrases**. "Can I keep a
  cat in my bedsitter?" needs one chunk with *keep* + *cat* + *bedsitter*; none has all
  three. Switching `&` to `|` ("any word, ranked") lifts recall to 0.90.
- **Keyword fails exactly where words differ:** all 5 misses are paraphrases. "Airbnb" vs
  "holiday letting", "charge anything to apply" vs "application fee", "Can I bring my dog?"
  vs Pwani's "Pets are allowed only with ... approval" (*dog* never appears).
- **But it fixes vector's one miss:** `rr-09` "Do I need to **book**...?" is rank 2 for vector
  and rank 1 for keyword: the question's stems *need* and *hous* both appear in Ruaka's
  viewings chunk ("No appointment is need**ed**", "quote the **house** number"). Each method
  gets right some questions the other gets wrong, and that's the argument for hybrid (M5).
  Run `uv run compare` for the full list.
- **~20x faster:** no embedding call, just a GIN-indexed query (~25 ms, mostly connection
  setup). Vector spends ~500 ms on the CPU embedding the question.
- **Honest limitation of this eval:** our `exact` questions *ask for* a number ("What's the
  paybill?") rather than *contain* one, so vector does just as well on them. Keyword's real
  edge is queries that contain codes, such as a listing reference or "522100", which this
  eval doesn't yet test. Worth adding before claiming hybrid helps with exact matches.

**How it works:** the `search` column is a generated `tsvector` (word stems, stopwords
removed, kept in sync by Postgres) with a GIN index. See
[`sql/explore.sql`](sql/explore.sql) to look at it, and at the embeddings, in DBeaver/psql.

### M5 · Hybrid search (Reciprocal Rank Fusion)

Each method returns its top 20; RRF scores every chunk `weight / (rrf_k + rank)` per list
and sorts by the sum. Ranks, not raw scores, because cosine distance and `ts_rank` aren't on
comparable scales.

| Setup (headings, tenant filter) | Recall@5 | Recall@1 | MRR | p50 latency |
|---------------------------------|----------|----------|-----|-------------|
| Vector only | 1.00 | **0.98** | **0.99** | ~480 ms |
| Keyword only | 0.90 | 0.78 | 0.83 | ~26 ms |
| **Hybrid, standard RRF** (rrf_k 60, equal weights) | 1.00 | 0.92 | 0.95 | ~510 ms |
| Hybrid, rrf_k 10, keyword weight 0.5 | 1.00 | 0.94 | 0.96 | ~510 ms |
| Hybrid, rrf_k 1, keyword weight 0.5 (best found) | 1.00 | 0.96 | 0.98 | ~510 ms |
| Hybrid, standard RRF, all tenants | 0.94 | | 0.75 | ~530 ms |

**Hybrid did *not* beat vector search on this eval, and that's the finding.**

- **What hybrid fixed:** `rr-09` "Do I need to **book**...?" moved from rank 2 (vector) to
  rank 1, thanks to keyword's match on *need* and *house*.
- **What it broke:** 4 paraphrases that vector ranked 1st fell to 2nd–5th (`nh-09` Airbnb,
  `nh-17` commute, `rr-11` "charge anything to apply", `pl-10` "without travelling").
  Keyword ranked the wrong chunk 1st for these, and equal-weight RRF let it outvote vector.
- **Why weights barely helped at first:** each agency has ~21 chunks, so 20 candidates per
  method means almost *every* chunk is in *both* lists. With rrf_k = 60, rank 1 and rank 4
  score nearly the same (1/61 vs 1/64), so the fused order is mostly noise from the weaker
  list. A smaller rrf_k sharpens the top ranks; down-weighting keyword limits the damage.
- **Why I kept the standard defaults:** the "best found" row was chosen by scoring settings
  on this same 51-question exam. Choosing settings that way is overfitting: they'd look
  great here and might not generalise. Tuning needs a separate dev set.
- **When hybrid *does* win:** when keyword search is right on a meaningful share of queries:
  product codes, SKUs, names, error codes, rare jargon. This eval is mostly paraphrased
  questions about prose, which is vector's home ground, and its `exact` questions ask *for*
  numbers instead of containing them (M4's eval gap). Run hybrid on your e-commerce catalogue,
  where people type "SKU-4471" or a brand name, and the result is likely to flip.
- **Cost:** hybrid latency is vector + keyword run one after the other (~510 ms). Running
  them concurrently would bring it down to roughly the vector time.

Try it: `uv run compare` (vector vs keyword vs hybrid, question by question), or
`uv run evaluate --searcher hybrid --rrf-k 10 --keyword-weight 0.5 --k 1`.

## Results table (fill this in as you go)

| Retrieval setup | Recall@5 | MRR | p50 latency |
|-----------------|----------|-----|-------------|
| Random, the floor (headings / fixed) | 0.08 / 0.14 | 0.03 / 0.06 | |
| Vector only (headings / fixed) | 1.00 / 1.00 | 0.99 / 0.97 | ~0.4–1 s |
| Keyword only (any word, headings) | 0.90 | 0.83 | ~25 ms |
| Hybrid (RRF, rrf_k 60, equal weights) | 1.00 | 0.95 | ~510 ms |
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
