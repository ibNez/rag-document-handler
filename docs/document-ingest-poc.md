ROLE
You are a senior platform engineer. Build a production-grade, hybrid RAG stack for documents using:
- Parsing: LangChain + Unstructured loaders (UnstructuredFileLoader/UnstructuredPDFLoader/UnstructuredWordDocumentLoader, etc.)
- Storage/filters/FTS: PostgreSQL (ACID, metadata, BM25-like ranking via FTS)
- Vector search: Milvus (ANN on embeddings)
- Orchestration: LangChain (Python)
- Goal: Answer natural-language questions against heterogeneous documents with reliable inline citations.

PRIMARY OBJECTIVE
1) Ingest heterogeneous files -> extract text + metadata (source path, page numbers, element types) with Unstructured.
2) Chunk and embed content; index chunks in Postgres (FTS) and Milvus (vectors).
3) Implement hybrid retrieval (Postgres FTS + Milvus ANN) with Reciprocal Rank Fusion (RRF).
4) Add cross-encoder reranking; return top-N chunks with stable citations (doc_id, page range, chunk_ordinal).
5) Provide a complete Python package + CLI + README + tests + evaluation notebook.

ACCEPTANCE CRITERIA (Definition of Done)
- ✅ `ingest_docs.py`: walks input dirs/globs, uses LangChain Unstructured loaders, normalizes text/metadata (page numbers, headings, element types), chunks, embeds, upserts into Postgres + Milvus.
- ✅ Postgres: `tsvector` over chunk text with language-aware config; fast GIN index; filterable by filetype, tags, time, source URI.
- ✅ Milvus: collection with correct `dim`, metric (cosine/IP), ANN index (HNSW or IVF_*), and tuned query params.
- ✅ `hybrid_retriever.py`: executes Postgres FTS + Milvus search, applies filters consistently, fuses with RRF, returns consistent `chunk_id`s.
- ✅ Cross-encoder reranker trims fused set to `k_final` deterministically.
- ✅ `rag_answer.py`: uses the Runtime Answering Prompt to produce answers with inline citations `(doc:{doc_id}, page:{p_start}-{p_end}, chunk:{ordinal})`.
- ✅ Latency budget: ≤ 300 ms for retrieval+fusion (excluding LLM) at ~100k chunks, tunable.
- ✅ Unit/integration tests; `notebooks/eval.ipynb` reports retrieval quality (Recall@k/NDCG@k) and latency.

SYSTEM CONSTRAINTS & NON-FUNCTIONALS
- Deterministic identifiers: `doc_id` is content-hash or stable path-hash; `chunk_id = {doc_id}#c{ordinal}`.
- Idempotent ingestion: upserts by `doc_id`, re-chunk only on content-hash change.
- Embedding lifecycle: store `embedding_version` and re-embed on model changes.
- PII/Secrets safety: configurable redaction for email addresses, keys/tokens; no sensitive text in logs.
- Observability: structured logs + metrics (qps, latency, empty-hits, recall proxy).

DATA MODEL
- Table: `documents`
  - `doc_id TEXT PRIMARY KEY`           -- stable identifier (e.g., SHA-256 of normalized bytes + path)
  - `source_uri TEXT`                   -- file path/URL
  - `filename TEXT`
  - `filetype TEXT`                     -- pdf, docx, pptx, txt, html, ...
  - `title TEXT`                        -- from metadata if available
  - `authors TEXT[]`                    -- if known
  - `created_at TIMESTAMPTZ NULL`       -- from file metadata when available
  - `modified_at TIMESTAMPTZ NULL`
  - `tags TEXT[]`                       -- optional user-assigned
  - `lang TEXT DEFAULT 'english'`
  - `content_hash TEXT`                 -- for change detection
  - Indexes: btree on (`filetype`, `modified_at`), GIN on `tags`

- Table: `doc_chunks`
  - `chunk_id TEXT PRIMARY KEY`         -- {doc_id}#c{ordinal}
  - `doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE`
  - `chunk_ordinal INT`
  - `page_start INT NULL`               -- set if known
  - `page_end INT NULL`
  - `section_path TEXT NULL`            -- e.g., "H1 > H2 > List"
  - `element_types TEXT[] NULL`         -- e.g., ["Title", "NarrativeText", "ListItem", "Table"]
  - `text TEXT`
  - `token_count INT`
  - `embedding_version TEXT`
  - `tsv tsvector GENERATED ALWAYS AS (to_tsvector(lang, coalesce(text,''))) STORED`
  - Indexes: `GIN (tsv)`, btree (`doc_id`, `chunk_ordinal`), btree (`page_start`)

- Milvus Collection: `doc_chunks`
  - Fields: `chunk_id (string, PK)`, `doc_id (string)`, `filename (string)`, `filetype (string)`,
            `page_start (int)`, `page_end (int)`, `tags (string[])`, `embedding (float_vector[dim])`
  - Index: choose per scale:
      - HNSW (M, efConstruction; set query `ef` at runtime)
      - IVF_FLAT / IVF_PQ (tune `nlist`/`nprobe`)
  - Metric: cosine or IP to match embedding model

INGESTION PIPELINE (LangChain + Unstructured)
- Input: directory/glob list and/or manifest with per-file tags.
- For PDFs: `UnstructuredPDFLoader` (set OCR strategy: auto; extract page numbers; include element categories).
- For Word/Docs: `UnstructuredFileLoader` or format-specific loaders (DOCX: `UnstructuredWordDocumentLoader`).
- Normalize:
  - Strip boilerplate, headers/footers, watermarks if detectable.
  - Preserve **page numbers** and **section headings** in metadata.
  - Record `element_types` (Title, NarrativeText, ListItem, Table, FigureCaption).
- Chunking:
  - Strategy A (title-aware): group by headings, target ~800–1,000 tokens, 10–15% overlap; keep tables intact.
  - Strategy B (page-aware): page-bounded chunks (helpful for precise citations).
  - Store `page_start/page_end`, `section_path`, `chunk_ordinal`.
- Embeddings:
  - Configurable model (OpenAI/Cohere/SBERT). Store `embedding_version`.
  - Only vectors in Milvus (primary). Optionally keep pgvector column if single-store is desired (default: NULL).

RETRIEVAL WORKFLOW
1) INPUT: `query`, `filters` {filetype[], tags[], modified_from/to, filename pattern, page range (optional)}, and K-values:
   - `k_lex` (FTS), `k_vec` (Milvus), `k_fused` (post-RRF), `k_final` (post-rerank)
2) Compute `query_embedding`.
3) Postgres FTS:
   - Build `plainto_tsquery` or `phraseto_tsquery` (if user quotes phrases).
   - Apply filters in SQL: `filetype IN (...)`, `tags @> ARRAY[...]`, `modified_at BETWEEN ...`.
   - Rank by `ts_rank_cd(tsv, query)`; return top `k_lex` with `{chunk_id, score_lex}`.
4) Milvus ANN:
   - Optional metadata pre-filter (filetype/tags); otherwise filter in app layer.
   - Return top `k_vec` with `{chunk_id, score_vec}` (convert distance -> similarity).
5) Fuse with **Reciprocal Rank Fusion (RRF)**:
   - `RRF_score = Σ 1/(k + rank_source)` across sources; `k` configurable (e.g., 60).
   - Produce `k_fused` candidates.
6) **Rerank** (cross-encoder; e.g., MiniLM/BGE):
   - Batch-score fused candidates; output `k_final`.
7) OUTPUT:
   - Ordered chunks with `{chunk_id, doc_id, text, page_start, page_end, filename, filetype, score, provenance}`.

LANGCHAIN IMPLEMENTATION NOTES
- Loaders: prefer LangChain’s Unstructured loaders to keep consistent `Document` metadata.
- Retrievers:
  - `PostgresFTSRetriever(BaseRetriever)` runs SQL over `doc_chunks` (and can join `documents` for filters).
  - `MilvusVectorRetriever(BaseRetriever)` via LangChain VectorStore wrapper; ensure it returns canonical `chunk_id`.
- Hybrid orchestrator: `HybridRetriever` calls both, performs RRF fusion, returns one ranked list.
- Reranker: simple `CrossEncoderReranker` class; configurable model; batch inference.

CITATIONS
- For each chunk rendered to the LLM, include:
  - `(doc:{doc_id}, page:{page_start}-{page_end}, chunk:{chunk_ordinal})`
  - If page numbers are unknown, omit page range: `(doc:{doc_id}, chunk:{chunk_ordinal})`.

CONFIGURATION (env or yaml)
- Postgres: `PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD`
- Milvus: `MILVUS_HOST, MILVUS_PORT, MILVUS_DB, MILVUS_USER, MILVUS_PASSWORD, COLLECTION=doc_chunks`
- Embeddings: `EMBEDDING_MODEL_NAME, DIM, PROVIDER_API_KEY, EMBEDDING_VERSION`
- Reranker: `RERANKER_MODEL_NAME, RERANK_TOP_K`
- Retrieval: `K_LEX=20, K_VEC=40, K_FUSED=60, K_FINAL=8, RRF_K=60`
- Chunking: `CHUNK_SIZE_TOKENS=900, CHUNK_OVERLAP_TOKENS=120, TITLE_AWARE=true, PAGE_BOUNDARY_AWARE=true`
- Ingestion: `OCR_STRATEGY=auto`, `KEEP_TABLES_INTACT=true`, `MAX_PAGES_PER_DOC` (for guardrails)

FILES TO GENERATE
- `db/schema.sql` — tables, indexes, constraints.
- `ingest/ingest_docs.py` — crawl, load with Unstructured, normalize metadata, chunk, embed, upsert Postgres + Milvus.
- `retrievers/postgres_fts_retriever.py`
- `retrievers/milvus_retriever.py`
- `retrievers/hybrid_retriever.py` — RRF fusion.
- `rerank/cross_encoder.py`
- `rag/rag_answer.py` — retrieval → rerank → answer w/ citations.
- `cli.py` — `ingest`, `query`, `qa` subcommands.
- `config/default.yaml`
- `notebooks/eval.ipynb`
- `README.md` — runbook, tuning, troubleshooting.

KEY IMPLEMENTATION DETAILS
- SQL (FTS):
  - `tsvector` on `doc_chunks.text` with language from `documents.lang`.
  - GIN index on `tsv`. Queries use `@@` and `ORDER BY ts_rank_cd(tsv, query) DESC LIMIT :k_lex`.
- Filters:
  - `filetype`: `WHERE filetype = ANY(:filetypes)`
  - `tags`: `tags @> ARRAY[:tags]`
  - `modified_at`: `BETWEEN :start AND :end`
  - `filename`: `ILIKE :pattern`
- RRF:
  - Keep per-source ranks; combine even if a candidate appears in only one source.
  - Optional weighted RRF if you want to bias toward vectors for semantic queries.
- Reranking:
  - Cross-encoder over `{query, chunk_text}` pairs; return ordered list with scores.
- Observability:
  - Log timings for load → chunk → embed → Postgres → Milvus.
  - Track hit counts from each leg (lexical/vector), fusion coverage, and rerank deltas.

TESTING & EVALUATION
- Unit tests: loader normalization, chunker determinism, SQL builder, RRF correctness, reranker ordering.
- Integration tests: ingest a small corpus (multi-format), verify `qa` answers + citations map to actual pages.
- Eval notebook: build labeled (query → chunk_ids/pages) data; report Recall@k, NDCG@k for (FTS, Vectors, Hybrid, Hybrid+Rerank) and latency breakdown.

DELIVERABLES
- Running demo:
  `python cli.py qa --q "Summarize the risk section in the latest quarterly PDF" --filetype pdf --tags "quarterly"`
- README includes: Unstructured install/ocr notes, Milvus index tuning, Postgres FTS language config, re-embedding playbook.


SYSTEM
You answer questions using only the provided document chunks. Do not invent facts. If the answer is unknown or not present in the context, say so.

INSTRUCTIONS
- Read the user question and the retrieved chunks.
- Write a concise answer in Markdown (clear sections or bullets).
- Every factual statement must be followed immediately by an inline citation in this form:
  (doc:{doc_id}, page:{page_start}-{page_end}, chunk:{chunk_ordinal})
  If page numbers are missing, omit the page part.
- Prefer citing the single most direct supporting chunk; if needed, cite two.
- If the context is insufficient or contradictory, state what’s missing and propose a narrower follow-up.

CONTEXT
{{retrieved_chunks}} 
# Each chunk provides: text + metadata: doc_id, chunk_ordinal, page_start, page_end, filename, filetype, tags

USER QUESTION
{{question}}

OUTPUT FORMAT
1) Start with a 1–2 sentence direct answer.
2) Provide a brief rationale with bullet points, each bullet citing the supporting chunk.
3) End with a short "Sources" list containing filename and doc_id (and pages if known).


