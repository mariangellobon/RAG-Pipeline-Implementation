# RAG Pipeline — FastAPI + Mistral AI

A Python backend for a Retrieval-Augmented Generation (RAG) system over PDF files.  
Users upload documents, ask questions in natural language, and receive grounded answers with source citations.

Built without any external RAG framework or third-party vector database — all retrieval logic is implemented from scratch.

---

## System Design

### High-Level Architecture

Both endpoints use `POST` because both receive data from the client in the request body —
`/api/ingest` receives PDF files, `/api/query` receives a JSON payload `{ "query": "..." }`.
`POST` means "send data to be processed", which applies to both.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser UI)                        │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │                                   │
    POST /api/ingest                      POST /api/query
    (send PDF files)                      (send JSON query)
               │                                   │
┌──────────────▼───────────────────────────────────▼───────────────────┐
│                            FastAPI Backend                           │
│                                                                      │
│  ┌─────────────────────┐           ┌──────────────────────────────┐  │
│  │  Ingestion Pipeline │           │       Query Pipeline          │  │
│  │                     │           │                              │  │
│  │  Extract → Chunk    │  [1] WRITE│  Intent Detection            │  │
│  │          ↓          │ ─────────►│       ↓                      │  │
│  │  [1] Embed chunks   │           │  Transform query             │  │
│  │  (Mistral embed)    │           │       ↓                      │  │
│  │          ↓          │           │  [2] Embed query             │  │
│  │   Store vectors     │           │  (Mistral embed)             │  │
│  │   + BM25 index      │           │       ↓                      │  │
│  └─────────────────────┘           │  [3] READ from Storage       │  │
│                                    │  Semantic + BM25 search      │  │
│  ┌─────────────────────┐  [3] READ │       ↓                      │  │
│  │    Storage Layer     │◄─────────│  RRF Re-rank                 │  │
│  │                      │          │       ↓                      │  │
│  │  VectorStore (numpy) │          │  Threshold Gate              │  │
│  │  BM25 Index          │          │       ↓                      │  │
│  │  Pickle on disk      │          │  [4] Generate answer         │  │
│  └─────────────────────┘          │  (Mistral chat)              │  │
│                                    │       ↓                      │  │
│                                    │  Hallucination check         │  │
│                                    └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

Mistral AI API is called at 4 numbered points above:
  [1] Ingestion  — embed each text chunk (mistral-embed)
  [2] Querying   — embed the user's query (mistral-embed)
  [3]            — Storage is READ here (no Mistral call, pure numpy/BM25)
  [4] Querying   — generate the final answer (mistral-small-latest)
```

---

### Ingestion Pipeline

```
PDF File(s)
    │
    ▼
┌───────────────┐
│  pdfplumber   │  ← page-by-page text extraction
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Chunker     │  ← fixed-size windows, 512 chars, 64-char overlap
└───────┬───────┘
        │  list of Chunk(text, source, page, index)
        ▼
┌───────────────┐
│Mistral Embed  │  ← batch calls to mistral-embed, 64 chunks/batch
└───────┬───────┘
        │  float32 vectors, shape (N, 1024)
        ▼
┌──────────────────────────────┐
│  VectorStore   │  BM25 Index │  ← both updated, store persisted to disk
└──────────────────────────────┘
```

---

### Query Pipeline

```
User Query
    │
    ▼
┌──────────────────────┐
│   Intent Detection   │  ← Mistral LLM call (temp=0)
└──────────┬───────────┘
           │
    ┌──────┴───────────────────────────────────┐
    │              Intent routing              │
    │                                          │
  GREETING /           REFUSAL            KB_SEARCH
  CONVERSATIONAL          │                   │
    │                     ▼                   ▼
    │              Return policy        Query Transform
    │              message              (rewrite for retrieval)
    │                                        │
    ▼                                        ▼
Direct LLM                          Embed transformed query
response                                    │
    │                          ┌────────────┼────────────┐
    │                          ▼                         ▼
    │                  Semantic Search             BM25 Search
    │                  (cosine similarity)         (keyword match)
    │                          │                         │
    │                          └────────────┬────────────┘
    │                                       ▼
    │                              RRF Fusion & Re-rank
    │                                       │
    │                              Similarity Threshold
    │                               (score ≥ 0.35?)
    │                              /                \
    │                            YES                NO
    │                             │                  │
    │                             ▼                  ▼
    │                       LLM Generation    "Insufficient
    │                             │             evidence"
    │                             ▼
    │                    Hallucination Check
    │                             │
    └─────────────────────────────┤
                                  ▼
                       Response + Citations
```

---

## How It Operates

### 1. Data Ingestion (`POST /api/ingest`)

PDFs are extracted page by page using `pdfplumber`. Each page's text is split into fixed-size overlapping character windows (default: 512 chars, 64-char overlap). Each chunk carries its source filename and page number for citation purposes.

Chunks are embedded in batches using Mistral's `mistral-embed` model, producing 1024-dimensional vectors. These are L2-normalised and stored in a numpy array alongside the original chunk objects. A BM25 index is rebuilt over all chunks to support keyword retrieval. The store is pickled to disk so it survives restarts.

**Chunking design rationale:** See `DECISIONS.txt` § 1 for full reasoning on window size, overlap percentage, and the filtering of near-empty pages.

### 2. Query Processing (`POST /api/query`)

Every query goes through four stages before an answer is generated:

**Intent detection** — A Mistral LLM call (zero-shot, temperature=0) classifies the query into one of four intents:

| Intent | Behaviour |
|---|---|
| `GREETING` | Answered conversationally, no retrieval |
| `CONVERSATIONAL` | Answered conversationally, no retrieval |
| `KB_SEARCH` | Full retrieval pipeline triggered |
| `REFUSAL` | Query declined (PII, legal/medical content) |

**Query transformation** — For `KB_SEARCH` queries, a second LLM call rewrites the user's question into a retrieval-optimised form: filler words removed, abbreviations expanded, domain terms preserved. This bridges the vocabulary gap between how users ask questions and how documents are written.

**Hybrid retrieval** — Two independent rankers run over the same corpus:
- *Semantic search*: cosine similarity between the query embedding and all stored vectors (pure numpy, O(N) with `argpartition`)
- *Keyword search*: BM25 scoring using a from-scratch implementation (Robertson IDF, k₁=1.5, b=0.75)

Results are fused using **Reciprocal Rank Fusion (RRF)**:

```
rrf_score(doc) = Σ  1 / (60 + rank_in_ranker)
```

RRF is rank-based, so it avoids the scale mismatch between cosine scores (bounded 0–1) and BM25 scores (unbounded). Documents appearing in both result lists are naturally promoted.

**Similarity threshold gate** — If the best semantic score from the vector store falls below `SIMILARITY_THRESHOLD` (default 0.35), the system returns `"Insufficient evidence"` rather than allowing the LLM to speculate.

### 3. Answer Generation

Retrieved chunks are assembled into a context block with source labels. A Mistral chat completion call produces a factual answer with inline citations (`[Source: filename, p.N]`).

A second post-hoc LLM call acts as a hallucination filter: it checks each sentence in the answer against the context chunks and surfaces any claims not supported by the retrieved evidence.

---

## Project Structure

```
rag-pipeline/
├── app/
│   ├── main.py               FastAPI app, lifespan (load/save store), health endpoint
│   ├── state.py              Shared singletons: VectorStore + BM25
│   ├── api/
│   │   ├── ingest.py         POST /api/ingest
│   │   └── query.py          POST /api/query
│   ├── core/
│   │   ├── pdf_parser.py     pdfplumber extraction + overlapping chunker
│   │   ├── vector_store.py   numpy vector store, cosine similarity, pickle persistence
│   │   ├── bm25.py           BM25 from scratch (no rank_bm25)
│   │   ├── retriever.py      Hybrid search + RRF fusion
│   │   ├── intent.py         Intent detection + query transformation
│   │   ├── generator.py      LLM answer generation + hallucination filter
│   │   ├── embeddings.py     Mistral embed wrapper
│   │   └── config.py         Settings from .env (Pydantic)
│   └── models/
│       └── schemas.py        Pydantic request/response models
├── ui/
│   └── index.html            Chat UI (file upload + Q&A)
├── DECISIONS.txt             Full design rationale and chunking considerations
├── .env.example
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Set MISTRAL_API_KEY in .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | Required |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Embedding model |
| `MISTRAL_CHAT_MODEL` | `mistral-small-latest` | Chat model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum score to trigger generation |
| `TOP_K` | `5` | Chunks returned per query |

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

| URL | Description |
|---|---|
| `http://localhost:8000/` | Chat UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/health` | Store status |

---

## API Reference

### `POST /api/ingest`
Upload one or more PDF files. Accepts `multipart/form-data` with field name `files`.

**Response:**
```json
{
  "message": "Ingestion complete.",
  "files_processed": 2,
  "chunks_created": 145,
  "filenames": ["report.pdf", "financials.pdf"]
}
```

### `POST /api/query`
Ask a question over the ingested knowledge base. Accepts JSON.

**Request:** `{ "query": "What are the key risks?", "top_k": 5 }`

**Response:**
```json
{
  "answer": "The key risks include... [Source: report.pdf, p.4]",
  "intent": "KB_SEARCH",
  "search_triggered": true,
  "query_used": "key risks identified in the report",
  "citations": [
    { "source": "report.pdf", "page": 4, "score": 0.631, "text_snippet": "..." }
  ]
}
```

---

## Libraries Used

| Library | Purpose | Link |
|---|---|---|
| FastAPI | API framework | https://fastapi.tiangolo.com |
| Uvicorn | ASGI server | https://www.uvicorn.org |
| Mistral AI SDK | Embeddings + chat completions | https://docs.mistral.ai |
| pdfplumber | PDF text extraction | https://github.com/jsvine/pdfplumber |
| NumPy | Vector math (cosine similarity, storage) | https://numpy.org |
| Pydantic / pydantic-settings | Data validation + config | https://docs.pydantic.dev |

---

## Design Decisions

Full rationale for all decisions: chunking strategy, BM25 from scratch, RRF fusion, threshold gating, hallucination filter, security, and scalability notes is documented in [`DECISIONS.txt`](DECISIONS.txt).
