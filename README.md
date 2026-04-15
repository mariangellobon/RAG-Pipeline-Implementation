# RAG Pipeline (FastAPI + Mistral)

A simple PDF-based Retrieval-Augmented Generation (RAG) system built for the StackAI take-home technical exercise.

The system lets users upload PDF files, indexes them for retrieval, and answers questions with grounded responses and citations.

## Features

- FastAPI backend with two required endpoints:
  - `POST /api/ingest` for PDF ingestion
  - `POST /api/query` for question answering
- PDF parsing and chunking with configurable chunk size/overlap
- Query intent detection and query transformation
- Hybrid retrieval:
  - Semantic search (Mistral embeddings + cosine similarity)
  - Keyword search (custom BM25 implementation)
  - Reciprocal Rank Fusion (RRF) for re-ranking
- "Insufficient evidence" threshold gate
- Chat UI for file upload + Q&A
- Citation metadata in query responses
- No third-party vector database
- No external RAG orchestration framework

## Architecture

1. Upload PDFs via `POST /api/ingest`
2. Extract and clean text page-by-page
3. Chunk text into overlapping windows
4. Generate embeddings with Mistral
5. Store chunks + vectors in an in-memory numpy vector store (persisted to disk)
6. Build BM25 index over all chunks
7. Query flow (`POST /api/query`):
   - Detect intent
   - Transform user query for retrieval
   - Embed transformed query
   - Run hybrid retrieval (semantic + BM25)
   - Fuse/re-rank with RRF
   - Apply similarity threshold gate
   - Generate answer with Mistral from retrieved context
   - Return answer + citations

## Project Structure

```text
app/
  api/
    ingest.py
    query.py
  core/
    pdf_parser.py
    vector_store.py
    bm25.py
    retriever.py
    intent.py
    generator.py
    config.py
  models/
    schemas.py
  main.py
  state.py
ui/
  index.html
DECISIONS.txt
PART3_REQUIREMENTS_GUIDE.md
```

## API

### `POST /api/ingest`

Upload one or more PDF files using multipart form data key: `files`.

Example:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/file2.pdf"
```

Response:

```json
{
  "message": "Ingestion complete.",
  "files_processed": 2,
  "chunks_created": 145,
  "filenames": ["file1.pdf", "file2.pdf"]
}
```

### `POST /api/query`

Ask a question over the ingested knowledge base.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key risks?", "top_k":5}'
```

Response shape:

```json
{
  "answer": "...",
  "intent": "KB_SEARCH",
  "citations": [
    {
      "source": "file1.pdf",
      "page": 12,
      "chunk_index": 44,
      "score": 0.61,
      "text_snippet": "..."
    }
  ],
  "search_triggered": true,
  "query_used": "key risks discussed in the report"
}
```

### `GET /health`

Returns service and index health:

```json
{
  "status": "ok",
  "chunks_in_store": 145,
  "files_ingested": ["file1.pdf", "file2.pdf"]
}
```

## Setup

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment variables

Copy `.env.example` to `.env` and set values:

```bash
cp .env.example .env
```

Required:

- `MISTRAL_API_KEY`

Common optional tuning:

- `MISTRAL_EMBED_MODEL` (default `mistral-embed`)
- `MISTRAL_CHAT_MODEL` (default `mistral-small-latest`)
- `TOP_K` (default `5`)
- `SIMILARITY_THRESHOLD` (default `0.35`)
- `CHUNK_SIZE` (default `512`)
- `CHUNK_OVERLAP` (default `64`)
- `VECTOR_STORE_PATH` (default `data/vector_store.pkl`)

### 3) Run

```bash
uvicorn app.main:app --reload
```

Open:

- App UI: <http://localhost:8000/>
- API docs: <http://localhost:8000/docs>

## Design Notes

- Chunking and retrieval considerations are documented in `DECISIONS.txt`.
- A full Part 3 requirement walkthrough is in `PART3_REQUIREMENTS_GUIDE.md`.

## Libraries and Software Used

- FastAPI: <https://fastapi.tiangolo.com/>
- Uvicorn: <https://www.uvicorn.org/>
- Mistral AI API and SDK:
  - <https://docs.mistral.ai/>
  - <https://github.com/mistralai/client-python>
- pdfplumber: <https://github.com/jsvine/pdfplumber>
- NumPy: <https://numpy.org/>
- Pydantic / pydantic-settings:
  - <https://docs.pydantic.dev/>
  - <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>

## Constraints Compliance

- Uses FastAPI (required)
- Uses Mistral API for embeddings and generation (required)
- Implements retrieval logic directly without external RAG/search frameworks
- Uses an in-process vector store (no third-party vector DB)

## Known Limitations

- Current storage/index approach is single-process and memory-bound
- BM25 is rebuilt across full corpus on each ingestion
- Embedding calls are synchronous within request flow
- CORS is permissive for local development (`*`)

These are intentional tradeoffs for a simple, interview-friendly implementation.
