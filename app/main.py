from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import ingest, query
from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.state import bm25, vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted vector store on startup
    settings = get_settings()
    vector_store.load(settings.vector_store_path)
    if vector_store.size > 0:
        bm25.index(vector_store._chunks)
    yield
    # Persist on shutdown
    vector_store.save(settings.vector_store_path)


app = FastAPI(
    title="RAG Pipeline",
    description="PDF-based RAG with hybrid retrieval (semantic + BM25) powered by Mistral AI.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api", tags=["ingestion"])
app.include_router(query.router, prefix="/api", tags=["querying"])

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="ok",
        chunks_in_store=vector_store.size,
        files_ingested=vector_store.sources,
    )


# Serve the chat UI
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
