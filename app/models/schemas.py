from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    filenames: list[str]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    source: str
    page: int
    chunk_index: int
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation]
    search_triggered: bool
    query_used: str
    answer_rejected: bool
    hallucination_warnings: list[str]   # sentences that lack a supporting quote (layer 1)
    consistency_warnings: list[str]     # contradictions found across two generations (layer 3)


class HealthResponse(BaseModel):
    status: str
    chunks_in_store: int
    files_ingested: list[str]
