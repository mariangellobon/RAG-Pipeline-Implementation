"""
Shared application state — singletons for the vector store and BM25 index.

store_lock guards all mutations to vector_store and bm25. Reads (search) are
safe without the lock because numpy reads are non-destructive. The lock is
acquired only for the in-memory write + BM25 rebuild, so expensive operations
(PDF parsing, embedding API calls) can still run concurrently.
"""

import asyncio

from app.core.bm25 import BM25
from app.core.vector_store import VectorStore

vector_store = VectorStore()
bm25 = BM25()
store_lock = asyncio.Lock()
