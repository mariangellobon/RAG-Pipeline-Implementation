"""
Shared application state — singletons for the vector store and BM25 index.

Both are loaded from disk on startup (if a persisted store exists) and kept
in memory for the lifetime of the process. Concurrent write access is not
needed at this scale; if it ever is, wrap with asyncio.Lock.
"""

from app.core.bm25 import BM25
from app.core.vector_store import VectorStore

vector_store = VectorStore()
bm25 = BM25()
