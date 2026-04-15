"""
In-memory vector store — no third-party vector DB.

Storage layout (persisted as pickle):
  {
    "chunks":     list[Chunk],
    "embeddings": np.ndarray  shape (N, D)   float32
  }

Cosine similarity is computed with pure numpy:
  sim(a, b) = (a · b) / (||a|| * ||b||)

For N ≤ ~100k chunks this is fast enough on CPU with numpy's BLAS backend.
If scale ever demands it, the interface is identical to what FAISS expects, so
swapping in an ANN index requires only changing this module.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.pdf_parser import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None   # shape (N, D)
        self._sources: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        new_vecs = np.array(embeddings, dtype=np.float32)
        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new_vecs = new_vecs / norms

        self._chunks.extend(chunks)
        self._sources.update(c.source for c in chunks)

        if self._embeddings is None:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
        self._sources = set()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_vec: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine_score) sorted descending."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        q = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        scores = self._embeddings @ q           # (N,)
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"chunks": self._chunks, "embeddings": self._embeddings},
                f,
            )

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self._sources = {c.source for c in self._chunks}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        return sorted(self._sources)
