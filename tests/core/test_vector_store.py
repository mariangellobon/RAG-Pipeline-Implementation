"""Unit tests for the in-memory numpy vector store."""

import numpy as np
import pytest

from app.core.pdf_parser import Chunk
from app.core.vector_store import VectorStore


def _chunk(idx: int = 0) -> Chunk:
    return Chunk(text=f"chunk {idx}", source="test.pdf", page=1, chunk_index=idx, char_start=0)


def _unit(dim: int, pos: int) -> list[float]:
    """One-hot unit vector along axis `pos`."""
    v = [0.0] * dim
    v[pos] = 1.0
    return v


class TestVectorStore:
    def test_empty_store_returns_nothing(self):
        store = VectorStore()
        assert store.search([1.0, 0.0, 0.0], top_k=5) == []

    def test_identical_vector_has_similarity_one(self):
        store = VectorStore()
        vec = _unit(4, 0)
        store.add([_chunk(0)], [vec])
        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_orthogonal_vectors_have_similarity_zero(self):
        store = VectorStore()
        store.add([_chunk(0)], [_unit(4, 0)])
        results = store.search(_unit(4, 1), top_k=1)
        assert abs(results[0][1]) < 1e-5

    def test_top_k_ordering(self):
        store = VectorStore()
        query = _unit(4, 0)
        # chunk 0: aligned with query → high score
        # chunk 1: orthogonal → zero score
        # chunk 2: partially aligned → medium score
        store.add(
            [_chunk(0), _chunk(1), _chunk(2)],
            [_unit(4, 0), _unit(4, 1), [0.7, 0.7, 0.0, 0.0]],
        )
        results = store.search(query, top_k=3)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0][0].chunk_index == 0

    def test_size_and_sources(self):
        store = VectorStore()
        store.add([_chunk(0), _chunk(1)], [_unit(3, 0), _unit(3, 1)])
        assert store.size == 2
        assert "test.pdf" in store.sources

    def test_zero_query_vector_returns_empty(self):
        store = VectorStore()
        store.add([_chunk(0)], [_unit(3, 0)])
        assert store.search([0.0, 0.0, 0.0], top_k=1) == []
