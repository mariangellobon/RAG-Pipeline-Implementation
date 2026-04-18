"""Unit tests for hybrid retrieval and RRF fusion."""

import pytest

from app.core.bm25 import BM25
from app.core.pdf_parser import Chunk
from app.core.retriever import _rrf, hybrid_search
from app.core.vector_store import VectorStore


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(text=text, source="test.pdf", page=1, chunk_index=idx, char_start=0)


def _unit(dim: int, pos: int) -> list[float]:
    v = [0.0] * dim
    v[pos] = 1.0
    return v


class TestRRF:
    def test_document_in_both_lists_ranks_higher(self):
        shared = _chunk("shared document", 0)
        only_semantic = _chunk("only semantic", 1)
        only_keyword = _chunk("only keyword", 2)

        semantic = [(shared, 0.9), (only_semantic, 0.8)]
        keyword = [(shared, 15.0), (only_keyword, 12.0)]

        results = _rrf(semantic, keyword, top_k=3)
        # shared appears in both lists and should rank first
        assert results[0][0].chunk_index == 0

    def test_top_k_respected(self):
        chunks = [_chunk(f"doc {i}", i) for i in range(5)]
        semantic = [(c, 1.0) for c in chunks]
        results = _rrf(semantic, [], top_k=2)
        assert len(results) == 2

    def test_empty_inputs_return_empty(self):
        assert _rrf([], [], top_k=5) == []


class TestHybridSearch:
    def _setup(self, texts: list[str], vecs: list[list[float]]):
        chunks = [_chunk(t, i) for i, t in enumerate(texts)]
        store = VectorStore()
        store.add(chunks, vecs)
        bm25 = BM25()
        bm25.index(chunks)
        return store, bm25, chunks

    def test_returns_empty_when_store_empty(self):
        store = VectorStore()
        bm25 = BM25()
        results = hybrid_search(
            query_vec=_unit(4, 0),
            query_text="anything",
            vector_store=store,
            bm25=bm25,
            similarity_threshold=0.35,
        )
        assert results == []

    def test_below_threshold_returns_empty(self):
        store, bm25, _ = self._setup(
            ["completely unrelated content"], [_unit(4, 3)]
        )
        # query is orthogonal to all stored vectors
        results = hybrid_search(
            query_vec=_unit(4, 0),
            query_text="query",
            vector_store=store,
            bm25=bm25,
            similarity_threshold=0.9,  # very high threshold
        )
        assert results == []

    def test_matching_query_returns_results(self):
        store, bm25, _ = self._setup(
            ["python programming language tutorial"],
            [_unit(4, 0)],
        )
        results = hybrid_search(
            query_vec=_unit(4, 0),
            query_text="python tutorial",
            vector_store=store,
            bm25=bm25,
            similarity_threshold=0.35,
        )
        assert len(results) > 0
