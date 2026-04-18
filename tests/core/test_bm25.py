"""Unit tests for the BM25 keyword search implementation."""

import pytest

from app.core.bm25 import BM25, _tokenize
from app.core.pdf_parser import Chunk


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, source="test.pdf", page=1, chunk_index=idx, char_start=0)


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize("foo, bar.") == ["foo", "bar"]

    def test_keeps_numbers(self):
        assert "2024" in _tokenize("revenue in 2024 grew")


class TestBM25:
    def test_empty_index_returns_nothing(self):
        bm25 = BM25()
        assert bm25.search("anything") == []

    def test_exact_match_scores_higher(self):
        bm25 = BM25()
        chunks = [
            _chunk("The quick brown fox jumps", 0),
            _chunk("A lazy dog sat in the sun", 1),
        ]
        bm25.index(chunks)
        results = bm25.search("fox", top_k=2)
        assert results[0][0].chunk_index == 0
        assert results[0][1] > 0

    def test_no_match_returns_empty(self):
        bm25 = BM25()
        bm25.index([_chunk("The quick brown fox", 0)])
        results = bm25.search("unrelated zebra taxonomy")
        assert results == []

    def test_top_k_respected(self):
        bm25 = BM25()
        chunks = [_chunk(f"document number {i} about python", i) for i in range(10)]
        bm25.index(chunks)
        results = bm25.search("python", top_k=3)
        assert len(results) <= 3

    def test_reindex_replaces_old_data(self):
        bm25 = BM25()
        bm25.index([_chunk("old content about cats", 0)])
        bm25.index([_chunk("new content about dogs", 0)])
        results = bm25.search("cats")
        assert results == []
        assert bm25.search("dogs")[0][1] > 0
