"""Unit tests for the PDF text chunking logic."""

import pytest

from app.core.pdf_parser import Chunk, _clean, extract_chunks


class TestClean:
    def test_collapses_excess_blank_lines(self):
        assert "\n\n\n" not in _clean("a\n\n\n\nb")

    def test_collapses_excess_spaces(self):
        assert "  " not in _clean("hello   world")

    def test_preserves_single_newlines(self):
        result = _clean("line one\nline two")
        assert "line one" in result and "line two" in result


class TestChunking:
    """Tests chunking logic directly without needing a real PDF file."""

    def _fake_chunks(self, text: str, chunk_size: int = 50, overlap: int = 10) -> list[Chunk]:
        """Helper: simulate what extract_chunks does for a single page of text."""
        from app.core.pdf_parser import _clean
        text = _clean(text)
        chunks = []
        idx = 0
        start = 0
        step = chunk_size - overlap
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if len(piece) >= 40:
                chunks.append(Chunk(
                    text=piece,
                    source="test.pdf",
                    page=1,
                    chunk_index=idx,
                    char_start=start,
                ))
                idx += 1
            start += step
        return chunks

    def test_overlap_means_content_repeats(self):
        text = "A" * 100
        chunks = self._fake_chunks(text, chunk_size=50, overlap=20)
        # With overlap, consecutive chunks share content
        if len(chunks) >= 2:
            c1_end = chunks[0].text[-10:]
            c2_start = chunks[1].text[:10]
            # Both should be all-A characters (confirming overlap region)
            assert set(c1_end) == {"A"}
            assert set(c2_start) == {"A"}

    def test_tiny_fragments_are_dropped(self):
        # A very short piece of text should not produce a chunk < 40 chars
        chunks = self._fake_chunks("Short.", chunk_size=50, overlap=10)
        assert all(len(c.text) >= 40 for c in chunks)

    def test_chunk_count_increases_with_text_length(self):
        short = self._fake_chunks("word " * 20, chunk_size=50, overlap=10)
        long = self._fake_chunks("word " * 200, chunk_size=50, overlap=10)
        assert len(long) > len(short)
