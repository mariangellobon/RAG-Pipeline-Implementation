"""
Tests for POST /api/ingest.

PDF parsing and Mistral embed calls are mocked.
Tests cover validation, content-type checking, and the happy path.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.pdf_parser import Chunk


def _fake_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(text=f"chunk text number {i} with enough words", source="test.pdf",
              page=1, chunk_index=i, char_start=i * 50)
        for i in range(n)
    ]


_FAKE_PDF = b"%PDF-1.4 minimal pdf content for testing"
_FAKE_EMBEDDINGS = [[0.1] * 1024] * 3


class TestIngestValidation:
    def test_non_pdf_extension_rejected(self, client):
        r = client.post(
            "/api/ingest",
            files=[("files", ("document.txt", b"hello", "text/plain"))],
        )
        assert r.status_code == 415

    def test_wrong_content_type_rejected(self, client):
        r = client.post(
            "/api/ingest",
            files=[("files", ("doc.pdf", _FAKE_PDF, "text/html"))],
        )
        assert r.status_code == 415

    def test_no_files_rejected(self, client):
        # FastAPI will return 422 when the required `files` field is missing
        r = client.post("/api/ingest")
        assert r.status_code == 422


class TestIngestHappyPath:
    def test_valid_pdf_returns_ingest_response(self, client):
        with patch("app.api.ingest.extract_chunks", return_value=_fake_chunks(3)), \
             patch("app.api.ingest._embed", return_value=_FAKE_EMBEDDINGS):
            r = client.post(
                "/api/ingest",
                files=[("files", ("report.pdf", _FAKE_PDF, "application/pdf"))],
            )
        assert r.status_code == 200
        data = r.json()
        assert data["files_processed"] == 1
        assert data["chunks_created"] == 3
        assert "report.pdf" in data["filenames"]

    def test_multiple_files_processed(self, client):
        with patch("app.api.ingest.extract_chunks", return_value=_fake_chunks(2)), \
             patch("app.api.ingest._embed", return_value=[[0.1] * 1024, [0.2] * 1024]):
            r = client.post(
                "/api/ingest",
                files=[
                    ("files", ("a.pdf", _FAKE_PDF, "application/pdf")),
                    ("files", ("b.pdf", _FAKE_PDF, "application/pdf")),
                ],
            )
        assert r.status_code == 200
        data = r.json()
        assert data["files_processed"] == 2

    def test_empty_pdf_skipped_gracefully(self, client):
        with patch("app.api.ingest.extract_chunks", return_value=[]):
            r = client.post(
                "/api/ingest",
                files=[("files", ("empty.pdf", _FAKE_PDF, "application/pdf"))],
            )
        # No processable content → 422
        assert r.status_code == 422
