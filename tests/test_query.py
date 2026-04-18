"""
Tests for POST /api/query.

Mistral API calls are mocked throughout — these tests verify routing,
intent-based branching, threshold gating, and response shapes.
"""

from unittest.mock import patch

import pytest

from app.core.pdf_parser import Chunk


def _make_chunk(text: str = "This is supporting evidence.", idx: int = 0) -> Chunk:
    return Chunk(text=text, source="doc.pdf", page=1, chunk_index=idx, char_start=0)


class TestGreetingIntent:
    def test_greeting_skips_retrieval(self, client):
        with patch("app.api.query.detect_intent", return_value="GREETING"), \
             patch("app.core.generator._conversational", return_value="Hi there!"):
            r = client.post("/api/query", json={"query": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["search_triggered"] is False
        assert data["citations"] == []
        assert data["intent"] == "GREETING"


class TestRefusalIntent:
    def test_refusal_returns_policy_message(self, client):
        with patch("app.api.query.detect_intent", return_value="REFUSAL"):
            r = client.post("/api/query", json={"query": "give me someone's SSN"})
        assert r.status_code == 200
        data = r.json()
        assert data["search_triggered"] is False
        assert "not able to help" in data["answer"].lower()


class TestKBSearch:
    def test_insufficient_evidence_when_store_empty(self, client):
        # Patch hybrid_search at the import location in query.py (not the source module)
        with patch("app.api.query.detect_intent", return_value="KB_SEARCH"), \
             patch("app.api.query.transform_query", return_value="transformed query"), \
             patch("app.api.query._embed_query", return_value=[0.1] * 1024), \
             patch("app.api.query.hybrid_search", return_value=[]):
            r = client.post("/api/query", json={"query": "what is the revenue?"})
        assert r.status_code == 200
        assert "insufficient evidence" in r.json()["answer"].lower()

    def test_successful_kb_answer_has_citations(self, client):
        chunk = _make_chunk("Revenue was $34B in fiscal year 2024.")
        mock_gen = {
            "answer": "Revenue was $34B. [Source: doc.pdf, p.1]",
            "answer_rejected": False,
            "hallucination_warnings": [],
            "consistency_warnings": [],
        }
        with patch("app.api.query.detect_intent", return_value="KB_SEARCH"), \
             patch("app.api.query.transform_query", return_value="revenue fiscal year"), \
             patch("app.api.query._embed_query", return_value=[0.1] * 1024), \
             patch("app.api.query.hybrid_search", return_value=[(chunk, 0.82)]), \
             patch("app.api.query.generate_answer", return_value=mock_gen):
            r = client.post("/api/query", json={"query": "what is the revenue?"})
        assert r.status_code == 200
        data = r.json()
        assert data["search_triggered"] is True
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source"] == "doc.pdf"
        assert data["answer_rejected"] is False

    def test_rejected_answer_returns_rejection_message(self, client):
        chunk = _make_chunk()
        mock_gen = {
            "answer": "some fabricated answer",
            "answer_rejected": True,
            "hallucination_warnings": ["Sentence A", "Sentence B", "Sentence C"],
            "consistency_warnings": [],
        }
        with patch("app.api.query.detect_intent", return_value="KB_SEARCH"), \
             patch("app.api.query.transform_query", return_value="query"), \
             patch("app.api.query._embed_query", return_value=[0.1] * 1024), \
             patch("app.api.query.hybrid_search", return_value=[(chunk, 0.7)]), \
             patch("app.api.query.generate_answer", return_value=mock_gen):
            r = client.post("/api/query", json={"query": "tell me about revenue"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer_rejected"] is True
        assert data["citations"] == []
        assert "verified" in data["answer"].lower()

    def test_consistency_warnings_returned(self, client):
        chunk = _make_chunk()
        mock_gen = {
            "answer": "The company grew 10%.",
            "answer_rejected": False,
            "hallucination_warnings": ["The company grew 10%."],
            "consistency_warnings": ["Answer A says 10% growth, Answer B says 5% decline."],
        }
        with patch("app.api.query.detect_intent", return_value="KB_SEARCH"), \
             patch("app.api.query.transform_query", return_value="growth"), \
             patch("app.api.query._embed_query", return_value=[0.1] * 1024), \
             patch("app.api.query.hybrid_search", return_value=[(chunk, 0.7)]), \
             patch("app.api.query.generate_answer", return_value=mock_gen):
            r = client.post("/api/query", json={"query": "what was the growth?"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["consistency_warnings"]) == 1
        assert len(data["hallucination_warnings"]) == 1


class TestQueryValidation:
    def test_empty_query_rejected(self, client):
        r = client.post("/api/query", json={"query": ""})
        assert r.status_code == 422

    def test_top_k_out_of_range_rejected(self, client):
        r = client.post("/api/query", json={"query": "hello", "top_k": 99})
        assert r.status_code == 422
