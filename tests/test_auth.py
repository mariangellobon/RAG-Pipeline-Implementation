"""
Tests for API key authentication.

The auth dependency is applied to /api/ingest and /api/query.
/health is intentionally public.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client_auth_enabled():
    os.environ["RAG_API_KEY"] = "secret-key"
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    os.environ.pop("RAG_API_KEY", None)
    get_settings.cache_clear()


class TestHealthIsPublic:
    def test_health_requires_no_key(self, client_auth_enabled):
        response = client_auth_enabled.get("/health")
        assert response.status_code == 200


class TestQueryAuth:
    def test_missing_key_returns_403(self, client_auth_enabled):
        response = client_auth_enabled.post("/api/query", json={"query": "hello"})
        assert response.status_code == 403

    def test_wrong_key_returns_403(self, client_auth_enabled):
        response = client_auth_enabled.post(
            "/api/query",
            json={"query": "hello"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403

    def test_correct_key_passes_auth(self, client_auth_enabled):
        # Patch intent detection so the request doesn't hit Mistral
        with patch("app.api.query.detect_intent", return_value="GREETING"), \
             patch("app.core.generator._conversational", return_value="Hello!"):
            response = client_auth_enabled.post(
                "/api/query",
                json={"query": "hello"},
                headers={"X-API-Key": "secret-key"},
            )
        assert response.status_code == 200

    def test_auth_disabled_when_no_key_set(self, client):
        # client fixture has no RAG_API_KEY — all requests pass through
        with patch("app.api.query.detect_intent", return_value="GREETING"), \
             patch("app.core.generator._conversational", return_value="Hello!"):
            response = client.post("/api/query", json={"query": "hello"})
        assert response.status_code == 200


class TestIngestAuth:
    def test_missing_key_returns_403(self, client_auth_enabled):
        response = client_auth_enabled.post(
            "/api/ingest",
            files=[("files", ("test.pdf", b"%PDF-1.4 fake", "application/pdf"))],
        )
        assert response.status_code == 403

    def test_wrong_key_returns_403(self, client_auth_enabled):
        response = client_auth_enabled.post(
            "/api/ingest",
            files=[("files", ("test.pdf", b"%PDF-1.4 fake", "application/pdf"))],
            headers={"X-API-Key": "bad-key"},
        )
        assert response.status_code == 403
