"""
Pytest configuration and shared fixtures.

Sets required env vars before any app module is imported so Settings()
never raises a validation error during testing. The MISTRAL_API_KEY is
a dummy value — actual Mistral calls are mocked in every test that needs them.
"""

import os

import pytest

# Must be set before app imports resolve Settings()
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-key-not-real")
os.environ.setdefault("RAG_API_KEY", "test-rag-key")
os.environ.setdefault("ALLOWED_ORIGINS", "*")

from app.core.config import get_settings  # noqa: E402 — must come after env setup
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure settings are re-read for each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_store():
    """Clear the shared in-memory store before each test to prevent state leaking."""
    from app.state import bm25, vector_store
    vector_store.clear()
    bm25.index([])
    yield
    vector_store.clear()
    bm25.index([])


@pytest.fixture
def client():
    """Unauthenticated test client (auth disabled — RAG_API_KEY set to empty string)."""
    os.environ["RAG_API_KEY"] = ""   # empty string beats .env file value; auth disabled
    get_settings.cache_clear()
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    os.environ["RAG_API_KEY"] = "test-rag-key"


@pytest.fixture
def authed_client():
    """Authenticated test client with valid X-API-Key header."""
    os.environ["RAG_API_KEY"] = "test-rag-key"
    get_settings.cache_clear()
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": "test-rag-key"})
        yield c
