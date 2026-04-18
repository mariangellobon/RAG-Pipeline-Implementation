"""
Shared FastAPI dependencies.

Auth strategy: static API key passed as X-API-Key request header.
If RAG_API_KEY is not set in the environment, auth is disabled and all
requests are allowed through (useful for local development).
When set, any request to a protected endpoint without the correct key
receives a 403 Forbidden — not 401, to avoid leaking whether auth exists.
"""

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    settings = get_settings()
    if not settings.rag_api_key:
        return  # auth disabled in dev mode
    if api_key != settings.rag_api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
