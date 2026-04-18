from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-small-latest"

    # Auth — if unset, auth is disabled (dev mode)
    rag_api_key: Optional[str] = None

    # CORS — comma-separated origins, or "*" for dev
    allowed_origins: str = "*"

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.35
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Hallucination filter
    hallucination_rejection_threshold: float = 0.4

    vector_store_path: str = "data/vector_store.pkl"

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
