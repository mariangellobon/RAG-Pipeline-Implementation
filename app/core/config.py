from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-small-latest"

    top_k: int = 5
    similarity_threshold: float = 0.35
    chunk_size: int = 512
    chunk_overlap: int = 64

    vector_store_path: str = "data/vector_store.pkl"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
