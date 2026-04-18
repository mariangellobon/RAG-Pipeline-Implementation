"""
Thin wrapper around Mistral's embedding endpoint.
Kept separate so it can be swapped for a local model without touching other modules.
"""

from mistralai.client import Mistral

from app.core.config import get_settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batching to respect API limits."""
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)

    batch_size = 64
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=settings.mistral_embed_model,
            inputs=batch,
        )
        all_embeddings.extend([e.embedding for e in resp.data])

    return all_embeddings
