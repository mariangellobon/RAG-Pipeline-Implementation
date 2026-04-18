import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import verify_api_key
from app.core.config import get_settings
from app.core.pdf_parser import extract_chunks
from app.models.schemas import IngestResponse, ResetResponse
from app.state import bm25, store_lock, vector_store

router = APIRouter()

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


async def _embed(texts: list[str]) -> list[list[float]]:
    from mistralai.client import Mistral

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


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ingest_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    settings = get_settings()

    # --- Validation + parsing + embedding happen OUTSIDE the lock ---
    # These are slow (I/O, API calls) and safe to run concurrently.
    # Multiple ingest requests process their PDFs in parallel; they only
    # serialise at the in-memory write step below.
    prepared: list[tuple] = []  # (chunks, embeddings, filename)

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail=f"Only PDF files are accepted. Got: {upload.filename}",
            )
        if upload.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported content type for {upload.filename}: "
                    f"{upload.content_type}. Expected PDF."
                ),
            )

        content = await upload.read()
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} exceeds the 50 MB limit.",
            )

        chunks = extract_chunks(
            io.BytesIO(content),
            filename=upload.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            continue

        embeddings = await _embed([c.text for c in chunks])
        prepared.append((chunks, embeddings, upload.filename))

    if not prepared:
        raise HTTPException(status_code=422, detail="No processable content found in uploaded files.")

    # --- Lock only for the in-memory mutation ---
    total_chunks = 0
    processed_names: list[str] = []

    async with store_lock:
        for chunks, embeddings, filename in prepared:
            vector_store.add(chunks, embeddings)
            total_chunks += len(chunks)
            processed_names.append(filename)

        bm25.index(vector_store._chunks)
        vector_store.save(settings.vector_store_path)

    return IngestResponse(
        message="Ingestion complete.",
        files_processed=len(processed_names),
        chunks_created=total_chunks,
        filenames=processed_names,
    )


@router.delete(
    "/store",
    response_model=ResetResponse,
    dependencies=[Depends(verify_api_key)],
)
async def reset_store():
    """Clear all ingested documents and reset the knowledge base."""
    import os
    settings = get_settings()

    async with store_lock:
        vector_store.clear()
        bm25.index([])
        if os.path.exists(settings.vector_store_path):
            os.remove(settings.vector_store_path)

    return ResetResponse(message="Knowledge base cleared.")
