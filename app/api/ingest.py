from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.pdf_parser import extract_chunks
from app.models.schemas import IngestResponse
from app.state import bm25, vector_store

router = APIRouter()

_ALLOWED_TYPES = {"application/pdf", "application/octet-stream"}
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


async def _embed(texts: list[str]) -> list[list[float]]:
    from mistralai import Mistral

    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)

    # Mistral embed endpoint accepts up to 16k tokens per batch;
    # we batch in groups of 64 chunks to stay well within limits.
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


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    settings = get_settings()
    total_chunks = 0
    processed_names: list[str] = []

    for upload in files:
        # Basic validation
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail=f"Only PDF files are accepted. Got: {upload.filename}",
            )
        if upload.content_type not in _ALLOWED_TYPES:
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

        import io

        chunks = extract_chunks(
            io.BytesIO(content),
            filename=upload.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            continue  # skip empty / un-parseable PDFs

        texts = [c.text for c in chunks]
        embeddings = await _embed(texts)

        vector_store.add(chunks, embeddings)
        total_chunks += len(chunks)
        processed_names.append(upload.filename)

    # Rebuild BM25 index over the full (updated) store
    bm25.index(vector_store._chunks)

    # Persist to disk so the store survives restarts
    vector_store.save(settings.vector_store_path)

    return IngestResponse(
        message="Ingestion complete.",
        files_processed=len(processed_names),
        chunks_created=total_chunks,
        filenames=processed_names,
    )
