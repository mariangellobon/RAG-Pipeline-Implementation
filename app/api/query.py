from fastapi import APIRouter, HTTPException
from mistralai import Mistral

from app.core.config import get_settings
from app.core.generator import generate_answer
from app.core.intent import detect_intent, transform_query
from app.core.retriever import hybrid_search
from app.models.schemas import Citation, QueryRequest, QueryResponse
from app.state import bm25, vector_store

router = APIRouter()

_NO_EVIDENCE_MSG = (
    "Insufficient evidence: the knowledge base does not contain enough relevant "
    "information to answer this question confidently."
)


async def _embed_query(text: str) -> list[float]:
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)
    resp = client.embeddings.create(
        model=settings.mistral_embed_model,
        inputs=[text],
    )
    return resp.data[0].embedding


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    settings = get_settings()
    query_text = request.query.strip()

    # 1. Intent detection
    intent = detect_intent(query_text)

    if intent == "REFUSAL":
        return QueryResponse(
            answer=(
                "I'm not able to help with that. "
                "This system does not provide legal, medical, or personal data advice."
            ),
            intent=intent,
            citations=[],
            search_triggered=False,
            query_used=query_text,
        )

    if intent in ("GREETING", "CONVERSATIONAL"):
        from app.core.generator import generate_answer as gen

        result = gen(query_text, intent=intent, chunks_with_scores=[])
        return QueryResponse(
            answer=result["answer"],
            intent=intent,
            citations=[],
            search_triggered=False,
            query_used=query_text,
        )

    # 2. Query transformation for KB_SEARCH
    transformed = transform_query(query_text)

    # 3. Embed transformed query
    query_vec = await _embed_query(transformed)

    # 4. Hybrid retrieval
    results = hybrid_search(
        query_vec=query_vec,
        query_text=transformed,
        vector_store=vector_store,
        bm25=bm25,
        top_k=request.top_k,
        similarity_threshold=settings.similarity_threshold,
    )

    if not results:
        return QueryResponse(
            answer=_NO_EVIDENCE_MSG,
            intent=intent,
            citations=[],
            search_triggered=True,
            query_used=transformed,
        )

    # 5. Generate answer
    gen_result = generate_answer(
        question=query_text,
        intent=intent,
        chunks_with_scores=results,
    )

    answer = gen_result["answer"]
    warnings = gen_result.get("hallucination_warnings", [])
    if warnings:
        disclaimer = "\n\n---\n**Note:** The following claims could not be verified against the source documents:\n"
        disclaimer += "\n".join(f"- {w}" for w in warnings)
        answer += disclaimer

    citations = [
        Citation(
            source=chunk.source,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            score=round(score, 4),
            text_snippet=chunk.text[:200],
        )
        for chunk, score in results
    ]

    return QueryResponse(
        answer=answer,
        intent=intent,
        citations=citations,
        search_triggered=True,
        query_used=transformed,
    )
