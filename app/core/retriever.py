"""
Hybrid retrieval: semantic (cosine) + keyword (BM25) fused via
Reciprocal Rank Fusion (RRF).

RRF formula:  rrf(d) = Σ_r  1 / (k + rank_r(d))   where k=60

RRF is robust to score-scale differences between the two rankers, so we
don't need to normalise the cosine scores against the BM25 scores — the
ranks carry all the signal we need.
"""

from collections import defaultdict

from app.core.bm25 import BM25
from app.core.pdf_parser import Chunk
from app.core.vector_store import VectorStore

_RRF_K = 60


def _rrf(
    semantic_results: list[tuple[Chunk, float]],
    keyword_results: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    scores: dict[int, float] = defaultdict(float)
    chunk_map: dict[int, Chunk] = {}

    for rank, (chunk, _) in enumerate(semantic_results, start=1):
        scores[chunk.chunk_index] += 1 / (_RRF_K + rank)
        chunk_map[chunk.chunk_index] = chunk

    for rank, (chunk, _) in enumerate(keyword_results, start=1):
        scores[chunk.chunk_index] += 1 / (_RRF_K + rank)
        chunk_map[chunk.chunk_index] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[idx], score) for idx, score in ranked[:top_k]]


def hybrid_search(
    query_vec: list[float],
    query_text: str,
    vector_store: VectorStore,
    bm25: BM25,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> list[tuple[Chunk, float]]:
    """
    Returns up to top_k (chunk, rrf_score) pairs whose best semantic score
    meets the similarity threshold. If no chunk clears the threshold,
    returns an empty list (caller should respond with 'insufficient evidence').
    """
    # Pull more candidates than needed so RRF has good material to work with
    fetch_k = top_k * 3

    semantic = vector_store.search(query_vec, top_k=fetch_k)
    keyword = bm25.search(query_text, top_k=fetch_k)

    if not semantic and not keyword:
        return []

    # Gate on semantic threshold: if the *best* semantic hit is below the
    # threshold we consider the KB irrelevant to this query.
    if semantic and semantic[0][1] < similarity_threshold:
        return []

    return _rrf(semantic, keyword, top_k=top_k)
