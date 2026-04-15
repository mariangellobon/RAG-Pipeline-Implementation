"""
Answer generation with Mistral.

Templates switch by intent (answer shaping bonus):
  - KB_SEARCH → factual answer with inline citations
  - CONVERSATIONAL → friendly conversational reply
  - INSUFFICIENT_EVIDENCE → polite refusal

Hallucination filter: after generation, a second LLM call checks each
sentence in the answer against the retrieved context. Sentences not
supported by any context chunk are flagged.
"""

from mistralai import Mistral

from app.core.config import get_settings
from app.core.pdf_parser import Chunk

_KB_SYSTEM = """\
You are a precise assistant that answers questions strictly from the provided context.
- Answer concisely and factually.
- For each claim, add an inline citation like [Source: <filename>, p.<page>].
- If the context does not contain enough information, say so explicitly.
- Do NOT invent facts not present in the context.
"""

_KB_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer (with citations):"""

_CONVERSATIONAL_TEMPLATE = """\
You are a helpful assistant. Answer the following message naturally and concisely.

Message: {question}"""

_HALLUCINATION_CHECK_PROMPT = """\
You are a fact-checker. Given the CONTEXT and ANSWER below, identify any sentences
in the ANSWER that are NOT supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

List unsupported sentences as a JSON array of strings. If all sentences are supported,
return an empty array []. Respond with ONLY the JSON array.
"""


def _build_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in chunks_with_scores:
        parts.append(
            f"[{chunk.source}, p.{chunk.page}] (score: {score:.3f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    intent: str,
    chunks_with_scores: list[tuple[Chunk, float]],
    run_hallucination_check: bool = True,
) -> dict:
    """
    Returns:
      {
        "answer": str,
        "hallucination_warnings": list[str]   # sentences flagged by the checker
      }
    """
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)

    if intent == "CONVERSATIONAL" or not chunks_with_scores:
        answer = _conversational(client, question, settings.mistral_chat_model)
        return {"answer": answer, "hallucination_warnings": []}

    context = _build_context(chunks_with_scores)
    answer = _kb_answer(client, question, context, settings.mistral_chat_model)

    warnings: list[str] = []
    if run_hallucination_check:
        warnings = _hallucination_check(client, context, answer, settings.mistral_chat_model)

    return {"answer": answer, "hallucination_warnings": warnings}


def _kb_answer(client: Mistral, question: str, context: str, model: str) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": _KB_SYSTEM},
            {"role": "user", "content": _KB_USER_TEMPLATE.format(context=context, question=question)},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _conversational(client: Mistral, question: str, model: str) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": _CONVERSATIONAL_TEMPLATE.format(question=question)},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _hallucination_check(
    client: Mistral, context: str, answer: str, model: str
) -> list[str]:
    import json

    prompt = _HALLUCINATION_CHECK_PROMPT.format(context=context, answer=answer)
    raw = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    ).choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []
