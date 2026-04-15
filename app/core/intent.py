"""
Intent detection and query transformation via Mistral.

Intent categories:
  - GREETING     : no search needed ("hello", "thanks")
  - CONVERSATIONAL: general chat not about the KB ("what can you do?")
  - KB_SEARCH    : question that should trigger retrieval
  - REFUSAL      : PII / legal / medical queries we won't answer

Query transformation rewrites the user's raw question into a retrieval-
optimised form (more noun-heavy, removes filler words, expands abbreviations).
"""

from mistralai import Mistral

from app.core.config import get_settings

_INTENT_PROMPT = """\
Classify the following user message into exactly one of these intents:
GREETING, CONVERSATIONAL, KB_SEARCH, REFUSAL

Rules:
- GREETING: simple greetings, thanks, farewells (e.g. "hello", "thanks!")
- CONVERSATIONAL: general questions not requiring document lookup (e.g. "what can you do?")
- KB_SEARCH: any question that would benefit from searching the knowledge base
- REFUSAL: requests involving PII (SSNs, passwords), legal advice, or medical diagnoses

Respond with ONLY the intent label, nothing else.

Message: {query}
"""

_TRANSFORM_PROMPT = """\
Rewrite the following question into a concise, keyword-rich search query
optimised for dense retrieval over a document corpus.
- Remove filler words and conversational phrasing
- Expand abbreviations
- Keep all domain-specific terms
- Output ONLY the rewritten query, no explanation

Original question: {query}
"""


def _call(client: Mistral, prompt: str, model: str) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def detect_intent(query: str) -> str:
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)
    raw = _call(client, _INTENT_PROMPT.format(query=query), settings.mistral_chat_model)
    # guard against model being verbose
    for label in ("GREETING", "CONVERSATIONAL", "KB_SEARCH", "REFUSAL"):
        if label in raw.upper():
            return label
    return "KB_SEARCH"   # default to searching if uncertain


def transform_query(query: str) -> str:
    """Rewrite user question into a retrieval-optimised form."""
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)
    return _call(client, _TRANSFORM_PROMPT.format(query=query), settings.mistral_chat_model)
