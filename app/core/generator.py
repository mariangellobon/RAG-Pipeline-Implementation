"""
Answer generation with Mistral.

Hallucination detection — three-layer approach:

  Layer 1 — Fact-introduction check
    The model reviews each sentence and flags it only if it introduces a NEW
    specific fact (number, name, date, statistic, event) that is absent from
    and not inferable from the retrieved context.

    Explanatory sentences — elaborations, logical inferences, summaries of
    quoted facts — are NOT flagged even without a verbatim quote. A
    10%-quote / 90%-explanation answer is fine as long as the explanation
    does not introduce new uncited specifics. Only fabricated facts are flagged.

  Layer 2 — Rejection threshold
    If more than HALLUCINATION_REJECTION_THRESHOLD (default 40%) of checkable
    sentences are flagged, the answer is rejected entirely. The caller returns
    a clear rejection message rather than serving a partially fabricated answer.

  Layer 3 — Self-consistency check (borderline cases)
    When the answer passes the rejection threshold but has at least one flagged
    sentence, a second independent generation is run at higher temperature (0.5).
    The two answers are then compared for factual contradictions. Disagreement
    between two independent draws from the same model is a genuine signal that
    the underlying claim is uncertain or hallucinated.
"""

import json
import re

from mistralai.client import Mistral

from app.core.config import get_settings
from app.core.pdf_parser import Chunk

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

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

_QUOTE_CHECK_PROMPT = """\
You are a fact-checker. For each sentence in the ANSWER, decide whether it \
introduces any NEW factual claims that are NOT supported by the CONTEXT.

A sentence is SUPPORTED if ANY of the following are true:
- It is a transition, heading, or structural phrase ("Based on the context:", "In summary:")
- It explains, elaborates, or paraphrases facts already present in the CONTEXT
- It draws a logical inference from facts already in the CONTEXT
- Its specific claims (numbers, names, dates, events) can be traced to the CONTEXT

A sentence is UNSUPPORTED only if it states a SPECIFIC FACT (number, name, date, \
statistic, event) that is NOT present in and CANNOT be inferred from the CONTEXT.

Citation markers like [Source: ...] are metadata — always mark as supported.

Return a JSON array — one object per sentence:
[
  {{"sentence": "...", "supported": true, "anchor": "brief passage from context, or null for elaborations"}},
  {{"sentence": "...", "supported": false, "missing_fact": "what specific fact is absent from the context"}}
]

Respond with ONLY the JSON array.

CONTEXT:
{context}

ANSWER:
{answer}
"""

_CONSISTENCY_PROMPT = """\
You are comparing two independent answers to the same question to find factual contradictions.

QUESTION: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

List any sentences where Answer A and Answer B make contradictory factual claims.
Return a JSON array of plain-English contradiction descriptions.
If there are no contradictions, return an empty array [].

Example: ["Answer A states revenue grew 12%% while Answer B states it declined 3%%"]

Respond with ONLY the JSON array.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks_with_scores: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in chunks_with_scores:
        parts.append(f"[{chunk.source}, p.{chunk.page}] (score: {score:.3f})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    """
    Extract checkable factual sentences from an answer.
    Strips citation markers and markdown formatting before splitting.
    """
    clean = re.sub(r"\[Source:[^\]]+\]", "", text)       # remove citation markers
    clean = re.sub(r"\*+([^*]+)\*+", r"\1", clean)       # remove bold/italic
    clean = re.sub(r"#+\s.*", "", clean)                  # remove headings
    clean = re.sub(r"-{3,}", "", clean)                   # remove horizontal rules
    parts = re.split(r"(?<=[.!?])\s+", clean.strip())
    return [s.strip() for s in parts if len(s.strip()) > 30]


def _kb_answer(
    client: Mistral, question: str, context: str, model: str, temperature: float = 0.1
) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": _KB_SYSTEM},
            {"role": "user", "content": _KB_USER_TEMPLATE.format(context=context, question=question)},
        ],
        max_tokens=1024,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _conversational(client: Mistral, question: str, model: str) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": _CONVERSATIONAL_TEMPLATE.format(question=question)}],
        max_tokens=256,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _parse_json_array(raw: str) -> list:
    """Robustly extract the first JSON array from a model response."""
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        result = json.loads(match.group())
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# Layer 1 + 2: Quote-grounded check with rejection threshold
# ---------------------------------------------------------------------------

def _quote_grounded_check(
    client: Mistral, context: str, answer: str, model: str, rejection_threshold: float
) -> dict:
    """
    Check whether any sentence in the answer introduces a specific fact
    (number, name, date, statistic) not present in or inferable from the context.

    Sentences that already carry a [Source: ...] citation are considered
    pre-supported and skipped — the model already grounded them during generation.
    Only uncited sentences are passed to the checker LLM.

    The rejection threshold applies to the ratio of flagged sentences among
    all checkable sentences. A 90% explanation / 10% quote answer will not
    be rejected unless the explanatory sentences make up new facts.
    """
    # Partition sentences: cited ones are pre-supported, uncited ones need checking.
    raw_parts = re.split(r'(?<=[.!?])\s+', answer.strip())
    uncited_parts = [s for s in raw_parts if '[Source:' not in s]
    n_cited = len(raw_parts) - len(uncited_parts)

    # Clean uncited sentences (strip formatting, drop trivially short ones)
    sentences = _split_sentences("\n".join(uncited_parts))

    # Total checkable = cited (auto-supported) + uncited sentences sent to checker
    total_sentences = n_cited + len(sentences)

    if not sentences:
        return {"flagged": [], "details": [], "answer_rejected": False, "rejection_ratio": 0.0}

    prompt = _QUOTE_CHECK_PROMPT.format(context=context, answer="\n".join(sentences))
    raw = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0,
    ).choices[0].message.content.strip()

    details = _parse_json_array(raw)

    # A sentence is flagged only if it explicitly introduces a new unsupported fact
    flagged = [
        f"{d.get('sentence', '')} — missing: {d.get('missing_fact', 'unspecified')}"
        for d in details
        if isinstance(d, dict) and not d.get("supported", True)
    ]

    rejection_ratio = len(flagged) / max(total_sentences, 1)
    # Require at least 2 flagged sentences before rejecting — prevents a single
    # borderline sentence from killing a mostly-correct answer on short outputs.
    answer_rejected = rejection_ratio > rejection_threshold and len(flagged) >= 2

    return {
        "flagged": flagged,
        "details": details,
        "answer_rejected": answer_rejected,
        "rejection_ratio": rejection_ratio,
    }


# ---------------------------------------------------------------------------
# Layer 3: Self-consistency check
# ---------------------------------------------------------------------------

def _self_consistency_check(
    client: Mistral, question: str, context: str, answer_a: str, model: str
) -> list[str]:
    """
    Generate a second independent answer (higher temperature) and compare it
    against the first for factual contradictions. Returns a list of
    contradiction descriptions found.
    """
    answer_b = _kb_answer(client, question, context, model, temperature=0.5)

    prompt = _CONSISTENCY_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    raw = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    ).choices[0].message.content.strip()

    return _parse_json_array(raw)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    intent: str,
    chunks_with_scores: list[tuple[Chunk, float]],
) -> dict:
    """
    Generate an answer and run the full three-layer hallucination pipeline.

    Returns:
      {
        "answer":               str,
        "answer_rejected":      bool,   # True → caller should return insufficient evidence
        "hallucination_warnings": list[str],  # unsupported sentences (layer 1)
        "consistency_warnings": list[str],    # contradictions found (layer 3)
      }
    """
    settings = get_settings()
    client = Mistral(api_key=settings.mistral_api_key)

    if intent == "CONVERSATIONAL" or not chunks_with_scores:
        answer = _conversational(client, question, settings.mistral_chat_model)
        return {
            "answer": answer,
            "answer_rejected": False,
            "hallucination_warnings": [],
            "consistency_warnings": [],
        }

    context = _build_context(chunks_with_scores)
    answer = _kb_answer(client, question, context, settings.mistral_chat_model)

    # Layer 1 + 2: quote-grounded check
    check = _quote_grounded_check(
        client, context, answer, settings.mistral_chat_model,
        rejection_threshold=settings.hallucination_rejection_threshold,
    )

    if check["answer_rejected"]:
        return {
            "answer": answer,
            "answer_rejected": True,
            "hallucination_warnings": check["flagged"],
            "consistency_warnings": [],
        }

    # Layer 3: self-consistency — only triggered when borderline (some flags but not rejected)
    consistency_warnings: list[str] = []
    if check["flagged"]:
        consistency_warnings = _self_consistency_check(
            client, question, context, answer, settings.mistral_chat_model
        )

    return {
        "answer": answer,
        "answer_rejected": False,
        "hallucination_warnings": check["flagged"],
        "consistency_warnings": consistency_warnings,
    }
