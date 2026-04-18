"""
PDF text extraction and chunking.

Chunking strategy — paragraph-aware with sentence-boundary fallback:
- Pages are split on paragraph breaks (double newlines) first, which preserves
  the document's natural structure (headings, bullet groups, body paragraphs).
- Paragraphs that exceed chunk_size are further split at sentence boundaries
  (. ? !) so chunks never cut mid-sentence.
- Adjacent paragraphs are merged until adding the next one would exceed
  chunk_size, keeping related ideas together without hard character cuts.
- Pages with very little text (covers, dividers) are skipped to avoid
  polluting the index with low-signal chunks.
- Chunk size of 512 chars (~100-130 tokens) is tunable via CHUNK_SIZE env var.
"""

import re
from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber


@dataclass
class Chunk:
    text: str
    source: str          # filename
    page: int
    chunk_index: int     # global index across all chunks for this source
    char_start: int      # position in original page text


def _clean(text: str) -> str:
    """Normalise whitespace without collapsing intentional paragraph breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ? ! boundaries."""
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunks_from_paragraphs(text: str, chunk_size: int) -> list[str]:
    """
    Split page text into chunks that respect paragraph and sentence boundaries.

    Strategy:
    1. Split on double-newline paragraph breaks.
    2. Merge adjacent short paragraphs until adding the next would exceed chunk_size.
    3. Paragraphs that are themselves longer than chunk_size are split at sentence
       boundaries using the same greedy merge logic.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    # Break oversized paragraphs at sentence boundaries
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sentences = _split_sentences(para)
            current = ""
            for sent in sentences:
                if not current:
                    current = sent
                elif len(current) + 1 + len(sent) <= chunk_size:
                    current += " " + sent
                else:
                    units.append(current)
                    current = sent
            if current:
                units.append(current)

    # Merge adjacent units greedily up to chunk_size
    result: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= chunk_size:
            current += "\n\n" + unit
        else:
            result.append(current)
            current = unit
    if current:
        result.append(current)

    return result


def extract_chunks(
    file: BinaryIO,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,  # kept for API compatibility, not used in paragraph mode
) -> list[Chunk]:
    """Extract text from a PDF and split into paragraph-aware chunks."""
    chunks: list[Chunk] = []
    global_index = 0

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean(raw)

            # Skip near-empty pages (covers, dividers, etc.)
            if len(text) < 80:
                continue

            for chunk_text in _chunks_from_paragraphs(text, chunk_size):
                if len(chunk_text) >= 40:   # ignore tiny fragments
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            source=filename,
                            page=page_num,
                            chunk_index=global_index,
                            char_start=text.find(chunk_text),
                        )
                    )
                    global_index += 1

    return chunks
