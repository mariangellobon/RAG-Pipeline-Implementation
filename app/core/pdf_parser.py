"""
PDF text extraction and chunking.

Chunking considerations:
- Fixed-size token-approximate chunks (char-based) with overlap to preserve context
  across boundaries. Overlap of ~12% of chunk size ensures sentences split at a
  boundary still appear in a neighboring chunk, reducing retrieval misses.
- Pages are extracted individually so we can attach page numbers to citations.
- We strip excessive whitespace and skip pages with very little text (e.g. cover
  pages, blank pages) to avoid polluting the index with low-signal chunks.
- Chunk size of 512 chars (~100-130 tokens for English text) is a deliberate
  trade-off: small enough for precise retrieval, large enough to carry context.
  Users can tune CHUNK_SIZE and CHUNK_OVERLAP via env vars.
"""

import re
from dataclasses import dataclass, field
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
    """Normalise whitespace without collapsing intentional line breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_chunks(
    file: BinaryIO,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Extract text from a PDF and split it into overlapping chunks."""
    chunks: list[Chunk] = []
    global_index = 0

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean(raw)

            # Skip near-empty pages (covers, dividers, etc.)
            if len(text) < 80:
                continue

            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()

                if len(chunk_text) >= 40:   # ignore tiny trailing fragments
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            source=filename,
                            page=page_num,
                            chunk_index=global_index,
                            char_start=start,
                        )
                    )
                    global_index += 1

                # Advance by (chunk_size - overlap) so the next chunk
                # re-reads the last `overlap` chars of this one.
                step = chunk_size - chunk_overlap
                start += step

    return chunks
