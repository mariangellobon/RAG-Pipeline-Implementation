"""
BM25 keyword retrieval — implemented from scratch (no rank_bm25 or similar).

BM25 formula:
  score(q, d) = Σ_t  IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))

Where:
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   [Robertson IDF]
  k1 = 1.5   (term saturation)
  b  = 0.75  (length normalisation)
"""

import math
import re
from collections import Counter

from app.core.pdf_parser import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._doc_freqs: list[Counter] = []     # per-doc term counts
        self._df: Counter = Counter()            # document frequency per term
        self._avgdl: float = 0.0
        self._N: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._doc_freqs = []
        self._df = Counter()
        total_len = 0

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            self._doc_freqs.append(tf)
            for term in tf:
                self._df[term] += 1
            total_len += len(tokens)

        self._N = len(chunks)
        self._avgdl = total_len / self._N if self._N else 1.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self._N == 0:
            return []

        terms = _tokenize(query)
        scores = [0.0] * self._N

        for term in terms:
            if term not in self._df:
                continue
            idf = math.log(
                (self._N - self._df[term] + 0.5) / (self._df[term] + 0.5) + 1
            )
            for i, tf_counter in enumerate(self._doc_freqs):
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                dl = sum(tf_counter.values())
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        # top-k by score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append((self._chunks[idx], score))
        return results
