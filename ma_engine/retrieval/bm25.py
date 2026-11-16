"""Okapi BM25 sparse retrieval (no external dependency)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .tokenizer import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(t)) for t in texts]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_tokens], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        df: Counter[str] = Counter()
        for c in self.doc_tokens:
            df.update(c.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.doc_tokens)

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(len(self.doc_tokens))
        if not q_terms or self.avg_len == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        for term in set(q_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.doc_tokens], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
