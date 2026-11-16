"""Hybrid retriever: dense cosine + BM25, fused with Reciprocal Rank Fusion.

Legal questions mix exact terms ("Section 8.3", "Termination Fee") with
paraphrase ("what does the buyer owe if the deal dies"). BM25 nails the first,
embeddings the second; RRF combines both rankings without score calibration.
"""

from __future__ import annotations

import html

import numpy as np

from ..config import RetrievalConfig
from ..models import Chunk, RetrievedChunk
from .index_store import IndexStore


class HybridRetriever:
    def __init__(self, index: IndexStore, cfg: RetrievalConfig | None = None):
        self.index = index
        self.cfg = cfg or RetrievalConfig()

    @staticmethod
    def _ranks(scores: np.ndarray, mask: np.ndarray, depth: int) -> dict[int, int]:
        """Map chunk position -> 1-based rank among masked chunks with positive score."""
        idx = np.where(mask & (scores > 0))[0]
        order = idx[np.argsort(-scores[idx], kind="stable")][:depth]
        return {int(p): r + 1 for r, p in enumerate(order)}

    def _fused(self, query: str, mask: np.ndarray, depth: int) -> dict[int, tuple[float, int | None, int | None]]:
        dense_scores = self.index.dense @ self.index.embedder.embed_query(query)
        sparse_scores = self.index.bm25.scores(query)
        d_ranks = self._ranks(dense_scores, mask, depth)
        s_ranks = self._ranks(sparse_scores, mask, depth)
        k = self.cfg.rrf_k
        fused: dict[int, tuple[float, int | None, int | None]] = {}
        for pos in set(d_ranks) | set(s_ranks):
            dr, sr = d_ranks.get(pos), s_ranks.get(pos)
            score = (self.cfg.dense_weight / (k + dr) if dr else 0.0) + (self.cfg.sparse_weight / (k + sr) if sr else 0.0)
            fused[pos] = (score, dr, sr)
        return fused

    def search(self, query: str, k: int | None = None, doc_id: str | None = None) -> list[RetrievedChunk]:
        return self.search_many([query], k=k, doc_id=doc_id)

    def search_many(self, queries: list[str], k: int | None = None, doc_id: str | None = None) -> list[RetrievedChunk]:
        """Multi-query retrieval: fuse rankings from several phrasings of the same need."""
        k = k or self.cfg.top_k
        if len(self.index) == 0:
            return []
        mask = self.index.mask_for(doc_id)
        depth = max(50, k * 5)
        def best_rank(a: int | None, b: int | None) -> int | None:
            ranks = [r for r in (a, b) if r]
            return min(ranks) if ranks else None

        total: dict[int, list] = {}
        for q in queries:
            if not q.strip():
                continue
            for pos, (score, dr, sr) in self._fused(q, mask, depth).items():
                entry = total.setdefault(pos, [0.0, None, None])
                entry[0] += score
                entry[1] = best_rank(entry[1], dr)
                entry[2] = best_rank(entry[2], sr)
        best = sorted(total.items(), key=lambda kv: -kv[1][0])[:k]
        return [
            RetrievedChunk(chunk=self.index.chunks[pos], score=round(v[0], 6), dense_rank=v[1], sparse_rank=v[2])
            for pos, v in best
        ]

    def neighbors(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Adjacent chunks from the same document (useful when a clause spans a boundary)."""
        pos = self.index.position(chunk_id)
        if pos is None:
            return []
        doc = self.index.chunks[pos].doc_id
        out = []
        for p in range(pos - window, pos + window + 1):
            if 0 <= p < len(self.index) and self.index.chunks[p].doc_id == doc:
                out.append(self.index.chunks[p])
        return out


def format_context(results: list[RetrievedChunk] | list[Chunk], max_chars: int | None = None) -> str:
    """Render chunks as tagged excerpts for a prompt, ordered by position in the document."""
    chunks = [r.chunk if isinstance(r, RetrievedChunk) else r for r in results]
    chunks = sorted({c.chunk_id: c for c in chunks}.values(), key=lambda c: (c.doc_id, c.index))
    parts, used = [], 0
    for c in chunks:
        section = html.escape(c.section, quote=True)
        block = f'<excerpt id="{c.chunk_id}" section="{section}">\n{c.text}\n</excerpt>'
        if max_chars and used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
