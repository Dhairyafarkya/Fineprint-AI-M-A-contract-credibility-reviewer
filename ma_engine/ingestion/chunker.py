"""Clause-aware overlapping chunker.

Splits a cleaned document into ~500-word windows with ~100 words of overlap.
Chunk boundaries are nudged to the nearest paragraph break (or, failing that,
sentence end) so clauses are not cut mid-sentence. The original text slice is
kept verbatim, including line breaks, so quotes taken from chunks are exact.
"""

from __future__ import annotations

import re
from bisect import bisect_left

from ..config import ChunkingConfig
from ..models import Chunk
from .sections import find_headings, section_at

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.;:?!][\"')\]]*$")


def _nearest(sorted_points: list[int], target: int, lo: int, hi: int) -> int | None:
    """Closest value to target inside [lo, hi] from a sorted list, or None."""
    if lo > hi:
        return None
    i = bisect_left(sorted_points, target)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sorted_points) and lo <= sorted_points[j] <= hi:
            if best is None or abs(sorted_points[j] - target) < abs(best - target):
                best = sorted_points[j]
    if best is None:  # fall back to a scan of the window (rare)
        inside = [p for p in sorted_points if lo <= p <= hi]
        if inside:
            best = min(inside, key=lambda p: abs(p - target))
    return best


def chunk_document(doc_id: str, text: str, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    if cfg.overlap_words >= cfg.chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    n = len(spans)
    if n == 0:
        return []

    # Word indices where a new paragraph / sentence starts (valid cut points).
    para_breaks: list[int] = []
    sent_breaks: list[int] = []
    for i in range(1, n):
        gap = text[spans[i - 1][1]:spans[i][0]]
        if "\n" in gap:
            para_breaks.append(i)
        prev_word = text[spans[i - 1][0]:spans[i - 1][1]]
        if _SENTENCE_END.search(prev_word):
            sent_breaks.append(i)

    tol = max(20, cfg.chunk_words // 6)
    headings = find_headings(text)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        target = start + cfg.chunk_words
        if target >= n:
            end = n
        else:
            lo, hi = max(start + cfg.chunk_words // 2, target - tol), min(n - 1, target + tol // 2)
            end = (
                _nearest(para_breaks, target, lo, hi)
                or _nearest(sent_breaks, target, lo, hi)
                or target
            )
        bounds.append((start, end))
        if end >= n:
            break
        nxt = end - cfg.overlap_words
        snapped = _nearest(sent_breaks, nxt, max(start + 1, nxt - tol // 2), min(end - 1, nxt + tol // 2))
        start = snapped if snapped is not None else max(start + 1, nxt)

    # Merge a tiny tail into the previous chunk.
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < cfg.min_chunk_words:
        last_end = bounds.pop()[1]
        bounds[-1] = (bounds[-1][0], last_end)

    chunks: list[Chunk] = []
    for idx, (ws, we) in enumerate(bounds):
        c_start, c_end = spans[ws][0], spans[we - 1][1]
        # Section in force at the chunk start (a heading opening the chunk counts).
        probe = min(c_end, c_start + 40)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-c{idx:04d}",
                doc_id=doc_id,
                index=idx,
                text=text[c_start:c_end],
                word_start=ws,
                word_end=we,
                section=section_at(headings, probe),
            )
        )
    return chunks
