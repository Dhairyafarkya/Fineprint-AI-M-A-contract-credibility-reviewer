"""Stitch pre-split document pieces back into whole documents.

Datasets often ship one long agreement as many small, overlapping files, e.g.
``contract_0.txt_1.txt, contract_0.txt_2.txt, ...``. Analysing each piece on its
own would treat fragments as separate deals, so pieces that share a base name
are ordered by their number and joined, removing the overlapping text between
consecutive pieces. The joined document is then cleaned and re-chunked normally.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# "<base>.txt_<n>.txt", "<base>_part_<n>.txt", "<base>_chunk<n>.txt", "<base>-piece-<n>.txt"
PIECE_PATTERNS = (
    re.compile(r"^(?P<base>.+?\.(?:txt|text))_(?P<n>\d+)\.(?:txt|text)$", re.I),
    re.compile(r"^(?P<base>.+?)[_\-.](?:part|chunk|piece|split)[_\-.]?(?P<n>\d+)\.(?:txt|text)$", re.I),
)
MIN_OVERLAP_PROBE = 20


def piece_key(path: Path) -> tuple[str, int] | None:
    """(base name, piece number) if the filename looks like one piece of a split document."""
    for pat in PIECE_PATTERNS:
        m = pat.match(path.name)
        if m:
            base = re.sub(r"\.(?:txt|text)$", "", m.group("base"), flags=re.I)
            return base, int(m.group("n"))
    return None


def find_overlap(a: str, b: str, search_window: int = 20_000) -> int:
    """Length of the longest suffix of `a` that is also a prefix of `b` (0 if none).

    Overlaps shorter than MIN_OVERLAP_PROBE characters are ignored as coincidental.
    """
    probe = b[:MIN_OVERLAP_PROBE]
    if len(probe) < MIN_OVERLAP_PROBE:
        return 0
    start = max(0, len(a) - search_window)
    i = a.find(probe, start)
    while i != -1:
        if b.startswith(a[i:]):
            return len(a) - i
        i = a.find(probe, i + 1)
    return 0


def join_pieces(texts: list[str]) -> str:
    if not texts:
        return ""
    out = texts[0]
    for nxt in texts[1:]:
        k = find_overlap(out, nxt)
        out += nxt[k:] if k else "\n" + nxt
    return out


def group_pieces(paths: list[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    """Split paths into {base: ordered pieces} (2+ pieces) and standalone files."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    singles: list[Path] = []
    for p in paths:
        key = piece_key(p)
        if key is None:
            singles.append(p)
        else:
            groups[str(p.parent / key[0])].append((key[1], p))
    multi: dict[str, list[Path]] = {}
    for base, items in groups.items():
        if len(items) >= 2:
            multi[base] = [p for _, p in sorted(items)]
        else:
            singles.extend(p for _, p in items)
    return multi, sorted(singles)
