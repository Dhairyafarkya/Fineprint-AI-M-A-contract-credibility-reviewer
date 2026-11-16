"""Lightweight tokenizer for lexical (BM25) search over legal text."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or such that the
    their there these this to was were will with any all shall may which who whom other than
    upon under each been being hereof herein hereto thereof therein hereunder""".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping: 'terminations' -> 'termination', 'liabilities' -> 'liability'."""
    if len(token) <= 4 or token[0].isdigit():
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
