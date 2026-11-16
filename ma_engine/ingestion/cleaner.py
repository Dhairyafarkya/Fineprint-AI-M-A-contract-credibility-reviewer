"""Strip messy formatting from raw legal text.

Handles the usual mess in agreements exported to .txt: SEC/EDGAR SGML headers,
stray HTML, page numbers, repeated running headers/footers, hyphenation at line
breaks, hard-wrapped lines, separator rules and odd unicode.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

_UNICODE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
    " ": " ", " ": " ", " ": " ", "​": "",
    "﻿": "", "­": "", "•": "-", "·": "-",
}

_SEC_HEADER = re.compile(r"<SEC-HEADER>.*?</SEC-HEADER>", re.S | re.I)
_SGML_TAG = re.compile(r"</?(?:DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION|TEXT|PAGE|PDF)[^>]*>", re.I)
_HTML_TAG = re.compile(r"<[^>\n]{1,400}>")
_PAGE_LINE = re.compile(
    r"^\s*(?:-\s*\d{1,4}\s*-|page\s+\d{1,4}(?:\s+of\s+\d{1,4})?|\d{1,4}|[ivxlc]{1,6}|\[\s*\d{1,4}\s*\])\s*$",
    re.I,
)
_SEPARATOR_LINE = re.compile(r"^\s*[-=_*~#.]{4,}\s*$")
_HYPHEN_BREAK = re.compile(r"(\S+)-\n(\w)")
HEADING_LINE = re.compile(
    r"^\s*(?:ARTICLE\s+[IVXLC\d]+\b|SECTION\s+\d+(?:\.\d+)*|Section\s+\d+(?:\.\d+)*|§\s*\d+|"
    r"EXHIBIT\s+[A-Z\d]+|SCHEDULE\s+[A-Z\d.]+|ANNEX\s+[A-Z\d]+|\d+(?:\.\d+)+\.?\s+[A-Z])"
)
_LIST_ITEM = re.compile(r"^\s*(?:\([a-zA-Z0-9]{1,4}\)|[a-z]\.|\d+\.|[ivx]+\.|-)\s+")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _UNICODE_MAP.items():
        text = text.replace(bad, good)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")


def strip_markup(text: str) -> str:
    text = _SEC_HEADER.sub(" ", text)
    text = _SGML_TAG.sub("\n", text)
    if re.search(r"<(?:p|div|br|td|tr|span|font|table)\b", text, re.I):
        text = re.sub(r"<(?:br|/p|/div|/tr|/h\d)[^>]*>", "\n", text, flags=re.I)
        text = _HTML_TAG.sub(" ", text)
    return html.unescape(text)


def remove_noise_lines(text: str, repeat_threshold: int = 4) -> str:
    """Drop page numbers, separator rules and running headers/footers."""
    lines = text.split("\n")
    stripped = [ln.strip() for ln in lines]
    counts = Counter(s for s in stripped if s)
    repeated = {
        s for s, c in counts.items()
        if c >= repeat_threshold and len(s) <= 80 and not HEADING_LINE.match(s)
        and not s.endswith((".", ";", ":"))
    }
    kept = []
    for raw, s in zip(lines, stripped):
        if s and (_PAGE_LINE.match(s) or _SEPARATOR_LINE.match(s) or s in repeated):
            continue
        kept.append(raw.rstrip())
    return "\n".join(kept)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if HEADING_LINE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(s) <= 80 and len(letters) >= 3 and all(c.isupper() for c in letters)


def _is_pure_heading(line: str) -> bool:
    """A heading with no sentence body after it (e.g. 'Section 8.3 Termination Fee')."""
    if not _is_heading(line):
        return False
    s = line.strip()
    return not (HEADING_LINE.match(s) and re.search(r"[.:]\s+\S", s))


def _join_hyphenated(m: re.Match) -> str:
    """Rejoin a word split across lines. Keep the hyphen for real compounds ('go-to-' + 'market')."""
    head, tail = m.group(1), m.group(2)
    keep = "-" in head or not head[-1].isalpha() or tail.isupper() or tail.isdigit()
    return f"{head}-{tail}" if keep else f"{head}{tail}"


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines back into paragraphs while keeping headings and list items."""
    text = _HYPHEN_BREAK.sub(_join_hyphenated, text)
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if out and out[-1]:
            prev = out[-1]
            can_join = (
                not _is_pure_heading(prev)
                and not _is_heading(s)
                and not _LIST_ITEM.match(s)
                and not prev.rstrip().endswith((".", ":", ";", "?", "!"))
            )
            if can_join:
                out[-1] = prev.rstrip() + " " + s
                continue
        out.append(s)
    return "\n".join(out)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw: str) -> str:
    """Full cleaning pipeline. Idempotent: clean_text(clean_text(x)) == clean_text(x)."""
    text = normalize_unicode(raw)
    text = strip_markup(text)
    text = remove_noise_lines(text)
    text = unwrap_lines(text)
    return collapse_whitespace(text)
