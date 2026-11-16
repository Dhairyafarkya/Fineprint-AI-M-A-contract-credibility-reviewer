"""Detect article/section headings so every chunk knows where it sits in the agreement."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLE = re.compile(r"^(ARTICLE\s+[IVXLC\d]+)\b[.:\s-]*(.*)$", re.I)
_SECTION = re.compile(r"^((?:Section|SECTION|§)\s*\d+(?:\.\d+)*)\.?\s+([^.\n]{0,90})")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+([A-Z][^.\n]{0,90})")
_CAPS_TITLE = re.compile(r"^[A-Z][A-Z ,;&'/()-]{2,80}$")
# Table-of-contents entries end with a page number ("1.3. Closing 10", "Article VIII GENERAL PROVISIONS 74 iii").
_TOC_ENTRY = re.compile(r"\s(?:\d{1,3}|[ivxlc]{1,6})(?:\s+[ivxlc]{1,6})?\s*$")


@dataclass
class Heading:
    char_pos: int
    level: int  # 1 = article, 2 = section
    label: str


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" .:-")[:80]


def find_headings(text: str) -> list[Heading]:
    """Return headings in document order with their character offsets."""
    headings: list[Heading] = []
    pos = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not _TOC_ENTRY.search(s):
            m = _ARTICLE.match(s)
            if m and len(s) <= 120:
                title = _clean_title(m.group(2))
                if not title:
                    nxt = next((ln.strip() for ln in lines[i + 1 : i + 4] if ln.strip()), "")
                    if _CAPS_TITLE.match(nxt) and not _ARTICLE.match(nxt):
                        title = _clean_title(nxt)
                label = m.group(1).upper() + (f" - {title.title()}" if title else "")
                headings.append(Heading(pos, 1, label))
            else:
                m = _SECTION.match(s) or _NUMBERED.match(s)
                if m:
                    num = m.group(1)
                    if not num.lower().startswith(("section", "§")):
                        num = f"Section {num}"
                    headings.append(Heading(pos, 2, f"{num} {_clean_title(m.group(2))}".strip()))
        pos += len(line) + 1
    return headings


def section_at(headings: list[Heading], char_pos: int) -> str:
    """Breadcrumb ('ARTICLE VIII - Termination > Section 8.3 Termination Fee') at a position."""
    article = section = ""
    for h in headings:
        if h.char_pos > char_pos:
            break
        if h.level == 1:
            article, section = h.label, ""
        else:
            section = h.label
    return " > ".join(p for p in (article, section) if p)


def heading_labels(headings: list[Heading]) -> list[str]:
    """Unique heading labels in document order."""
    return list(dict.fromkeys(h.label for h in headings))
