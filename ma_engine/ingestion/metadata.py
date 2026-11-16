"""Light metadata from an agreement's cover page, used to label documents in pickers.

Cover pages follow a few patterns ("AGREEMENT AND PLAN OF MERGER among A, B and C
dated as of ...", "... by and between A and B"). We take the party list after
"among"/"between", split it into names, and drop merger subsidiaries so the label
reads "Party / Party" in the order the agreement lists them.
"""

from __future__ import annotations

import re

_DOC_TYPES = (
    "AGREEMENT AND PLAN OF MERGER", "AGREEMENT AND PLAN OF REORGANIZATION", "MERGER AGREEMENT",
    "STOCK PURCHASE AGREEMENT", "ASSET PURCHASE AGREEMENT", "SHARE PURCHASE AGREEMENT",
    "BUSINESS COMBINATION AGREEMENT", "TRANSACTION AGREEMENT", "LETTER OF INTENT", "ARRANGEMENT AGREEMENT",
)
_INTRO = re.compile(r"\b(?:by and among|by and between|among|between)\s*:?\s+", re.I)
_MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
_END = re.compile(r"\s(?:dated|date[d]?\s+as\s+of|table\s+of\s+contents|article\s+[ivx\d]+\b|section\s+1|_{3,}|"
                  rf"this\s+agreement|execution\s+(?:version|copy)|page\b|{_MONTHS}\s+\d{{1,2}})", re.I)
_DATED = re.compile(r"\b(?:dated\s+)?as\s+of\s+[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4},?|\bdated\s+[A-Z][a-z]+\s+\d{1,2},?\s+\d{4},?", re.I)
_SUFFIX = r"(?:INC|LLC|L\.L\.C|L\.P|LP|LTD|CORP|CO|PLC|N\.V|S\.A|AG|SE|LIMITED|CORPORATION|HOLDINGS|COMPANY)\b\.?"
_SPLIT = re.compile(rf"\s+and\s+(?!{_SUFFIX})|;\s+|,\s+(?!{_SUFFIX}|a\s|an\s)", re.I)
_DESCRIPTOR = re.compile(r",?\s+an?\s+[^,;]{0,60}?(?:corporation|company|aktieselskab|partnership|trust|bank|"
                         r"association|aktiengesellschaft)(?:\s*\([^)]*\))?", re.I)
_SUB = re.compile(r"merger\s*sub|mergersub|\bsub\b|merger\s+co\b|\bmergerco\b|\bbidco\b|acquisition\s+sub", re.I)
_SMALL = {"and", "of", "the", "&", "de", "du"}
_KEEP_UPPER = {"LLC", "PLC", "LP", "L.P.", "N.V.", "S.A.", "AG", "SE", "II", "III", "CC", "US", "USA", "SVB",
               "WSFS", "CIT", "HC", "IRC", "BMC", "CAI", "STERIS"}


def _nice(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;:")
    if not name.isupper() and sum(c.isupper() for c in name) < 0.6 * sum(c.isalpha() for c in name):
        return name
    words = []
    for i, w in enumerate(name.split()):
        bare = w.strip(",.")
        if bare.upper() in _KEEP_UPPER:
            words.append(w.upper() if w.strip(",.").upper() == bare.upper() else w)
        elif w.lower() in _SMALL and i:
            words.append(w.lower())
        else:
            words.append("-".join("&".join(q.upper() if len(q) == 1 else q.capitalize() for q in p.split("&"))
                                   for p in w.split("-")))
    return " ".join(words)


def document_type(text: str) -> str:
    head = re.sub(r"\s+", " ", text[:3000]).upper()
    return next((t.title().replace("Of ", "of ").replace("And ", "and ") for t in _DOC_TYPES if t in head), "")


def parties(text: str, limit: int = 2) -> list[str]:
    """Principal parties on the cover page, in document order, without merger subsidiaries."""
    head = text[:2500].replace("\u200b", "").replace("\ufeff", "")
    head = re.sub(r"https?://\S+|\b\d+/\d+\b", " ", head)  # page-footer URLs and "2/116" page counters
    head = _DATED.sub(" ", re.sub(r"\s+", " ", head))
    intro = _INTRO.search(head)
    if not intro:
        return []
    segment = head[intro.end(): intro.end() + 700]
    end = _END.search(" " + segment)
    if end:
        segment = segment[: max(0, end.start() - 1)]
    segment = _DESCRIPTOR.sub("", segment)
    names: list[str] = []
    for raw in _SPLIT.split(segment):
        raw = re.sub(r"^\s*(?:and|,)\s+", "", raw, flags=re.I).strip(" ,;:\u200b")
        if raw.endswith(".") and not re.search(rf"{_SUFFIX}$", raw, re.I):
            raw = raw.rstrip(".")
        raw = re.sub(r"\s+\d+$", "", raw)  # footnote markers
        if re.match(r"^\W*(?:article|section|table)\b", raw, re.I):
            break
        if len(raw) < 3 or len(raw) > 90 or not re.search(r"[A-Za-z]{3}", raw) or re.fullmatch(_SUFFIX, raw, re.I):
            continue
        if not any(raw.lower() == n.lower() for n in names):
            names.append(raw)
    principal = [n for n in names if not _SUB.search(n)]
    # With more than two parties, a later one sharing its first word with an earlier one is usually
    # an acquisition vehicle or bank subsidiary ("Supernus Pharmaceuticals" + "Supernus Reef").
    if len(principal) > 2:
        kept, firsts = [], set()
        for n in principal:
            first = n.split()[0].lower()
            if first not in firsts:
                kept.append(n)
            firsts.add(first)
        principal = kept
    if len(principal) > 2:
        principal = [principal[0], principal[-1]]
    return [_nice(n) for n in principal[:limit]]


def describe(text: str, fallback: str) -> str:
    """'Merck Sharp & Dohme Corp. / Acceleron Pharma Inc.' or the fallback title."""
    found = parties(text)
    return " / ".join(found) if len(found) >= 2 else fallback
