"""Small HTML building blocks for the dashboard.

Every piece of document- or model-derived text is HTML-escaped before it is
placed inside markup, so contract text can never break or inject into the page.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from ..visualization.theme import STATUS
from .theme import OXBLOOD

SEAL_COLORS = {"A": "#1F5E3B", "B": "#23466F", "C": "#8A6420", "D": OXBLOOD}
SEVERITY_LABEL = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}


def display_title(title: str, text: str = "") -> str:
    """Readable title: a known name, else the parties on the cover page, else the tidied file name."""
    from ..ingestion.metadata import describe
    from .library import FRIENDLY_NAMES

    if title in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[title]
    if text:
        who = describe(text, "")
        if who:
            return who
    words = title.replace("_", " ").replace("-", " ").split()
    return " ".join(w if w.isupper() or any(c.isdigit() for c in w) else w.capitalize() for w in words) or "Untitled agreement"


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def masthead(right_note: str = "") -> None:
    note = f'<div class="fp-note">{escape(right_note)}</div>' if right_note else ""
    html(f"""<div class="fp-masthead">
      <div class="fp-brand">Fine Print<small>M&amp;A agreement review and deal simulation</small></div>{note}
    </div>""")


def section_title(text: str) -> None:
    html(f'<div class="fp-section-title">{escape(text)}</div>')


def severity(sev: str) -> str:
    color = STATUS.get(sev, "#999")
    return f'<span class="fp-sev"><i style="background:{color}"></i>{SEVERITY_LABEL.get(sev, escape(sev))}</span>'


def tag(text: str) -> str:
    return f'<span class="fp-tag">{escape(text)}</span>'


def quote(text: str) -> None:
    if text:
        html(f'<div class="fp-quote">{escape(text)}</div>')


def note(text: str) -> None:
    html(f'<div class="fp-note">{escape(text)}</div>')


def verdict(score: float, rating: str, verdict_text: str, headline: str = "") -> None:
    grade = rating[:1] if rating else "D"
    color = SEAL_COLORS.get(grade, OXBLOOD)
    sub = f"<p style='margin-bottom:.5rem;color:var(--ink)'>{escape(headline)}</p>" if headline else ""
    html(f"""<div class="fp-verdict" style="--seal:{color}">
      <div class="fp-seal" role="img" aria-label="Deal score {score:.0f} out of 100">
        <b>{score:.0f}</b><span>of 100</span></div>
      <div><h2>{escape(rating)}</h2>{sub}<p>{escape(verdict_text)}</p></div>
    </div>""")


def figures(items: list[tuple[str, str]]) -> None:
    cells = "".join(f"<div><b>{escape(v)}</b><span>{escape(k)}</span></div>" for k, v in items)
    html(f'<div class="fp-figures">{cells}</div>')
