"""Validated chart palette and shared Plotly layout.

Categorical slots are assigned in fixed order (never cycled); status colours are
reserved for severity and always shown with a text label.
"""

from __future__ import annotations

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_LIGHT = "#cde2fb"  # sequential blue step 100 (bands)
SEQ_MID = "#86b6ef"  # step 250

STATUS = {"low": "#0ca30c", "medium": "#fab219", "high": "#ec835a", "critical": "#d03b3b"}

TEXT_PRIMARY = "#172033"
TEXT_SECONDARY = "#4F5B6E"
GRID = "#E8EBEF"
REFERENCE = "#7B2D3A"
SURFACE = "#FFFFFF"


SERIF_FONT = "'Source Serif 4', Georgia, serif"
SANS_FONT = "'IBM Plex Sans', -apple-system, Helvetica, Arial, sans-serif"


def base_layout(title: str, y_title: str = "", height: int = 380) -> dict:
    return {
        "title": {"text": title, "font": {"size": 17, "color": TEXT_PRIMARY, "family": SERIF_FONT}, "x": 0.01, "xanchor": "left"},
        "height": height,
        "margin": {"l": 16, "r": 16, "t": 64, "b": 16},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"color": TEXT_SECONDARY, "size": 12, "family": SANS_FONT},
        "hoverlabel": {"font": {"family": SANS_FONT}, "bgcolor": "#FFFFFF", "bordercolor": "#DCE0E6"},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        "xaxis": {"gridcolor": GRID, "zeroline": False, "showline": False},
        "yaxis": {"gridcolor": GRID, "zeroline": False, "title": {"text": y_title}, "automargin": True},
    }
