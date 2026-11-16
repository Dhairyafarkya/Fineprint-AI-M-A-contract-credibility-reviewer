"""Agreements available without uploading: bundled samples and the data/raw dataset folder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ..config import PROJECT_ROOT
from ..ingestion.loader import RawDocument, load_folder
from ..ingestion.metadata import describe

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

FRIENDLY_NAMES = {
    "cisco_acacia_merger_agreement": "Cisco / Acacia Communications merger agreement (2021)",
    "sample_merger_agreement": "Halcyon / Bluefin illustrative agreement (risk clauses planted)",
}


@dataclass
class LibraryItem:
    label: str
    doc: RawDocument


def _folder_signature(folder: Path) -> tuple:
    if not folder.is_dir():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in folder.rglob("*.txt")))


@st.cache_data(show_spinner=False)
def _load(folder: str, _signature: tuple) -> list[RawDocument]:
    return load_folder(folder)


def library_items() -> list[LibraryItem]:
    items: list[LibraryItem] = []
    seen: set[str] = set()
    for folder in (SAMPLE_DIR, DATASET_DIR):
        for doc in _load(str(folder), _folder_signature(folder)):
            fingerprint = hashlib.sha256(" ".join(doc.text.split()).encode("utf-8", "ignore")).hexdigest()
            if fingerprint in seen or len(doc.text.split()) < 50:
                continue  # same agreement stored twice (e.g. split pieces and a joined copy)
            seen.add(fingerprint)
            stem = Path(doc.title).stem
            words = len(doc.text.split())
            if stem in FRIENDLY_NAMES:
                label = FRIENDLY_NAMES[stem]
            else:
                who = describe(doc.text, "")
                label = f"{stem.replace('_', ' ')}: {who}" if who else stem.replace("_", " ").strip().capitalize()
            items.append(LibraryItem(f"{label} ({words:,} words)", doc))
    return items
