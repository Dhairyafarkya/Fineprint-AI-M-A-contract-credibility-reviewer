"""Load raw .txt documents from disk (single files, folders, or uploaded bytes)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".text"}
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass
class RawDocument:
    doc_id: str
    title: str
    source_path: str
    text: str


def decode_bytes(data: bytes) -> str:
    """Decode bytes trying common encodings; latin-1 never fails, so this always returns."""
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")  # pragma: no cover - unreachable


def make_doc_id(title: str, text: str) -> str:
    """Stable id: slugified title + short content hash (same file -> same id)."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "doc"
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{slug}-{digest}"


def load_text(text: str, title: str, source_path: str = "") -> RawDocument:
    if not text or not text.strip():
        raise ValueError(f"Document '{title}' is empty.")
    return RawDocument(doc_id=make_doc_id(title, text), title=title, source_path=source_path, text=text)


def load_bytes(data: bytes, filename: str) -> RawDocument:
    """Load an uploaded file (e.g. from Streamlit's file_uploader)."""
    title = Path(filename).stem
    return load_text(decode_bytes(data), title=title, source_path=filename)


def load_file(path: str | Path) -> RawDocument:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Only .txt files are supported.")
    return load_text(decode_bytes(path.read_bytes()), title=path.stem, source_path=str(path))


def load_folder(folder: str | Path, recursive: bool = True, reassemble: bool = True) -> list[RawDocument]:
    """Load every supported file in a folder, skipping empty files.

    With `reassemble` (default), files that are numbered pieces of one document
    (e.g. contract_0.txt_1.txt ... contract_0.txt_410.txt) are stitched back into
    a single document, removing the overlap between consecutive pieces.
    """
    from .reassembler import group_pieces, join_pieces

    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    pattern = "**/*" if recursive else "*"
    paths = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    groups, singles = group_pieces(paths) if reassemble else ({}, sorted(paths))

    docs: list[RawDocument] = []
    for base, pieces in sorted(groups.items()):
        text = join_pieces([decode_bytes(p.read_bytes()) for p in pieces])
        title = Path(base).name
        if text.strip():
            docs.append(load_text(text, title=title, source_path=f"{base} ({len(pieces)} pieces)"))
    for path in singles:
        try:
            docs.append(load_file(path))
        except ValueError:
            continue  # empty file
    from .archives import natural_key

    return sorted(docs, key=lambda d: natural_key(d.title))


def load_uploads(files: list[tuple[str, bytes]]) -> list[RawDocument]:
    """Load uploaded (filename, bytes) pairs, stitching numbered pieces like load_folder.

    Zip archives should be expanded first with ingestion.archives.unpack.
    """
    from .archives import natural_key
    from .reassembler import group_pieces, join_pieces

    by_name = {name: data for name, data in files if data and data.strip()}
    groups, singles = group_pieces([Path(n) for n in by_name])
    docs: list[RawDocument] = []
    for base, pieces in sorted(groups.items()):
        text = join_pieces([decode_bytes(by_name[str(p)]) for p in pieces])
        docs.append(load_text(text, title=Path(base).name, source_path=f"{Path(base).name} ({len(pieces)} pieces)"))
    for p in singles:
        try:
            docs.append(load_bytes(by_name[str(p)], str(p)))
        except ValueError:
            continue  # blank file
    return sorted(docs, key=lambda d: natural_key(d.title))
