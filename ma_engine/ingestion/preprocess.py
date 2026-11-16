"""Glue: raw document -> cleaned Document -> chunks."""

from __future__ import annotations

from ..config import ChunkingConfig
from ..models import Chunk, Document
from .chunker import chunk_document
from .cleaner import clean_text
from .loader import RawDocument
from .sections import find_headings, heading_labels


def preprocess(raw: RawDocument, cfg: ChunkingConfig | None = None) -> tuple[Document, list[Chunk]]:
    text = clean_text(raw.text)
    if not text:
        raise ValueError(f"Document '{raw.title}' has no text after cleaning.")
    doc = Document(
        doc_id=raw.doc_id,
        title=raw.title,
        source_path=raw.source_path,
        clean_text=text,
        word_count=len(text.split()),
        sections=heading_labels(find_headings(text)),
    )
    return doc, chunk_document(doc.doc_id, text, cfg)
