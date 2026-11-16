"""Build an IndexStore from raw documents (clean -> chunk -> embed -> index)."""

from __future__ import annotations

import logging

from ..config import Settings, get_settings
from ..ingestion.loader import RawDocument
from ..ingestion.preprocess import preprocess
from ..models import Chunk, Document
from .embedder import Embedder, create_embedder
from .index_store import IndexStore

log = logging.getLogger(__name__)


def build_index(
    raw_docs: list[RawDocument],
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> IndexStore:
    settings = settings or get_settings()
    documents: list[Document] = []
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for raw in raw_docs:
        if raw.doc_id in seen:
            log.info("Skipping duplicate document %s", raw.title)
            continue
        seen.add(raw.doc_id)
        doc, doc_chunks = preprocess(raw, settings.chunking)
        documents.append(doc)
        chunks.extend(doc_chunks)
        log.info("Prepared %s: %d words, %d chunks", doc.title, doc.word_count, len(doc_chunks))
    if not chunks:
        raise ValueError("No text to index.")

    if embedder is None:
        local = settings.llm.is_local
        embedder = create_embedder(
            settings.retrieval.embedding_backend, settings.retrieval.embedding_model,
            ollama_host=settings.llm.ollama_host if local else None,
            ollama_model=settings.llm.ollama_embed_model if local else None,
        )
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    dense = embedder.embed_documents(texts)
    log.info("Embedded %d chunks with %s", len(chunks), embedder.name)
    return IndexStore(documents, chunks, dense, embedder)
