"""In-memory vector + BM25 index with save/load to disk.

On-disk layout of an index directory::

    meta.json        documents (without full text duplication) and settings
    documents.jsonl  one cleaned Document per line
    chunks.jsonl     one Chunk per line
    dense.npy        float32 matrix, one L2-normalised row per chunk
    embedder.pkl     embedder state (vectorizer for TF-IDF, model name for ST)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ..models import Chunk, Document
from .bm25 import BM25Index
from .embedder import Embedder, embedder_from_state

INDEX_VERSION = 2  # v2: split documents are stitched back together before chunking


class IndexStore:
    def __init__(self, documents: list[Document], chunks: list[Chunk], dense: np.ndarray, embedder: Embedder):
        if len(chunks) != dense.shape[0]:
            raise ValueError("Number of chunks and embedding rows differ.")
        self.documents = {d.doc_id: d for d in documents}
        self.chunks = chunks
        self.dense = dense.astype(np.float32)
        self.embedder = embedder
        self.bm25 = BM25Index([c.text for c in chunks])
        self._pos = {c.chunk_id: i for i, c in enumerate(chunks)}

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        i = self._pos.get(chunk_id)
        return None if i is None else self.chunks[i]

    def position(self, chunk_id: str) -> int | None:
        return self._pos.get(chunk_id)

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def mask_for(self, doc_id: str | None) -> np.ndarray:
        if doc_id is None:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c.doc_id == doc_id for c in self.chunks], dtype=bool)

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "documents.jsonl", "w", encoding="utf-8") as f:
            for doc in self.documents.values():
                f.write(doc.model_dump_json() + "\n")
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        np.save(d / "dense.npy", self.dense)
        with open(d / "embedder.pkl", "wb") as f:
            pickle.dump(self.embedder.to_state(), f)
        meta = {
            "version": INDEX_VERSION,
            "embedder": self.embedder.name,
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "doc_ids": list(self.documents),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> IndexStore:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("version") != INDEX_VERSION:
            raise ValueError(f"Index at {d} has version {meta.get('version')}; rebuild it.")
        docs = [Document.model_validate_json(line) for line in (d / "documents.jsonl").read_text("utf-8").splitlines() if line]
        chunks = [Chunk.model_validate_json(line) for line in (d / "chunks.jsonl").read_text("utf-8").splitlines() if line]
        dense = np.load(d / "dense.npy")
        with open(d / "embedder.pkl", "rb") as f:
            embedder = embedder_from_state(pickle.load(f))  # noqa: S301 - own index files
        return cls(docs, chunks, dense, embedder)

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").is_file()
