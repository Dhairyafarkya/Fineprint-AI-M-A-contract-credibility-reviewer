"""Data models for documents, chunks and retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A cleaned source document (one merger agreement, LOI, filing...)."""

    doc_id: str
    title: str
    source_path: str = ""
    clean_text: str
    word_count: int
    sections: list[str] = Field(default_factory=list, description="Detected section headings")


class Chunk(BaseModel):
    """An overlapping window of words taken from a Document."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    word_start: int
    word_end: int
    section: str = ""

    @property
    def word_count(self) -> int:
        return self.word_end - self.word_start

    def citation_label(self) -> str:
        sec = f" | {self.section}" if self.section else ""
        return f"[{self.chunk_id}{sec}]"


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its fused relevance score."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
