"""Ask-the-agreement: retrieval-augmented question answering with citations."""

from __future__ import annotations

from ..llm.base import LLMClient
from ..llm.prompts import SYSTEM_ANALYST, qa_prompt
from ..models import QAAnswer, RetrievedChunk
from ..retrieval.retriever import HybridRetriever, format_context


def answer_question(
    llm: LLMClient, retriever: HybridRetriever, question: str, doc_id: str | None = None, k: int = 8,
    max_chars: int = 0,
) -> tuple[QAAnswer, list[RetrievedChunk]]:
    question = question.strip()
    if not question:
        raise ValueError("Question is empty.")
    hits = retriever.search(question, k=k, doc_id=doc_id)
    # Pull in neighbours of the top hit so clauses split across chunk boundaries stay whole.
    chunks = [h.chunk for h in hits]
    if hits:
        chunks += retriever.neighbors(hits[0].chunk.chunk_id)
    answer = llm.structured(
        system=SYSTEM_ANALYST, prompt=qa_prompt(question), context=format_context(chunks, max_chars=max_chars or None),
        output_model=QAAnswer, max_tokens=3000, task="qa",
    )
    return answer, hits
