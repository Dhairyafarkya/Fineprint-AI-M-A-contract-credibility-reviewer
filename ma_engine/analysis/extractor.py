"""Extract structured deal terms with multi-query RAG, then sanity-check them."""

from __future__ import annotations

from ..llm.base import LLMClient
from ..llm.prompts import EXTRACTION_PROMPT, SYSTEM_ANALYST
from ..models import Chunk, DealTerms
from ..retrieval.retriever import HybridRetriever, format_context

EXTRACTION_QUERIES: tuple[str, ...] = (
    "merger consideration per share cash stock aggregate equity value purchase price",
    "termination fee payable by the company",
    "parent termination fee reverse termination fee",
    "escrow amount release indemnification cap deductible survival",
    "earnout payment adjusted revenue period",
    "indebtedness credit agreement outstanding principal change of control prepayment",
    "change of control payments severance executives",
    "debt financing commitment term loan interest rate amortization",
    "synergies cost revenue run-rate integration costs",
    "closing expected timing outside date extension",
    "regulatory approvals HSR antitrust efforts divestiture",
    "financial statements revenue operating expenses cash",
    "acquirer revenue operating expenses cash on hand",
    "transaction expenses fees",
    "material adverse effect definition carve-outs disproportionately",
    "no solicitation superior proposal fiduciary out",
)

RATE_FIELDS = ("acquisition_debt_interest_rate", "indemnification_cap_pct", "target_debt_prepayment_premium")


def select_extraction_chunks(retriever: HybridRetriever, doc_id: str, max_chunks: int = 24,
                             full_text_max_words: int = 0) -> list[Chunk]:
    """The whole document when it fits (`full_text_max_words`), else multi-query RAG retrieval."""
    all_chunks = retriever.index.chunks_for(doc_id)
    words = sum(c.word_count for c in all_chunks)
    if len(all_chunks) <= max_chunks or (full_text_max_words and words <= full_text_max_words):
        return all_chunks
    picked: dict[str, Chunk] = {all_chunks[0].chunk_id: all_chunks[0]}  # cover page: parties, date
    results = [retriever.search(q, k=3, doc_id=doc_id) for q in EXTRACTION_QUERIES]
    for rank in range(3):  # every query gets its best passage before any query gets a second
        for hits in results:
            if rank < len(hits) and len(picked) < max_chunks:
                picked.setdefault(hits[rank].chunk.chunk_id, hits[rank].chunk)
    return sorted(picked.values(), key=lambda c: c.index)


def sanitize_terms(t: DealTerms) -> DealTerms:
    """Fix common unit mistakes (percent given as 7.5 instead of 0.075, negatives)."""
    data = t.model_dump()
    for k, v in data.items():
        if isinstance(v, float) and v < 0 and k.endswith("_usd"):
            data[k] = abs(v)
    for k in RATE_FIELDS:
        v = data.get(k)
        if isinstance(v, (int, float)) and v > 1:
            data[k] = v / 100
    return DealTerms.model_validate(data)


def extract_deal_terms(llm: LLMClient, retriever: HybridRetriever, doc_id: str,
                       full_text_max_words: int = 0, max_chunks: int = 24) -> DealTerms:
    chunks = select_extraction_chunks(retriever, doc_id, max_chunks=max_chunks, full_text_max_words=full_text_max_words)
    terms = llm.structured(
        system=SYSTEM_ANALYST,
        prompt=EXTRACTION_PROMPT,
        context=format_context(chunks),
        output_model=DealTerms,
        max_tokens=6000,
        task="extract-terms",
    )
    return sanitize_terms(terms)
