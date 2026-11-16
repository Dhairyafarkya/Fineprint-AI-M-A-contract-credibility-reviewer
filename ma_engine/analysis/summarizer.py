"""Dynamic executive summary that scales with the document.

* Short documents  -> punchy summary.
* Medium documents -> concise multi-paragraph summary.
* Long documents   -> comprehensive overview with subheadings.

Documents that fit in one request are summarised from the full chunked text.
Bigger ones use map-reduce: groups of chunks are summarised in parallel, then
the section summaries are combined into the final executive summary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import SummaryConfig
from ..llm.base import LLMClient
from ..llm.prompts import SYSTEM_ANALYST, summary_map_prompt, summary_reduce_prompt
from ..models import Chunk, Document, ExecutiveSummary, SectionSummary
from ..retrieval.retriever import HybridRetriever, format_context

# Topics an executive summary must cover; used to retrieve passages when the whole
# agreement does not fit in the model's context window (local models).
SUMMARY_QUERIES: tuple[str, ...] = (
    "parties merger agreement recitals board approval",
    "merger consideration per share cash stock equity value",
    "conditions to closing regulatory approvals stockholder approval",
    "termination fee reverse termination fee outside date",
    "financing commitment debt",
    "material adverse effect definition",
    "no solicitation superior proposal",
    "closing date effective time timeline",
    "indemnification escrow earnout",
)


def length_tier(word_count: int, cfg: SummaryConfig) -> str:
    if word_count <= cfg.short_doc_words:
        return "short"
    if word_count <= cfg.long_doc_words:
        return "medium"
    return "long"


def select_summary_chunks(retriever: HybridRetriever, doc_id: str, chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """Cover page plus the best passages for each summary topic, in document order, within max_chars."""
    picked: dict[str, Chunk] = {chunks[0].chunk_id: chunks[0]} if chunks else {}
    used = len(chunks[0].text) if chunks else 0
    for rank in range(3):  # round-robin: every topic gets its best passage before any gets a second
        for q in SUMMARY_QUERIES:
            hits = retriever.search(q, k=3, doc_id=doc_id)
            if rank < len(hits):
                c = hits[rank].chunk
                if c.chunk_id not in picked and used + len(c.text) + 120 <= max_chars:
                    picked[c.chunk_id] = c
                    used += len(c.text) + 120
    return sorted(picked.values(), key=lambda c: c.index)


def summarize_document(
    llm: LLMClient, doc: Document, chunks: list[Chunk], cfg: SummaryConfig | None = None, max_workers: int = 4,
    retriever: HybridRetriever | None = None, strategy: str = "auto", max_chars: int = 0,
) -> ExecutiveSummary:
    """strategy: "auto" | "full" | "map_reduce" | "rag".

    auto = full text when it fits; otherwise "rag" when a retriever and a character budget are
    given (local models), else map-reduce over the whole document (large-context API models).
    """
    cfg = cfg or SummaryConfig()
    tier = length_tier(doc.word_count, cfg)
    if strategy == "auto":
        if doc.word_count <= cfg.single_pass_max_words:
            strategy = "full"
        elif retriever is not None and max_chars:
            strategy = "rag"
        else:
            strategy = "map_reduce"
    if strategy == "rag" and retriever is not None:
        selected = select_summary_chunks(retriever, doc.doc_id, chunks, max_chars or 40_000)
        return llm.structured(
            system=SYSTEM_ANALYST,
            prompt=summary_reduce_prompt(tier, doc.word_count, from_sections=False, excerpted=True),
            context=format_context(selected),
            output_model=ExecutiveSummary,
            max_tokens=6000 if tier == "long" else 4000,
            task="summary-rag",
        )
    if strategy == "full":
        return llm.structured(
            system=SYSTEM_ANALYST,
            prompt=summary_reduce_prompt(tier, doc.word_count, from_sections=False),
            context=format_context(chunks),
            output_model=ExecutiveSummary,
            max_tokens=6000 if tier == "long" else 4000,
            task="summary",
        )

    groups = [chunks[i: i + cfg.map_group_chunks] for i in range(0, len(chunks), cfg.map_group_chunks)]

    def summarize_group(args: tuple[int, list[Chunk]]) -> SectionSummary:
        i, group = args
        return llm.structured(
            system=SYSTEM_ANALYST,
            prompt=summary_map_prompt(i + 1, len(groups)),
            context=format_context(group),
            output_model=SectionSummary,
            max_tokens=3000,
            task=f"summary-map-{i + 1}",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = list(pool.map(summarize_group, enumerate(groups)))

    sections_ctx = "\n\n".join(
        f'<section_summary part="{i + 1}" excerpts="{group[0].chunk_id}..{group[-1].chunk_id}">\n'
        f"{p.summary}\n" + "\n".join(f"- {k}" for k in p.key_points) + "\n</section_summary>"
        for i, (p, group) in enumerate(zip(parts, groups))
    )
    return llm.structured(
        system=SYSTEM_ANALYST,
        prompt=summary_reduce_prompt(tier, doc.word_count, from_sections=True),
        context=sections_ctx,
        output_model=ExecutiveSummary,
        max_tokens=6000,
        task="summary-reduce",
    )
