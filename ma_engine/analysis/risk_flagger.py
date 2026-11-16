"""Fine-print risk flagging: rule checks + anomaly detection + Gemini review per category.

Flow:
1. Deterministic rule checks and numeric anomaly detection over all chunks.
2. For each risk category, retrieve the most relevant chunks (multi-query RAG,
   plus any chunks the rules already hit) and ask Gemini to confirm, refine or
   add flags, judged against market-practice guidance.
3. Merge: a Gemini flag that shares category and chunks with a rule flag
   replaces it; uncovered rule/anomaly flags are kept for human review.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm.base import LLMClient, LLMError
from ..llm.prompts import SYSTEM_ANALYST, risk_review_prompt
from ..models import Chunk, RiskFlag, RiskReview
from ..retrieval.retriever import HybridRetriever, format_context
from .anomaly_detector import detect_anomalies
from .risk_taxonomy import TAXONOMY, RiskCategory, normalize_category
from .rule_checks import run_rule_checks

log = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _rule_hits_text(flags: list[RiskFlag]) -> str:
    return "\n".join(f"- [{f.severity}] {f.title} (excerpts: {', '.join(f.chunk_ids)}): \"{f.clause_excerpt}\"" for f in flags)


def _review_category(
    llm: LLMClient, retriever: HybridRetriever, doc_id: str, cat: RiskCategory,
    rule_flags: list[RiskFlag], top_k: int, max_chars: int = 0,
) -> tuple[list[RiskFlag], str]:
    retrieved = retriever.search_many(list(cat.queries), k=top_k, doc_id=doc_id)
    chunks: list[Chunk] = [r.chunk for r in retrieved]
    have = {c.chunk_id for c in chunks}
    # Passages behind the pattern-check hits come first (at most two per flag) so the model can confirm them.
    lead: list[Chunk] = []
    for f in rule_flags:
        for cid in f.chunk_ids[:2]:
            c = retriever.index.get(cid)
            if c and c.chunk_id not in have:
                lead.append(c)
                have.add(c.chunk_id)
    chunks = lead + chunks
    if max_chars:
        kept, used = [], 0
        for c in chunks:
            if used + len(c.text) + 120 > max_chars:
                break
            kept.append(c)
            used += len(c.text) + 120
        chunks = kept or chunks[:1]
    review = llm.structured(
        system=SYSTEM_ANALYST,
        prompt=risk_review_prompt(cat.name, cat.guidance, _rule_hits_text(rule_flags)),
        context=format_context(chunks),
        output_model=RiskReview,
        task=f"risk:{cat.name}",
    )
    valid_ids = {c.chunk_id for c in chunks}
    out = []
    for f in review.flags:
        f.category = normalize_category(f.category) if f.category != cat.name else cat.name
        f.source = "llm"
        f.chunk_ids = [cid for cid in f.chunk_ids if cid in valid_ids] or f.chunk_ids
        out.append(f)
    return out, review.category_assessment


def merge_flags(deterministic: list[RiskFlag], llm_flags: list[RiskFlag]) -> list[RiskFlag]:
    merged = list(llm_flags)
    for d in deterministic:
        covered = any(
            l.category == d.category and set(l.chunk_ids) & set(d.chunk_ids) and _similar(l.title, d.title)
            for l in llm_flags
        )
        if not covered:
            merged.append(d)
    merged.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.category, -f.confidence))
    for i, f in enumerate(merged, start=1):
        f.flag_id = f"F{i:02d}"
    return merged


def _similar(a: str, b: str) -> bool:
    """Loose title match: share at least two meaningful words."""
    stop = {"the", "a", "of", "and", "on", "in", "to", "for", "is", "not", "no", "with", "by"}
    wa = {w.strip(".,()-").lower() for w in a.split()} - stop
    wb = {w.strip(".,()-").lower() for w in b.split()} - stop
    return len(wa & wb) >= 2


def flag_risks(
    llm: LLMClient,
    retriever: HybridRetriever,
    doc_id: str,
    top_k: int = 6,
    max_workers: int = 4,
    use_llm: bool = True,
    max_chars: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[RiskFlag], dict[str, str], list[str]]:
    """Return (flags, per-category assessments, warnings)."""
    chunks = retriever.index.chunks_for(doc_id)
    doc = retriever.index.documents.get(doc_id)
    deterministic = run_rule_checks(chunks, doc.clean_text if doc else None) + detect_anomalies(chunks)
    assessments: dict[str, str] = {}
    warnings: list[str] = []
    llm_flags: list[RiskFlag] = []
    if use_llm:
        by_cat = {c.name: [f for f in deterministic if f.category == c.name] for c in TAXONOMY}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_review_category, llm, retriever, doc_id, cat, by_cat[cat.name], top_k, max_chars): cat.name
                for cat in TAXONOMY
            }
            done = 0
            for fut in as_completed(futures):
                name = futures[fut]
                done += 1
                if on_progress:
                    on_progress(done, len(futures))
                try:
                    flags, assessment = fut.result()
                    llm_flags.extend(flags)
                    assessments[name] = assessment
                except LLMError as exc:
                    log.warning("Risk review failed for %s: %s", name, exc)
                    warnings.append(f"Gemini review of '{name}' failed: {exc}. Rule-based flags kept.")
    return merge_flags(deterministic, llm_flags), assessments, warnings
