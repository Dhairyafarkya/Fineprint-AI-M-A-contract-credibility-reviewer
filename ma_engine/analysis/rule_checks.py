"""Deterministic pattern checks for well-known risky clause types.

These run before (and independently of) Gemini. They are cheap, explainable and
give the LLM reviewer concrete leads. Each rule fires at most once per document
and records every chunk where it matched.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from ..models import Chunk, RiskFlag

_FLAGS = re.I | re.S


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    title: str
    severity: str
    patterns: tuple[str, ...]  # all must match within `window` characters of the first
    description: str
    why_nonstandard: str
    window: int = 600
    after_only: bool = False  # look for the other patterns only after the first one (inside a definition)


RULES: tuple[Rule, ...] = (
    Rule("MAE_NO_DISPROPORTIONATE", "Material Adverse Effect", "MAE carve-outs apply even if the target is disproportionately affected",
         "high", (r"whether or not[^.]{0,80}disproportionately",),
         "Industry-wide and macro events excuse the target even where it is hit far harder than peers.",
         "Market MAE definitions claw back carve-outs that disproportionately affect the target.", 300),
    Rule("MAE_CUSTOMER_CARVEOUT", "Material Adverse Effect", "MAE excludes loss of customers",
         "high", (r"Material Adverse Effect\W{0,3}\s*(?:shall )?mean", r"(?<!announcement )(?<!pendency )loss of[^.;]{0,60}customer(?![^.;]{0,120}announcement)"),
         "Losing major customers between signing and closing would not let the buyer walk.",
         "Customer-loss carve-outs are uncommon outside of announcement-effect language.", 3500, after_only=True),
    Rule("MAE_CYBER_CARVEOUT", "Material Adverse Effect", "MAE excludes cybersecurity incidents",
         "medium", (r"Material Adverse Effect\W{0,3}\s*(?:shall )?mean", r"(?:cyber-?attack|cyber-?security|data breach|ransomware|hacking)"),
         "A breach or ransomware event before closing cannot be used to terminate.",
         "Cyber carve-outs are becoming more common but still shift a real risk to the buyer.", 3500, after_only=True),
    Rule("HELL_OR_HIGH_WATER", "Regulatory & Antitrust", "Unlimited divestiture obligation (hell-or-high-water)",
         "critical", (r"any and all (?:actions|steps)", r"(?:divest|divestiture|hold separate)"),
         "The buyer must accept any remedy regulators demand, regardless of the effect on deal value.",
         "Market practice is reasonable best efforts, often with a cap on required divestitures.", 800),
    Rule("REVERSE_FEE_NOT_EXCLUSIVE", "Termination & Break-up Fees", "Reverse termination fee is not the sole remedy",
         "high", (r"shall not be the[^.]{0,60}sole and exclusive remedy",),
         "The buyer can owe the fee AND damages / specific performance.",
         "Reverse fees are normally the target's exclusive remedy and cap buyer exposure.", 300),
    Rule("LOST_PREMIUM_DAMAGES", "Termination & Break-up Fees", "Buyer exposed to lost-premium damages",
         "high", (r"loss of the economic benefits",),
         "Damages measured by the stockholders' lost premium can far exceed the reverse fee.",
         "Lost-premium damages are contested and usually excluded or capped by buyers.", 300),
    Rule("DEBT_ACCELERATION", "Debt & Change of Control", "Target debt accelerates on change of control",
         "high", (r"change of control", r"(?:immediately due and payable|accelerat|event of default)",
                  r"(?:indebtedness|credit agreement|credit facility|lenders?|loan|notes|borrow)"),
         "Closing triggers repayment of target borrowings (plus any prepayment premium) in cash.",
         "Buyers normally require payoff letters or lender consents and price the payoff into the deal.", 500),
    Rule("NO_LENDER_CONSENT", "Debt & Change of Control", "Target has no obligation to obtain lender consent",
         "medium", (r"(?:no obligation|not obtained)[^.]{0,80}(?:waiver|consent)",),
         "The buyer bears the payoff or refinancing of target debt.",
         "Targets usually covenant to cooperate on consents and payoff letters.", 300),
    Rule("SINGLE_TRIGGER_COC", "Debt & Change of Control", "Single-trigger change-of-control payments",
         "medium", (r"(?:single-trigger|alone and without any further act)",),
         "Executives are paid on closing even if they stay; equity vesting accelerates.",
         "Double-trigger (closing + termination) is the buyer-friendly norm.", 400),
    Rule("ARR_INFLATED_DEFINITION", "Revenue & Financial Metrics", "Revenue metric definition likely overstates the business",
         "high", (r"(?:annual recurring revenue|\bARR\b|adjusted revenue|bookings)",
                  r"(?:pilot|trial|proof-of-concept|not yet commenced|termination for convenience)"),
         "The metric counts pilots, not-yet-live or easily cancellable contracts as recurring revenue.",
         "Market ARR counts live, committed, non-cancellable subscription revenue only.", 900),
    Rule("SELLER_CALCULATED_EARNOUT", "Other", "Earnout metric calculated by the sellers",
         "medium", (r"calculated by the (?:stockholder|seller|holder)s?'? representative",),
         "The sellers' representative controls the earnout measurement.",
         "The buyer normally prepares the earnout statement, with a dispute mechanism.", 300),
    Rule("EARNOUT_ACCELERATION", "Other", "Earnout accelerates in full on buyer actions",
         "high", (r"earnout", r"(?:immediately due and payable|become[s]? immediately due)"),
         "The full earnout becomes payable on a buyer change of control or strategy change, regardless of performance.",
         "Acceleration triggers should be narrow and tied to bad-faith interference.", 1200),
    Rule("UNRESERVED_LITIGATION", "Hidden & Contingent Liabilities", "Disclosed litigation with no reserve",
         "high", (r"(?:has not established any reserve|no (?:amount|reserve) (?:has been )?(?:accrued|established))",),
         "A pending claim is disclosed but nothing is accrued for it.",
         "Material known claims are normally reserved, priced or specially indemnified.", 300),
    Rule("ONGOING_INVESTIGATION", "Hidden & Contingent Liabilities", "Ongoing incident / investigation with undetermined consequences",
         "medium", (r"(?:remains? ongoing|have not been determined|not yet been determined)",),
         "The financial and regulatory consequences of a known incident are still open.",
         "Open investigations usually get a special indemnity or escrow.", 300),
    Rule("INDEMNITY_EXCLUDES_DISCLOSED", "Indemnification & Escrow", "Known/disclosed risks excluded from indemnity",
         "high", (r"no obligation to indemnify",),
         "Specific risks (often the largest known ones) are carved out of the sellers' indemnity.",
         "Buyers usually get special indemnities for known issues rather than exclusions.", 300),
    Rule("ANTI_SANDBAGGING", "Indemnification & Escrow", "Anti-sandbagging: buyer's knowledge bars claims",
         "medium", (r"(?:knowledge prior to the closing|deemed to have knowledge)",),
         "Anything in the data room is treated as known and cannot be claimed.",
         "Buyers typically insist on pro-sandbagging or silence on the point.", 400),
    Rule("NO_FINANCING_CONDITION", "Financing", "No financing condition for the buyer",
         "medium", (r"not (?:be )?(?:subject to any condition|subject to|conditioned (?:up)?on)[^.]{0,80}financing",),
         "The buyer must close even if its debt financing fails.",
         "Market for strategic buyers, but exposure should be capped by the reverse fee.", 300),
    Rule("NO_MATCHING_RIGHT", "Deal Protection", "Buyer has no matching right",
         "medium", (r"no matching right",),
         "The target can accept a topping bid without giving the buyer a chance to match.",
         "Matching rights (3-5 business days) are standard buyer protection.", 200),
    Rule("INTERIM_PRICING_FREEDOM", "Deal Protection", "Target may reprice contracts before closing without consent",
         "medium", (r"without the consent of parent", r"(?:price concessions|amend existing customer contracts)"),
         "The target can cut prices or rewrite customer deals between signing and closing.",
         "Interim covenants normally require buyer consent for material contract changes.", 400),
    Rule("FOREIGN_APPROVALS", "Regulatory & Antitrust", "Foreign antitrust / investment approvals required",
         "medium", (r"(?:State Administration for Market Regulation|\bSAMR\b|\bCFIUS\b|foreign investment|"
                    r"European Commission|Competition and Markets Authority|foreign antitrust)",),
         "Closing depends on non-US regulators, which adds timing risk and possible remedies.",
         "Pair with a long enough outside date and clear allocation of remedy obligations.", 200),
    Rule("INTERVENING_EVENT_OUT", "Deal Protection", "Target board may change recommendation for an Intervening Event",
         "low", (r"Intervening Event",),
         "The target board can withdraw its recommendation for reasons other than a competing bid.",
         "Common in Delaware deals, but buyers seek a narrow definition and a termination fee trigger.", 200),
    Rule("GO_SHOP", "Deal Protection", "Go-shop period lets the target solicit competing bids",
         "medium", (r"go-shop|No-Shop Period Start Date|Window Period End",),
         "The target may actively shop the deal after signing, often with a reduced break fee.",
         "Go-shops are negotiated concessions; check the reduced fee and matching rights.", 200),
    Rule("TICKING_FEE", "Termination & Break-up Fees", "Ticking fee increases the price if closing is delayed",
         "medium", (r"ticking fee",),
         "The buyer pays more the longer regulatory approval takes.",
         "Uncommon; usually only where the buyer controls regulatory risk.", 200),
    Rule("KNOWLEDGE_NO_INQUIRY", "Other", "Knowledge qualifier without duty of inquiry",
         "low", (r"without any duty of inquiry",),
         "Knowledge-qualified reps only cover what one person actually knows.",
         "Buyers usually require knowledge after reasonable inquiry of a group of officers.", 200),
    Rule("NON_RELIANCE", "Other", "Broad non-reliance clause",
         "low", (r"has not relied on any other information",),
         "Claims based on projections, presentations or data-room material are cut off.",
         "Common, but should carve out fraud.", 300),
)


def _sentence_around(text: str, pos: int, max_len: int = 320) -> str:
    """The sentence or semicolon-delimited clause containing `pos`."""
    seps = (". ", "; ", "\n")
    start = max(text.rfind(sep, 0, pos) for sep in seps)
    start = 0 if start < 0 else start + 1
    end_candidates = [i for i in (text.find(sep, pos) for sep in seps) if i >= 0]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    s = re.sub(r"\s+", " ", text[start:end]).strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _match(rule: Rule, text: str) -> int | None:
    """Character position of a match if all of the rule's patterns co-occur within its window."""
    first = list(re.finditer(rule.patterns[0], text, _FLAGS))
    for m in first:
        lo = m.start() if rule.after_only else max(0, m.start() - rule.window)
        hi = m.end() + rule.window
        if all(re.search(p, text[lo:hi], _FLAGS) for p in rule.patterns[1:]):
            # Report the operative pattern's location (the second one, when there is one).
            key = re.search(rule.patterns[1], text[lo:hi], _FLAGS) if len(rule.patterns) > 1 else None
            return lo + key.start() if key else m.start()
    return None


def _all_matches(rule: Rule, text: str) -> list[int]:
    """Positions of every place the rule matches (one per occurrence of its first pattern)."""
    hits = []
    for m in re.finditer(rule.patterns[0], text, _FLAGS):
        lo = m.start() if rule.after_only else max(0, m.start() - rule.window)
        hi = m.end() + rule.window
        if all(re.search(p, text[lo:hi], _FLAGS) for p in rule.patterns[1:]):
            key = re.search(rule.patterns[1], text[lo:hi], _FLAGS) if len(rule.patterns) > 1 else None
            hits.append(lo + key.start() if key else m.start())
    return hits


def _chunks_at(chunks: list[Chunk], word_index: int) -> list[str]:
    return [c.chunk_id for c in chunks if c.word_start <= word_index < c.word_end]


def run_rule_checks(chunks: list[Chunk], text: str | None = None) -> list[RiskFlag]:
    """Run every rule over the full document text (so clauses that span passages are caught),
    then report the passages that contain each hit. Without `text`, rules run per passage."""
    if text is None:
        return _run_per_chunk(chunks)
    word_starts = [m.start() for m in re.finditer(r"\S+", text)]
    flags: list[RiskFlag] = []
    for rule in RULES:
        hit_ids: list[str] = []
        excerpt = ""
        for pos in _all_matches(rule, text):
            widx = max(0, bisect_right(word_starts, pos) - 1)
            for cid in _chunks_at(chunks, widx):
                if cid not in hit_ids:
                    hit_ids.append(cid)
            candidate = _sentence_around(text, pos)
            if not excerpt or (len(excerpt.split()) < 12 <= len(candidate.split())):
                excerpt = candidate
        if hit_ids:
            flags.append(_flag(rule, hit_ids, excerpt))
    return flags


def _flag(rule: Rule, hit_ids: list[str], excerpt: str) -> RiskFlag:
    return RiskFlag(
        flag_id=rule.rule_id, category=rule.category, title=rule.title,
        severity=rule.severity,  # type: ignore[arg-type]
        description=rule.description, why_nonstandard=rule.why_nonstandard,
        clause_excerpt=excerpt, chunk_ids=hit_ids, source="rule", confidence=0.6,
    )


def _run_per_chunk(chunks: list[Chunk]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for rule in RULES:
        hit_ids: list[str] = []
        excerpt = ""
        for c in chunks:
            pos = _match(rule, c.text)
            if pos is not None:
                hit_ids.append(c.chunk_id)
                candidate = _sentence_around(c.text, pos)
                # Prefer a real sentence over an index / table-of-contents entry.
                if not excerpt or (len(excerpt.split()) < 12 <= len(candidate.split())):
                    excerpt = candidate
        if hit_ids:
            flags.append(_flag(rule, hit_ids, excerpt))
    return flags
