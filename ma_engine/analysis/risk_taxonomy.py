"""Risk categories: retrieval queries and market-practice guidance for each.

The queries drive multi-query RAG retrieval; the guidance is injected into the
prompt so Gemini judges each clause against a consistent benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCategory:
    name: str
    queries: tuple[str, ...]
    guidance: str


TAXONOMY: tuple[RiskCategory, ...] = (
    RiskCategory(
        "Termination & Break-up Fees",
        ("termination fee payable", "reverse termination fee parent", "sole and exclusive remedy damages",
         "termination of this agreement outside date"),
        "- Target break-up fees usually run 2-4% of equity value; below ~2% gives the buyer weak protection.\n"
        "- Reverse termination fees typically 4-7% (higher when antitrust or financing risk is large).\n"
        "- A reverse fee that is NOT the target's sole remedy (plus uncapped damages / lost-premium damages) "
        "is aggressive against the buyer.",
    ),
    RiskCategory(
        "Material Adverse Effect",
        ("material adverse effect definition", "MAE carve-outs exceptions disproportionately affect",
         "failure to meet projections customer loss cybersecurity"),
        "- Standard carve-outs: general economy, industry, law/GAAP changes, war, pandemics - normally with "
        "a 'disproportionate effect' qualifier.\n"
        "- Carving out customer losses, cyber incidents, or the underlying causes of missed projections is "
        "target-friendly and leaves the buyer almost no walk right.\n"
        "- Missing disproportionate-effect qualifier is a red flag for the buyer.",
    ),
    RiskCategory(
        "Hidden & Contingent Liabilities",
        ("pending litigation claims damages reserve", "undisclosed liabilities", "data breach investigation",
         "environmental tax contingent liabilities"),
        "- Disclosed but unreserved litigation, ongoing regulatory investigations, data breaches and "
        "tax exposures should be priced, specially indemnified or escrowed.\n"
        "- 'Would not have a Material Adverse Effect' qualifiers on the liabilities rep weaken protection.",
    ),
    RiskCategory(
        "Revenue & Financial Metrics",
        ("annual recurring revenue definition", "financial statements revenue operating expenses",
         "adjusted revenue EBITDA calculation", "bookings backlog pipeline"),
        "- Non-GAAP metrics (ARR, adjusted revenue, bookings) defined to include pilots, unsigned/not-live "
        "contracts or contracts cancellable on short notice overstate the business.\n"
        "- Inconsistent figures for the same metric, or growth claims not supported by the numbers, are "
        "anomalies requiring a quality-of-earnings review.",
    ),
    RiskCategory(
        "Debt & Change of Control",
        ("change of control event of default accelerate", "indebtedness credit agreement payoff",
         "change of control payments severance single trigger", "consent waiver lenders"),
        "- Target debt that accelerates on closing (with prepayment premiums) is a hidden cash cost.\n"
        "- Single-trigger change-of-control payments and full equity acceleration are less buyer-friendly "
        "than double-trigger.\n"
        "- The target normally covenants to seek lender consents or deliver payoff letters.",
    ),
    RiskCategory(
        "Regulatory & Antitrust",
        ("HSR antitrust approvals efforts", "divestiture remedy hell or high water", "regulatory conditions",
         "foreign investment European Commission"),
        "- 'Reasonable best efforts' is market; an unlimited 'hell-or-high-water' obligation (any divestiture "
        "regardless of effect) shifts all antitrust risk to the buyer.\n"
        "- Pair with the size of the reverse termination fee and the outside-date runway.",
    ),
    RiskCategory(
        "Indemnification & Escrow",
        ("indemnification cap deductible basket", "escrow amount release survival period",
         "exclusions from indemnification knowledge sandbagging", "sole source of recovery fraud"),
        "- Private-target norms: escrow ~10% of price, cap ~10%, deductible 0.5-1%, survival 12-18 months "
        "(longer for fundamental and tax reps).\n"
        "- Excluding disclosed matters, pre-closing taxes, or anything the buyer 'knew' (anti-sandbagging) "
        "from indemnity strips protection.",
    ),
    RiskCategory(
        "Deal Protection",
        ("no solicitation superior proposal fiduciary out", "matching rights", "conduct of business ordinary course",
         "go-shop window"),
        "- Buyers usually get matching rights and a no-shop with a narrow fiduciary out.\n"
        "- Interim operating covenants should stop the target from repricing or renegotiating contracts "
        "without consent.",
    ),
    RiskCategory(
        "Financing",
        ("financing condition debt commitment letter", "sufficient funds", "financing failure"),
        "- Public deals normally have no financing condition, but the buyer's exposure must be capped by "
        "the reverse termination fee; uncapped damages plus no financing condition is high risk.",
    ),
    RiskCategory(
        "Other",
        ("non-reliance representations", "earnout payment acceleration", "knowledge definition duty of inquiry",
         "specific performance governing law"),
        "- Earnouts measured by the sellers, or accelerating on buyer actions, are aggressive.\n"
        "- Knowledge defined as one officer's actual knowledge without inquiry narrows the reps.\n"
        "- Broad non-reliance clauses limit fraud claims on management materials.",
    ),
)

CATEGORY_NAMES = tuple(c.name for c in TAXONOMY)


def get_category(name: str) -> RiskCategory | None:
    return next((c for c in TAXONOMY if c.name.lower() == name.lower()), None)


def normalize_category(name: str) -> str:
    """Map a free-text category from Gemini onto the taxonomy (fallback 'Other')."""
    low = name.lower()
    for c in TAXONOMY:
        if c.name.lower() == low:
            return c.name
    for c in TAXONOMY:
        first_word = c.name.split()[0].lower()
        if first_word in low:
            return c.name
    return "Other"
