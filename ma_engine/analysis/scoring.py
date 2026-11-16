"""Multi-tier confidence scorecard (100 points) with a transparent rubric matrix.

Four categories, each made of criteria with fixed point budgets. Every
criterion is scored by an explicit, deterministic rule over the extracted deal
terms, risk flags and simulation outputs, and records WHY points were awarded
or docked. Category budgets come from ScoringWeights (default 30/25/20/25);
criterion points are scaled proportionally if the weights are changed.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import ScoringWeights
from ..models import (
    CategoryScore,
    CriterionScore,
    DealTerms,
    FinancialInputs,
    MitigationMath,
    MonteCarloResult,
    RiskFlag,
    ScoreCard,
    SimulationResult,
)

SEVERITY_WEIGHT = {"critical": 2.5, "high": 1.5, "medium": 0.5, "low": 0.0}
LEGAL_CATEGORIES = {
    "Termination & Break-up Fees", "Material Adverse Effect", "Hidden & Contingent Liabilities",
    "Indemnification & Escrow", "Deal Protection", "Debt & Change of Control", "Other",
}


def _m(x: float | None) -> str:
    return "n/a" if x is None else f"${x / 1e6:,.1f}M"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _tiers(value: float, tiers: list[tuple[float, float]], higher_is_better: bool = True) -> float:
    """Return points for the first threshold the value meets. tiers = [(threshold, points), ...]."""
    for threshold, pts in tiers:
        if (value >= threshold) if higher_is_better else (value <= threshold):
            return pts
    return 0.0


def _flags_in(flags: list[RiskFlag], cats: set[str] | str) -> list[RiskFlag]:
    cats = {cats} if isinstance(cats, str) else cats
    return [f for f in flags if f.category in cats]


def _flag_penalty(flags: list[RiskFlag]) -> float:
    return sum(SEVERITY_WEIGHT.get(f.severity, 0) * (0.5 + 0.5 * f.confidence) for f in flags)


def _flag_ids(flags: list[RiskFlag]) -> list[str]:
    return [f"{f.flag_id}: {f.title} ({f.severity})" for f in flags]


class _Ctx:
    def __init__(self, terms: DealTerms, fi: FinancialInputs, flags: list[RiskFlag],
                 sims: list[SimulationResult], mc: MonteCarloResult | None, mm: MitigationMath | None):
        self.t, self.fi, self.flags, self.mc, self.mm = terms, fi, flags, mc, mm
        self.sims = {s.scenario.name: s for s in sims}
        self.base = sims[0] if sims else None
        self.deal_value = fi.deal_value or (fi.cash_consideration + fi.stock_consideration) or None


# ----------------------------------------------------------------- Financial
def fin_liquidity(c: _Ctx, mx: float) -> CriterionScore:
    m = c.base.metrics
    ratio = (m.minimum_cash - c.fi.minimum_cash_covenant) / max(c.fi.minimum_cash_covenant, 1.0)
    if ratio >= 0:
        pts = _tiers(ratio, [(0.5, 1.0), (0.2, 0.8), (0.0, 0.6)]) * mx
    else:
        pts = max(0.0, 0.6 + 0.6 * ratio) * mx * 0.5
    return CriterionScore(
        name="Base-case liquidity headroom", max_points=mx, awarded=pts,
        rationale=f"Lowest base-case cash {_m(m.minimum_cash)} (month {m.minimum_cash_month}) vs covenant "
        f"{_m(c.fi.minimum_cash_covenant)}: headroom {ratio:+.0%}. Full marks at >=50% headroom.",
    )


def fin_downside(c: _Ctx, mx: float) -> CriterionScore:
    sev = c.sims.get("Severe Downside")
    if sev is None:
        return CriterionScore(name="Severe-downside survival", max_points=mx, awarded=mx / 2, rationale="Scenario not run.")
    m = sev.metrics
    if m.first_negative_cash_month is not None:
        pts, why = 0.0, f"cash turns negative in month {m.first_negative_cash_month}"
    elif m.first_covenant_breach_month is not None:
        pts, why = mx * 0.5, f"covenant breached in month {m.first_covenant_breach_month} but cash stays positive"
    else:
        pts, why = mx, "liquidity holds above the covenant throughout"
    return CriterionScore(name="Severe-downside survival", max_points=mx, awarded=pts,
                          rationale=f"Severe Downside scenario: {why} (min cash {_m(m.minimum_cash)}).")


def fin_monte_carlo(c: _Ctx, mx: float) -> CriterionScore:
    if c.mc is None:
        return CriterionScore(name="Probability of covenant breach", max_points=mx, awarded=mx / 2, rationale="Monte Carlo not run.")
    p = c.mc.prob_covenant_breach
    pts = _tiers(p, [(0.05, 1.0), (0.15, 0.75), (0.30, 0.5), (0.50, 0.25)], higher_is_better=False) * mx
    return CriterionScore(name="Probability of covenant breach", max_points=mx, awarded=pts,
                          rationale=f"{c.mc.runs:,} Monte Carlo runs: {_pct(p)} breach the cash covenant, "
                          f"{_pct(c.mc.prob_negative_cash)} run out of cash. Full marks at <=5%.")


def fin_leverage(c: _Ctx, mx: float) -> CriterionScore:
    fi, notes, pts = c.fi, [], mx
    if fi.debt_accelerates_on_change_of_control and fi.target_debt > 0:
        pts -= mx / 3
        notes.append(f"target debt of {_m(fi.target_debt)} accelerates at closing")
    combined_ebitda = (fi.acquirer_annual_revenue - fi.acquirer_annual_operating_costs
                       + fi.target_annual_revenue - fi.target_annual_operating_costs)
    total_debt = fi.new_acquisition_debt + (0 if fi.debt_accelerates_on_change_of_control else fi.target_debt)
    lev = total_debt / combined_ebitda if combined_ebitda > 0 else float("inf")
    if lev > 3:
        pts -= mx / 3
        notes.append(f"debt/EBITDA {lev:.1f}x (>3x)")
    elif lev > 2:
        pts -= mx / 6
        notes.append(f"debt/EBITDA {lev:.1f}x (2-3x)")
    else:
        notes.append(f"debt/EBITDA {lev:.1f}x")
    if fi.acquisition_debt_interest_rate > 0.09:
        pts -= mx / 6
        notes.append(f"acquisition debt costs {_pct(fi.acquisition_debt_interest_rate)}")
    return CriterionScore(name="Leverage & debt acceleration", max_points=mx, awarded=max(0.0, pts),
                          rationale=_cap("; ".join(notes))+ ".")


# --------------------------------------------------------------------- Legal
def legal_fees(c: _Ctx, mx: float) -> CriterionScore:
    half, notes, pts = mx / 2, [], 0.0
    dv = c.deal_value
    tf, rtf = c.t.termination_fee_usd, c.t.reverse_termination_fee_usd
    if tf and dv:
        p = tf / dv
        pts += half * (1.0 if 0.02 <= p <= 0.045 else 0.6 if p < 0.02 else 0.85)
        notes.append(f"target fee {_m(tf)} = {_pct(p)} of value (market 2-4.5%)")
    else:
        pts += half * 0.5
        notes.append("target termination fee not found")
    if rtf and dv:
        p = rtf / dv
        pts += half * _tiers(p, [(0.04, 1.0), (0.06, 0.7), (0.08, 0.3)], higher_is_better=False)
        notes.append(f"reverse fee {_m(rtf)} = {_pct(p)} of value (market <=4-6%)")
    else:
        pts += half * 0.7
        notes.append("no reverse termination fee found")
    fee_flags = _flags_in(c.flags, "Termination & Break-up Fees")
    severe = [f for f in fee_flags if f.severity in ("high", "critical")]
    if severe:
        pts -= min(pts, mx * 0.15 * len(severe))
        notes.append(f"{len(severe)} high-severity fee flag(s)")
    return CriterionScore(name="Termination fee structure", max_points=mx, awarded=max(0.0, pts),
                          rationale=_cap("; ".join(notes))+ ".", evidence=_flag_ids(fee_flags))


def legal_mae(c: _Ctx, mx: float) -> CriterionScore:
    mae_flags = _flags_in(c.flags, "Material Adverse Effect")
    worst = min((f.severity for f in mae_flags), key=lambda s: ["critical", "high", "medium", "low"].index(s), default=None)
    base = {"critical": 0.15, "high": 0.4, "medium": 0.7, "low": 0.9, None: 1.0}[worst]
    if worst is None and c.t.mae_has_standard_carveouts is None:
        base = 0.7
    pts = mx * base
    n_bad = len([f for f in mae_flags if f.severity in ("high", "critical")])
    pts = max(0.0, pts - mx * 0.1 * max(0, n_bad - 1))
    why = ("No MAE issues flagged" if not mae_flags else f"{len(mae_flags)} MAE issue(s); worst severity {worst}")
    return CriterionScore(name="MAE definition balance", max_points=mx, awarded=pts,
                          rationale=why + ". A buyer-protective MAE has market carve-outs with a disproportionate-effect qualifier.",
                          evidence=_flag_ids(mae_flags))


def legal_indemnity(c: _Ctx, mx: float) -> CriterionScore:
    dv, t = c.deal_value, c.t
    esc = (t.escrow_usd or c.fi.escrow_amount or 0.0) / dv if dv else 0.0
    cap = t.indemnification_cap_pct
    surv = t.indemnification_survival_months
    pts = mx * 0.5 * _tiers(esc, [(0.10, 1.0), (0.05, 0.66), (0.0001, 0.33)])
    pts += mx * 0.25 * (_tiers(cap, [(0.10, 1.0), (0.05, 0.5)]) if cap is not None else 0.5)
    pts += mx * 0.25 * (_tiers(surv, [(18, 1.0), (12, 0.5)]) if surv is not None else 0.5)
    ind_flags = [f for f in _flags_in(c.flags, "Indemnification & Escrow") if f.severity in ("high", "critical")]
    pts = max(0.0, pts - mx * 0.15 * len(ind_flags))
    return CriterionScore(
        name="Indemnification & escrow protection", max_points=mx, awarded=pts,
        rationale=f"Escrow {_pct(esc)} of value (market ~10%), cap {_pct(cap)}, survival "
        f"{'n/a' if surv is None else f'{surv:.0f} months'}; {len(ind_flags)} high-severity indemnity flag(s).",
        evidence=_flag_ids(ind_flags),
    )


def legal_unresolved(c: _Ctx, mx: float) -> CriterionScore:
    legal = _flags_in(c.flags, LEGAL_CATEGORIES)
    penalty = _flag_penalty(legal)
    pts = max(0.0, mx - penalty * mx / 12)
    counts = {s: sum(1 for f in legal if f.severity == s) for s in ("critical", "high", "medium", "low")}
    return CriterionScore(name="Open legal red flags", max_points=mx, awarded=pts,
                          rationale=f"{counts['critical']} critical, {counts['high']} high, {counts['medium']} medium, "
                          f"{counts['low']} low legal flags (weighted penalty {penalty:.1f}; zero points at >=12).",
                          evidence=_flag_ids([f for f in legal if f.severity in ("critical", "high")]))


# ---------------------------------------------------------------- Regulatory
def reg_burden(c: _Ctx, mx: float) -> CriterionScore:
    n = len(c.t.regulatory_approvals_required)
    pts = mx * {0: 0.85, 1: 0.85, 2: 0.7}.get(n, 0.5)  # 0 may just mean "none found"
    notes = [f"{n} regulatory approval(s) required" + (f": {', '.join(c.t.regulatory_approvals_required)}" if n else "")]
    if c.t.hell_or_high_water:
        pts -= mx * 0.45
        notes.append("unlimited divestiture (hell-or-high-water) obligation")
    return CriterionScore(name="Regulatory burden & remedy exposure", max_points=mx, awarded=max(0.0, pts),
                          rationale=_cap("; ".join(notes))+ ".")


def reg_timeline(c: _Ctx, mx: float) -> CriterionScore:
    close, outside = c.t.expected_close_months, c.t.outside_date_months
    if close is None or outside is None:
        return CriterionScore(name="Outside-date runway", max_points=mx, awarded=mx * 0.6,
                              rationale="Expected closing or outside date not found; partial credit.")
    runway = outside - close
    pts = mx * _tiers(runway, [(3, 1.0), (1, 0.6), (0, 0.3)])
    return CriterionScore(name="Outside-date runway", max_points=mx, awarded=pts,
                          rationale=f"Outside date at {outside:.0f} months vs expected close at {close:.0f}: "
                          f"{runway:.0f} months of runway (full marks at >=3).")


def reg_financing(c: _Ctx, mx: float) -> CriterionScore:
    fc = c.t.financing_condition
    if fc is False and c.fi.new_acquisition_debt > 0:
        pts, why = mx * 0.5, "no financing condition while relying on new debt: buyer bears financing risk"
    elif fc is True:
        pts, why = mx, "closing conditioned on financing"
    else:
        pts, why = mx * 0.8, "no new debt or financing terms not specified"
    return CriterionScore(name="Financing & closing certainty", max_points=mx, awarded=pts, rationale=_cap(why) + ".")


def reg_flags(c: _Ctx, mx: float) -> CriterionScore:
    rf = _flags_in(c.flags, {"Regulatory & Antitrust", "Financing"})
    pts = max(0.0, mx - _flag_penalty(rf) * mx / 5)
    return CriterionScore(name="Regulatory & financing red flags", max_points=mx, awarded=pts,
                          rationale=f"{len(rf)} regulatory/financing flag(s); weighted penalty {_flag_penalty(rf):.1f} (zero at >=5).",
                          evidence=_flag_ids(rf))


# ------------------------------------------------------------------ Synergies
def syn_payback(c: _Ctx, mx: float) -> CriterionScore:
    fi = c.fi
    run_rate = fi.annual_cost_synergies + fi.annual_revenue_synergies * fi.revenue_synergy_margin
    if run_rate <= 0:
        return CriterionScore(name="Integration cost payback", max_points=mx, awarded=0.0,
                              rationale="No synergies identified; integration costs are not recovered.")
    years = fi.integration_cost / run_rate
    pts = mx * _tiers(years, [(1.0, 1.0), (1.5, 0.8), (2.5, 0.55), (4.0, 0.3)], higher_is_better=False)
    return CriterionScore(name="Integration cost payback", max_points=mx, awarded=pts,
                          rationale=f"Integration cost {_m(fi.integration_cost)} vs run-rate synergy EBITDA "
                          f"{_m(run_rate)}/yr: {years:.1f} years to recover (full marks at <=1 year).")


def syn_value(c: _Ctx, mx: float) -> CriterionScore:
    npv = c.base.metrics.npv_synergies
    ratio = npv / c.deal_value if c.deal_value else 0.0
    pts = mx * _tiers(ratio, [(0.15, 1.0), (0.08, 0.75), (0.03, 0.5), (0.0001, 0.25)])
    return CriterionScore(name="Synergy value vs price", max_points=mx, awarded=pts,
                          rationale=f"{len(c.base.months) - 1}-month NPV of synergies net of integration {_m(npv)} = "
                          f"{_pct(ratio)} of deal value (full marks at >=15%).")


def syn_sensitivity(c: _Ctx, mx: float) -> CriterionScore:
    delayed = c.sims.get("Delayed Integration")
    if delayed is None:
        return CriterionScore(name="Sensitivity to integration delay", max_points=mx, awarded=mx / 2, rationale="Scenario not run.")
    b, d = c.base.metrics.npv_free_cash_flow, delayed.metrics.npv_free_cash_flow
    drop = (b - d) / abs(b) if b else 1.0
    pts = mx * _tiers(drop, [(0.05, 1.0), (0.10, 0.75), (0.20, 0.5), (0.35, 0.25)], higher_is_better=False)
    return CriterionScore(name="Sensitivity to integration delay", max_points=mx, awarded=pts,
                          rationale=f"A 6-month delay with 75% realisation cuts NPV of free cash flow by {_pct(drop)} "
                          f"({_m(b)} -> {_m(d)}). Full marks at <=5%.")


def syn_break_even(c: _Ctx, mx: float) -> CriterionScore:
    be = c.mm.break_even_synergy_realization if c.mm else None
    if be is None:
        return CriterionScore(name="Synergy break-even for liquidity", max_points=mx, awarded=0.0,
                              rationale="Even 200% of planned synergies does not keep cash above the covenant.")
    pts = mx * _tiers(be, [(0.0, 1.0), (0.5, 0.75), (0.8, 0.5), (1.0, 0.25)], higher_is_better=False)
    why = ("Liquidity holds even with zero synergies" if be == 0
           else f"At least {be:.0%} of planned synergies needed to stay above the covenant")
    return CriterionScore(name="Synergy break-even for liquidity", max_points=mx, awarded=pts, rationale=why + ".")


# ------------------------------------------------------------------- Rubric
Rule = Callable[[_Ctx, float], CriterionScore]
RUBRIC: dict[str, list[tuple[Rule, float]]] = {
    "Financial Risk": [(fin_liquidity, 10), (fin_downside, 8), (fin_monte_carlo, 6), (fin_leverage, 6)],
    "Legal Liability": [(legal_fees, 7), (legal_mae, 6), (legal_indemnity, 6), (legal_unresolved, 6)],
    "Regulatory Compliance": [(reg_burden, 7), (reg_timeline, 5), (reg_financing, 4), (reg_flags, 4)],
    "Synergy Realization": [(syn_payback, 7), (syn_value, 6), (syn_sensitivity, 6), (syn_break_even, 6)],
}

RATINGS = [
    (80, "A - Strong", "Proceed: risks are within market norms and liquidity is robust."),
    (65, "B - Acceptable with conditions", "Proceed only after fixing the flagged terms."),
    (50, "C - Renegotiate", "Material terms need to be renegotiated or re-priced before signing."),
    (0, "D - High risk", "Risk profile is poor; walk away unless the key terms change substantially."),
]


def rubric_matrix() -> list[dict[str, object]]:
    """The rubric itself (for display): category, criterion, default max points."""
    return [{"category": cat, "criterion": rule.__name__, "max_points": pts} for cat, rules in RUBRIC.items() for rule, pts in rules]


def score_deal(
    terms: DealTerms, fi: FinancialInputs, flags: list[RiskFlag], sims: list[SimulationResult],
    mc: MonteCarloResult | None, mm: MitigationMath | None, weights: ScoringWeights | None = None,
) -> ScoreCard:
    if not sims:
        raise ValueError("At least the base-case simulation is required to score the deal.")
    weights = weights or ScoringWeights()
    ctx = _Ctx(terms, fi, flags, sims, mc, mm)
    categories: list[CategoryScore] = []
    for cat_name, rules in RUBRIC.items():
        budget = weights.as_dict()[cat_name]
        scale = budget / sum(p for _, p in rules)
        crits = []
        for rule, pts in rules:
            cs = rule(ctx, pts * scale)
            cs.awarded = round(min(cs.max_points, max(0.0, cs.awarded)), 2)
            cs.max_points = round(cs.max_points, 2)
            crits.append(cs)
        categories.append(CategoryScore(name=cat_name, max_points=round(budget, 2),
                                        awarded=round(sum(x.awarded for x in crits), 2), criteria=crits))
    total = round(sum(c.awarded for c in categories), 1)
    rating, verdict = next((r, v) for threshold, r, v in RATINGS if total >= threshold)
    return ScoreCard(total=total, rating=rating, verdict=verdict, categories=categories)
