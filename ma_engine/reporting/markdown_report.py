"""Render an AnalysisReport as a Markdown memo for download / sharing."""

from __future__ import annotations

from ..models import AnalysisReport
from ..utils.formatting import SEVERITY_ICON, money, month, pct


def report_to_markdown(r: AnalysisReport) -> str:
    out: list[str] = [f"# Fine Print review: {r.document.title.replace('_', ' ')}", ""]
    if r.demo_mode:
        out += ["> **Offline preview** - generated without Gemini; AI sections are placeholders.", ""]
    out += [f"*{r.document.word_count:,} words, {r.num_chunks} chunks analysed.*", ""]

    if r.scorecard:
        sc = r.scorecard
        out += [f"## Confidence score: {sc.total:.0f}/100 ({sc.rating})", "", sc.verdict, "",
                "| Category | Score | Confidence |", "|---|---|---|"]
        out += [f"| {c.name} | {c.awarded:.1f}/{c.max_points:.0f} | {c.confidence_pct:.0f}% |" for c in sc.categories]
        out += [""]

    if r.summary:
        s = r.summary
        out += ["## Executive summary", "", f"**{s.headline}**", "", s.summary, ""]
        if s.top_concerns:
            out += ["**Top concerns**", ""] + [f"- {c}" for c in s.top_concerns] + [""]

    if r.flags:
        out += ["## Risk flags", "", "| ID | Severity | Category | Issue | Source |", "|---|---|---|---|---|"]
        out += [f"| {f.flag_id} | {SEVERITY_ICON.get(f.severity, '')} {f.severity} | {f.category} | {f.title} | {f.source} |"
                for f in r.flags]
        out += [""]
        for f in r.flags:
            if f.severity in ("critical", "high"):
                out += [f"### {f.flag_id} - {f.title}", "", f.description, ""]
                if f.why_nonstandard:
                    out += [f"*Market practice:* {f.why_nonstandard}", ""]
                if f.clause_excerpt:
                    out += [f"> {f.clause_excerpt}", ""]

    if r.simulations:
        out += ["## Cash-flow scenarios", "", "| Scenario | Min cash | Ending cash | Covenant breach | NPV of FCF |",
                "|---|---|---|---|---|"]
        out += [f"| {s.scenario.name} | {money(s.metrics.minimum_cash)} | {money(s.metrics.ending_cash)} | "
                f"{month(s.metrics.first_covenant_breach_month)} | {money(s.metrics.npv_free_cash_flow)} |"
                for s in r.simulations]
        out += [""]
    if r.monte_carlo:
        mc = r.monte_carlo
        out += [f"Monte Carlo ({mc.runs:,} runs): {pct(mc.prob_covenant_breach)} probability of breaching the cash "
                f"covenant; median ending cash {money(mc.ending_cash_p50)}.", ""]

    if r.advice:
        a = r.advice
        out += ["## Recommended actions", "", f"**Stance: {a.overall_stance.replace('_', ' ')}** - {a.stance_rationale}", ""]
        for rec in a.recommendations:
            impact = f" (impact {money(rec.estimated_financial_impact_usd)})" if rec.estimated_financial_impact_usd else ""
            out += [f"### [{rec.priority}] {rec.title}{impact}", "", rec.rationale, ""]
            if rec.proposed_terms:
                out += [f"*Proposed terms:* {rec.proposed_terms}", ""]
        if a.walk_away_triggers:
            out += ["**Walk-away triggers**", ""] + [f"- {t}" for t in a.walk_away_triggers] + [""]

    if r.warnings:
        out += ["## Notes", ""] + [f"- {w}" for w in r.warnings] + [""]
    out += ["---", "*Decision support only - not legal or investment advice. Verify against the source documents.*"]
    return "\n".join(out)
