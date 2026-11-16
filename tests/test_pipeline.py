"""End-to-end pipeline with a scripted mock model, plus a headless dashboard run."""

from pathlib import Path

import pytest

from ma_engine.llm.mock_client import MockLLMClient
from ma_engine.models import (
    AdvisorReport,
    DealTerms,
    ExecutiveSummary,
    Recommendation,
    RiskFlag,
    RiskReview,
)
from ma_engine.pipeline import MAEngine
from ma_engine.reporting.markdown_report import report_to_markdown


def _review(task, prompt, ctx):
    if "Termination & Break-up Fees" in task:
        import re
        cid = next(i for i, body in re.findall(r'<excerpt id="([^"]+)"[^>]*>\n(.*?)\n</excerpt>', ctx, re.S)
                   if "sole and exclusive remedy" in body)
        return RiskReview(flags=[RiskFlag(
            category="Termination & Break-up Fees", title="Reverse termination fee is not the exclusive remedy",
            severity="critical", description="Fee plus damages.", chunk_ids=[cid, "made-up-id"],
            estimated_exposure_usd=48e6, confidence=0.9)])
    return RiskReview()


@pytest.fixture
def engine(settings, sample_terms):
    llm = MockLLMClient(responses={
        ExecutiveSummary: ExecutiveSummary(headline="Halcyon buys Bluefin for $640M", deal_type="Reverse triangular merger",
                                           summary="Summary text."),
        DealTerms: sample_terms,
        RiskReview: _review,
        AdvisorReport: AdvisorReport(overall_stance="renegotiate", stance_rationale="Too many one-sided terms.",
                                     recommendations=[
                                         Recommendation(title="b", action_type="adjust_escrow", priority="P2", rationale="r"),
                                         Recommendation(title="a", action_type="walk_away", priority="P1", rationale="r")]),
    })
    return MAEngine(llm, settings)


def test_full_analysis(engine, raw_sample):
    ws = engine.build_workspace([raw_sample])
    steps = []
    r = engine.analyze(ws, progress=lambda m, f: steps.append(f))
    assert steps[-1] == 1.0
    assert r.summary.headline.startswith("Halcyon")
    assert r.deal_terms.deal_value_usd == 640e6
    llm_flags = [f for f in r.flags if f.source == "llm"]
    assert len(llm_flags) == 1 and llm_flags[0].severity == "critical"
    assert r.flags[0].severity == "critical" and r.flags[0].flag_id == "F01"
    assert "made-up-id" not in llm_flags[0].chunk_ids  # invented ids are dropped
    assert not any(f.title == "Reverse termination fee is not the sole remedy" for f in r.flags)  # merged into LLM flag
    assert len(r.simulations) == 5 and r.monte_carlo.runs == 300
    assert r.scorecard and 0 <= r.scorecard.total <= 100
    assert [x.priority for x in r.advice.recommendations] == ["P1", "P2"]
    assert len(r.warnings) == 1 and "minimum_cash_covenant" in r.warnings[0]  # covenant is never in the doc
    tasks = {c["task"] for c in engine.llm.calls}
    assert {"summary", "extract-terms", "advisor"} <= tasks and sum(t.startswith("risk:") for t in tasks) == 10

    md = report_to_markdown(r)
    assert "Confidence score" in md and "Recommended actions" in md
    assert r.model_validate_json(r.model_dump_json()).scorecard.total == r.scorecard.total


def test_recompute_financials_changes_numbers_without_llm(engine, raw_sample):
    ws = engine.build_workspace([raw_sample])
    r = engine.analyze(ws)
    calls = len(engine.llm.calls)
    r2 = engine.recompute_financials(r, {"acquirer_cash": 900e6})
    assert len(engine.llm.calls) == calls
    assert r2.simulations[0].cash[0] == pytest.approx(r.simulations[0].cash[0] + (900e6 - 410e6))
    assert r2.financial_inputs.sources["acquirer_cash"] == "user"
    assert r.simulations[0].cash[0] != r2.simulations[0].cash[0]  # original untouched
    engine.recompute_financials(r, None, rerun_advisor=True)
    assert len(engine.llm.calls) == calls + 1


def test_long_document_uses_map_reduce(settings, raw_sample):
    from ma_engine.ingestion.loader import load_text
    settings.summary.single_pass_max_words = 1000
    settings.summary.map_group_chunks = 3
    eng = MAEngine(MockLLMClient(), settings)
    ws = eng.build_workspace([load_text(raw_sample.text, "long")])
    eng.analyze(ws)
    tasks = [c["task"] for c in eng.llm.calls]
    assert sum(t.startswith("summary-map") for t in tasks) == 3 and "summary-reduce" in tasks


def test_ask(engine, raw_sample):
    ws = engine.build_workspace([raw_sample])
    ans, hits = engine.ask(ws, "What is the escrow amount?")
    assert hits and ans.answer
    with pytest.raises(ValueError):
        engine.ask(ws, "   ")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=300)


def _analyze(at, name="Halcyon"):
    at.run()
    assert not at.exception
    labels = at.selectbox[0].options
    at.selectbox[0].select_index(next(i for i, o in enumerate(labels) if name in o)).run()
    next(b for b in at.button if b.label == "Analyze agreement").click().run()
    assert not at.exception, at.exception


def test_dashboard_every_section_renders():
    at = _app()
    at.run()
    assert not at.text_input  # the app never asks for an API key
    _analyze(at)
    assert at.session_state.section == "Summary"
    sections = [
        "Document & RAG", "Summary", "Cash flow", "Score", "Growth", "Risks", "Mitigation", "Ask"]
    for name in sections:
        at.session_state.section = name
        at.run()
        assert not at.exception, (name, at.exception)
        assert not at.error, (name, [e.value for e in at.error])


def test_dashboard_assumptions_keep_section_and_mark_advice_stale():
    at = _app()
    _analyze(at)
    at.session_state.section = "Cash flow"
    at.run()
    next(n for n in at.number_input if n.label.startswith("Cash on hand")).set_value(900.0)
    next(b for b in at.button if b.label == "Update simulation").click().run()
    assert not at.exception
    assert at.session_state.report.financial_inputs.acquirer_cash == 900e6
    assert at.session_state.section == "Cash flow"


def test_dashboard_rag_search_ask_and_new_review():
    at = _app()
    _analyze(at)
    at.session_state.section = "Document & RAG"
    at.run()
    at.text_input(key="rag_query").set_value("termination fee").run()
    assert not at.exception and len(at.expander) >= 3
    at.session_state.section = "Ask"
    at.run()
    at.chat_input[0].set_value("What is the escrow amount?").run()
    assert not at.exception and len(at.chat_message) == 2
    next(b for b in at.button if b.label == "New review").click().run()
    assert not at.exception and any(b.label == "Analyze agreement" for b in at.button)
