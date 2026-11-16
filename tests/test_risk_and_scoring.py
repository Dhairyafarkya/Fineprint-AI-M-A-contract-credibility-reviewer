import pytest

from ma_engine.analysis.anomaly_detector import detect_anomalies
from ma_engine.analysis.risk_flagger import merge_flags
from ma_engine.analysis.risk_taxonomy import CATEGORY_NAMES, normalize_category
from ma_engine.analysis.rule_checks import RULES, run_rule_checks
from ma_engine.analysis.scoring import RUBRIC, score_deal
from ma_engine.config import ScoringWeights
from ma_engine.finance.input_builder import build_financial_inputs
from ma_engine.finance.mitigation_math import compute_mitigation_math
from ma_engine.finance.monte_carlo import run_monte_carlo
from ma_engine.finance.scenarios import default_scenarios
from ma_engine.finance.simulator import simulate
from ma_engine.ingestion.preprocess import preprocess
from ma_engine.models import Chunk, RiskFlag


@pytest.fixture(scope="module")
def doc_and_chunks(raw_sample):
    return preprocess(raw_sample)


@pytest.fixture(scope="module")
def chunks(doc_and_chunks):
    return doc_and_chunks[1]


REAL_WORLD_ONLY = {"INTERVENING_EVENT_OUT", "GO_SHOP", "TICKING_FEE"}


def test_every_rule_fires_on_sample(doc_and_chunks):
    doc, chunks = doc_and_chunks
    fired = {f.flag_id for f in run_rule_checks(chunks, doc.clean_text)}
    assert fired == {r.rule_id for r in RULES} - REAL_WORLD_ONLY


def test_rules_quiet_on_clean_text():
    clean = Chunk(chunk_id="c-c0000", doc_id="c", index=0, word_start=0, word_end=40,
                  text="The Company shall pay a termination fee of $10 million, which shall be the sole and exclusive "
                       "remedy. Parent has a matching right of four business days.")
    assert run_rule_checks([clean]) == []


def test_rule_categories_are_in_taxonomy():
    assert all(r.category in CATEGORY_NAMES for r in RULES)


def test_anomaly_detects_inconsistent_arr(chunks):
    flags = detect_anomalies(chunks)
    assert any(f.flag_id == "ANOMALY_ARR" and "13%" in f.title for f in flags)


def test_normalize_category():
    assert normalize_category("material adverse effect") == "Material Adverse Effect"
    assert normalize_category("Regulatory stuff") == "Regulatory & Antitrust"
    assert normalize_category("weird") == "Other"


def test_merge_flags_dedupes_and_orders():
    rule = RiskFlag(category="Financing", title="No financing condition for the buyer", severity="medium",
                    description="d", chunk_ids=["x-c0001"], source="rule")
    other = RiskFlag(category="Other", title="Broad non-reliance", severity="low", description="d",
                     chunk_ids=["x-c0002"], source="rule")
    llm = RiskFlag(category="Financing", title="Buyer must close without financing condition", severity="high",
                   description="d", chunk_ids=["x-c0001"], source="llm")
    merged = merge_flags([rule, other], [llm])
    assert [f.source for f in merged] == ["llm", "rule"]
    assert [f.flag_id for f in merged] == ["F01", "F02"]


def _score(terms, flags, weights=None):
    fi, _ = build_financial_inputs(terms, {"minimum_cash_covenant": 100e6})
    sims = [simulate(fi, s) for s in default_scenarios(fi, flags)]
    mc = run_monte_carlo(fi, runs=200)
    mm = compute_mitigation_math(fi, sims[0], flags)
    return score_deal(terms, fi, flags, sims, mc, mm, weights)


def test_scorecard_structure_and_bounds(sample_terms, chunks):
    flags = merge_flags(run_rule_checks(chunks) + detect_anomalies(chunks), [])
    card = _score(sample_terms, flags)
    assert [c.name for c in card.categories] == list(RUBRIC)
    assert sum(c.max_points for c in card.categories) == pytest.approx(100)
    assert card.total == pytest.approx(sum(c.awarded for c in card.categories), abs=0.1)
    for c in card.categories:
        for x in c.criteria:
            assert 0 <= x.awarded <= x.max_points and x.rationale
    assert card.rating[0] in "ABCD"


def test_fewer_flags_score_higher(sample_terms, chunks):
    flags = merge_flags(run_rule_checks(chunks), [])
    assert _score(sample_terms, []).total > _score(sample_terms, flags).total


def test_custom_weights_scale(sample_terms):
    w = ScoringWeights(financial_risk=40, legal_liability=20, regulatory_compliance=20, synergy_realization=20)
    card = _score(sample_terms, [], w)
    assert card.categories[0].max_points == 40
    assert sum(x.max_points for x in card.categories[0].criteria) == pytest.approx(40)


def test_definition_rules_span_passage_boundaries(doc_and_chunks):
    """The MAE definition starts in one passage and its carve-outs sit in the next."""
    doc, chunks = doc_and_chunks
    per_chunk = {f.flag_id for f in run_rule_checks(chunks)}
    full = {f.flag_id for f in run_rule_checks(chunks, doc.clean_text)}
    assert "MAE_CUSTOMER_CARVEOUT" in full and "MAE_CUSTOMER_CARVEOUT" not in per_chunk


def test_mae_rules_ignore_cyber_reps_outside_the_definition():
    text = ('"Company Material Adverse Effect" means any change that is materially adverse, other than changes in '
            'general economic conditions. ' + "Filler sentence about other matters. " * 120 +
            "Section 3.18 Data Privacy. The Company has not suffered any cybersecurity incident.")
    chunk = Chunk(chunk_id="x-c0000", doc_id="x", index=0, word_start=0, word_end=len(text.split()), text=text)
    assert "MAE_CYBER_CARVEOUT" not in {f.flag_id for f in run_rule_checks([chunk], text)}


def test_money_regex_ignores_stray_dollar_signs(chunks):
    stray = Chunk(chunk_id="s-c0000", doc_id="s", index=0, word_start=0, word_end=10,
                  text="ARR is $, as stated. Annual recurring revenue of $ , n/a. ARR of $12 million.")
    detect_anomalies([stray])  # must not raise
