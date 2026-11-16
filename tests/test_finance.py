import pytest

from ma_engine.finance.input_builder import build_financial_inputs
from ma_engine.finance.mitigation_math import break_even_synergy_realization, compute_mitigation_math
from ma_engine.finance.monte_carlo import run_monte_carlo
from ma_engine.finance.scenarios import default_scenarios, flagged_exposure
from ma_engine.finance.simulator import simulate, upfront_outflow
from ma_engine.models import DealTerms, FinancialInputs, RiskFlag, Scenario


@pytest.fixture
def fi(sample_terms):
    fi, warnings = build_financial_inputs(sample_terms, {"minimum_cash_covenant": 100e6})
    assert warnings == []
    return fi


def test_input_builder_sources(sample_terms):
    fi, _ = build_financial_inputs(sample_terms, {"acquirer_cash": 500e6, "escrow_release_month": 11.6})
    assert fi.sources["acquirer_cash"] == "user" and fi.acquirer_cash == 500e6
    assert fi.sources["target_debt"] == "extracted" and fi.target_debt == 85e6
    assert fi.escrow_release_month == 12 and isinstance(fi.escrow_release_month, int)
    assert fi.deal_value == 640e6


def test_input_builder_empty_terms_warns():
    fi, warnings = build_financial_inputs(DealTerms())
    assert warnings and "acquirer_cash" in warnings[0]
    assert all(v == "default" for k, v in fi.sources.items())


def test_input_builder_derives_cash_from_deal_value():
    fi, _ = build_financial_inputs(DealTerms(deal_value_usd=100e6, stock_consideration_usd=30e6))
    assert fi.cash_consideration == 70e6 and fi.target_debt == 0.0


def test_closing_cash_identity(fi):
    base = simulate(fi, Scenario(name="b"))
    expected = (fi.acquirer_cash + fi.new_acquisition_debt + fi.target_cash - fi.cash_consideration
                - fi.transaction_costs - fi.change_of_control_payments - fi.target_debt * 1.02)
    assert base.cash[0] == pytest.approx(expected)
    assert base.upfront_outflow == pytest.approx(upfront_outflow(fi, Scenario(name="b")))
    assert len(base.cash) == 37 and base.months[-1] == 36


def test_cash_roll_forward(fi):
    s = simulate(fi, Scenario(name="b"))
    principal = fi.new_acquisition_debt / (fi.acquisition_debt_amortization_years * 12)
    for t in range(1, 37):
        assert s.cash[t] == pytest.approx(s.cash[t - 1] + s.free_cash_flow[t] - principal)


def test_no_acceleration_keeps_debt_and_costs_interest(fi):
    no_acc = fi.model_copy(update={"debt_accelerates_on_change_of_control": False})
    a, b = simulate(fi, Scenario(name="a")), simulate(no_acc, Scenario(name="b"))
    assert b.cash[0] - a.cash[0] == pytest.approx(85e6 * 1.02)
    assert b.debt_balance[-1] > a.debt_balance[-1]


def test_hidden_liability_escrow_recovery(fi):
    base = simulate(fi, Scenario(name="b"))
    hit = simulate(fi, Scenario(name="h", hidden_liability=40e6, hidden_liability_month=6, hidden_liability_indemnifiable=0.5))
    late = simulate(fi, Scenario(name="h2", hidden_liability=40e6, hidden_liability_month=20, hidden_liability_indemnifiable=0.5))
    assert base.cash[5] == pytest.approx(hit.cash[5])
    assert base.cash[6] - hit.cash[6] == pytest.approx(20e6, rel=0.02)  # 20M recovered from 25M escrow (pre-tax ignored)
    assert base.cash[20] - late.cash[20] == pytest.approx(40e6, rel=0.02)  # escrow window (12m) has closed


def test_synergy_delay_and_realization(fi):
    base = simulate(fi, Scenario(name="b"))
    delayed = simulate(fi, Scenario(name="d", integration_delay_months=6, synergy_realization=0.5))
    assert delayed.synergies[6] == 0 and base.synergies[6] > 0
    assert base.synergies[30] == pytest.approx((38e6 + 30e6 * 0.6) / 12)
    assert delayed.synergies[36] == pytest.approx(base.synergies[36] * 0.5)
    assert delayed.metrics.npv_synergies < base.metrics.npv_synergies


def test_scenarios_ordered_and_severe_is_worst(fi):
    flags = [RiskFlag(category="Other", title="x", severity="high", description="d", estimated_exposure_usd=38e6)]
    assert flagged_exposure(flags) == 38e6
    sims = [simulate(fi, s) for s in default_scenarios(fi, flags)]
    assert sims[0].scenario.name == "Base Case"
    worst = min(sims, key=lambda s: s.metrics.ending_cash)
    assert worst.scenario.name == "Severe Downside"


def test_monte_carlo_reproducible(fi):
    a = run_monte_carlo(fi, runs=200, seed=7, liability_exposure=30e6)
    b = run_monte_carlo(fi, runs=200, seed=7, liability_exposure=30e6)
    assert a.cash_p50 == b.cash_p50
    assert 0 <= a.prob_covenant_breach <= 1
    assert all(lo <= mid <= hi for lo, mid, hi in zip(a.cash_p5, a.cash_p50, a.cash_p95))


def test_mitigation_math(fi):
    base = simulate(fi, Scenario(name="b"))
    mm = compute_mitigation_math(fi, base, [])
    assert mm.recommended_escrow_usd == pytest.approx(64e6)
    assert mm.escrow_gap_usd == pytest.approx(39e6)
    assert mm.price_cushion_usd == pytest.approx(base.metrics.minimum_cash - 100e6)
    tight = fi.model_copy(update={"minimum_cash_covenant": base.metrics.minimum_cash + 60e6})
    be = break_even_synergy_realization(tight, 36, 0.1)
    assert be is None or 0 < be <= 2
    assert break_even_synergy_realization(fi.model_copy(update={"minimum_cash_covenant": -1e12}), 36, 0.1) == 0.0


def test_zero_synergies_has_no_ratio():
    fi = FinancialInputs(annual_cost_synergies=0, annual_revenue_synergies=0)
    mm = compute_mitigation_math(fi, simulate(fi, Scenario(name="b")), [])
    assert mm.integration_cost_to_synergy_ratio is None
