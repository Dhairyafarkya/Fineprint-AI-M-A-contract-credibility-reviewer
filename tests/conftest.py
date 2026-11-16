from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ma_engine.config import get_settings  # noqa: E402
from ma_engine.ingestion.loader import load_file  # noqa: E402
from ma_engine.models import DealTerms  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_merger_agreement.txt"


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch, tmp_path):
    """Tests never hit the network, whatever key is in utils.py."""
    monkeypatch.setenv("MA_DEMO", "1")
    monkeypatch.setenv("MA_EMBEDDINGS", "tfidf")
    monkeypatch.setenv("MA_USE_CACHE", "0")  # never read or write the project's response cache
    settings_file = tmp_path / "test_utils.py"
    settings_file.write_text('LLM_PROVIDER = "gemini"\nGEMINI_API_KEY = ""\n')
    monkeypatch.setenv("MA_SETTINGS_FILE", str(settings_file))


@pytest.fixture
def settings(tmp_path):
    s = get_settings()
    s.llm.cache_dir = tmp_path / "llm-cache"
    s.simulation.monte_carlo_runs = 300
    return s


@pytest.fixture(scope="session")
def raw_sample():
    return load_file(SAMPLE)


@pytest.fixture
def sample_terms() -> DealTerms:
    """What Gemini should extract from the sample agreement."""
    return DealTerms(
        acquirer_name="Halcyon Robotics, Inc.", target_name="Bluefin Data Systems, Inc.",
        deal_value_usd=640e6, cash_consideration_usd=520e6, stock_consideration_usd=120e6, per_share_price_usd=18.40,
        termination_fee_usd=12e6, reverse_termination_fee_usd=48e6, escrow_usd=25e6, escrow_release_months=12,
        earnout_max_usd=60e6, earnout_period_months=24, change_of_control_payments_usd=14.5e6, target_debt_usd=85e6,
        debt_accelerates_on_change_of_control=True, target_debt_prepayment_premium=0.02,
        new_acquisition_debt_usd=350e6, acquisition_debt_interest_rate=0.076,
        expected_annual_cost_synergies_usd=38e6, expected_annual_revenue_synergies_usd=30e6, integration_cost_usd=52e6,
        integration_period_months=18, synergy_ramp_months=24, expected_close_months=6, outside_date_months=9,
        target_annual_revenue_usd=196.5e6, target_annual_operating_costs_usd=181.2e6, target_cash_usd=42e6,
        acquirer_cash_usd=410e6, acquirer_annual_revenue_usd=1.31e9, acquirer_annual_operating_costs_usd=1.12e9,
        transaction_costs_usd=21e6, regulatory_approvals_required=["HSR Act", "European Commission"],
        hell_or_high_water=True, financing_condition=False, mae_has_standard_carveouts=False,
        no_shop_has_fiduciary_out=True, indemnification_cap_pct=0.05, indemnification_survival_months=12,
    )
