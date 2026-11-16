"""Monte Carlo over the uncertain drivers of the base case.

Distributions (seeded, so results are reproducible):
    synergy realisation  ~ triangular(0.4, 0.85, 1.1)
    integration delay    ~ months in {0, 3, 6, 9, 12} with probs {.35, .30, .20, .10, .05}
    integration overrun  ~ triangular(0, 0.15, 0.6)
    revenue shock        ~ normal(0, 4%), clipped to +/-20%
    hidden liability     ~ Bernoulli(p) x lognormal around the flagged exposure
    rate shock           ~ normal(0, 75 bps), floored at -100 bps
    earnout              ~ Bernoulli(earnout_probability)
"""

from __future__ import annotations

import numpy as np

from ..models import FinancialInputs, MonteCarloResult, Scenario
from .simulator import simulate


def run_monte_carlo(
    fi: FinancialInputs,
    *,
    runs: int = 2000,
    horizon: int = 36,
    seed: int = 42,
    liability_exposure: float = 0.0,
    liability_probability: float = 0.3,
    annual_discount_rate: float = 0.10,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    realization = rng.triangular(0.4, 0.85, 1.1, runs)
    delay = rng.choice([0, 3, 6, 9, 12], size=runs, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    overrun = rng.triangular(0.0, 0.15, 0.6, runs)
    rev_shock = np.clip(rng.normal(0.0, 0.04, runs), -0.2, 0.2)
    has_liab = rng.random(runs) < liability_probability
    liab_size = liability_exposure * rng.lognormal(mean=0.0, sigma=0.5, size=runs) if liability_exposure > 0 else np.zeros(runs)
    liab_month = rng.integers(3, 19, size=runs)
    rate_shock = np.maximum(-0.01, rng.normal(0.0, 0.0075, runs))
    earnout = rng.random(runs) < fi.earnout_probability

    paths = np.empty((runs, horizon + 1))
    npvs = np.empty(runs)
    for i in range(runs):
        sc = Scenario(
            name=f"mc-{i}",
            synergy_realization=float(realization[i]),
            integration_delay_months=int(delay[i]),
            integration_cost_overrun=float(overrun[i]),
            revenue_shock=float(rev_shock[i]),
            hidden_liability=float(liab_size[i]) if has_liab[i] else 0.0,
            hidden_liability_month=int(liab_month[i]),
            hidden_liability_indemnifiable=0.3,
            interest_rate_shock=float(rate_shock[i]),
            earnout_paid=bool(earnout[i]),
        )
        res = simulate(fi, sc, horizon, annual_discount_rate)
        paths[i] = res.cash
        npvs[i] = res.metrics.npv_free_cash_flow

    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    mins = paths.min(axis=1)
    return MonteCarloResult(
        runs=runs,
        months=list(range(horizon + 1)),
        cash_p5=p5.tolist(),
        cash_p50=p50.tolist(),
        cash_p95=p95.tolist(),
        prob_covenant_breach=float((mins < fi.minimum_cash_covenant).mean()),
        prob_negative_cash=float((mins < 0).mean()),
        ending_cash_p5=float(np.percentile(paths[:, -1], 5)),
        ending_cash_p50=float(np.percentile(paths[:, -1], 50)),
        ending_cash_p95=float(np.percentile(paths[:, -1], 95)),
        expected_npv_fcf=float(npvs.mean()),
    )
