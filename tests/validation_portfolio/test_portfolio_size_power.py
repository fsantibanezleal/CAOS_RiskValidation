"""RV-U16-E2: the size and power of the three portfolio p-value tests: the binomial tail back-test (two levels), the
Monte Carlo z-test against a closed-form anchor and the Kolmogorov distance of the CreditRisk+ recursion against a
Monte Carlo sample drawn from the model's definition. No anchor is computed by the simulation it checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.generators.montecarlo import McQuantile
from riskvalidation.generators.size_studies import study
from riskvalidation.harness import agrees, simulate, size_bound, within_size
from size_kit import ROOT, check, page_paragraph

# U16-E2 (BL-088): a 4,000-repetition local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "portfolio" / "07_validation-tests.md"


def _exact_binomial_size(n: int, level: float, alpha: float) -> float:
    """The exact rejection probability of the tail back-test at p < alpha: P(X >= k) < alpha for X ~ Bin(n, 1 - q)."""
    k = np.arange(n + 1)
    reject = stats.binom.sf(k - 1, n, 1.0 - level) < alpha
    return float(stats.binom.pmf(k, n, 1.0 - level)[reject].sum())


def test_size_and_power_of_the_tail_backtest():
    """portfolio.tail_backtest at two levels: the exceedance count is exactly binomial under the null, so the measured
    size must agree with the exact rejection probability of the discrete test (conservative, never above nominal)."""
    for index, (n, level) in enumerate(((60, 0.95), (250, 0.99))):
        st = study("portfolio.tail_backtest", index)
        m = check(st, PAGE)
        for alpha, measured in ((0.05, m.size5), (0.01, m.size1)):
            exact = _exact_binomial_size(n, level, alpha)
            assert exact <= alpha
            assert f"{100 * exact:.2f}%" in page_paragraph(PAGE, st), (n, level, alpha, "page states the exact size")
            assert agrees(m.rate5 if alpha == 0.05 else m.rate1, exact), (n, level, alpha, measured, exact)


def test_size_and_power_of_the_mc_convergence():
    """portfolio.mc_convergence against the closed form of the LHP quantile. The standard error is estimated from the
    sample (Woodruff), as a caller does; a documented excess at 1% comes from that estimate, shown by the diagnostic
    with the exact standard error, which holds its size."""
    m = check(study("portfolio.mc_convergence"), PAGE)
    assert m.size5 <= m.bound5 and m.size1 > m.bound1
    exact = simulate("portfolio.mc_convergence", McQuantile(20000, se_method="asymptotic"), n_rep=4000, seed=5409)
    assert within_size(exact.rate(0.05), 0.05) and within_size(exact.rate(0.01), 0.01), (
        exact.rate(0.05).to_dict(), exact.rate(0.01).to_dict(), size_bound(0.01, 4000))
    para = open(PAGE, encoding="utf-8").read()
    assert f"{100 * exact.rate(0.05).rate:.2f}%" in para and f"{100 * exact.rate(0.01).rate:.2f}%" in para


def test_size_and_power_of_the_crp_recursion_vs_mc():
    """portfolio.crp_recursion_vs_mc: the recursion's pmf against a sample from the Gamma-Poisson definition. The
    Kolmogorov law is asymptotic for a continuous distribution, so the test is conservative on loss units."""
    m = check(study("portfolio.crp_recursion_vs_mc"), PAGE)
    assert m.size5 < 0.05 and m.size1 < 0.01
