"""RV-103, RV-112: stability against Yurdakul-Naranjo (2020) and SciPy references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.validation import (
    bin_counts,
    psi_benchmark,
    quantile_edges,
    stability_chi2,
    stability_ks,
    stability_psi,
    stability_wasserstein,
)


def test_psi_chi2_benchmark():
    # Yurdakul and Naranjo (2020), Table 2: B = 10, alpha = 0.05, chi2(0.95, 9) = 16.92
    for n, expected in [(100, 0.338), (200, 0.169), (400, 0.085), (1000, 0.034)]:
        assert round(psi_benchmark(n, n, bins=10, alpha=0.05), 3) == expected
    e = np.array([100, 120, 90, 110, 80])
    a = np.array([90, 140, 70, 100, 100])
    pe, pa = e / e.sum(), a / a.sum()
    psi = float(np.sum((pa - pe) * np.log(pa / pe)))
    r = stability_psi(e, a)
    assert r.statistic == pytest.approx(psi)
    scale = 1 / e.sum() + 1 / a.sum()
    assert r.p_value == pytest.approx(stats.chi2.sf(psi / scale, 4))
    assert r.extras["benchmark_05"] == pytest.approx(scale * stats.chi2.ppf(0.95, 4))
    assert r.extras["band"] == ("little change" if psi < 0.10 else "moderate change")
    # an empty bin is floored at half an observation, never an infinite PSI
    z = stability_psi([50, 50, 0], [40, 40, 20])
    assert np.isfinite(z.statistic) and z.extras["empty_bins_floored"] == 1


def test_shift_tests_match_scipy():
    rng = np.random.default_rng(21)
    dev = rng.normal(size=800)
    cur = rng.normal(loc=0.2, scale=1.1, size=600)
    edges = quantile_edges(dev, 10)
    ce, ca = bin_counts(dev, edges), bin_counts(cur, edges)
    assert ce.sum() == 800 and ca.sum() == 600
    chi = stability_chi2(ce, ca)
    ref = stats.chi2_contingency(np.vstack([ce, ca]), correction=False)
    assert chi.statistic == pytest.approx(ref.statistic) and chi.p_value == pytest.approx(ref.pvalue)
    ks = stability_ks(dev, cur)
    kr = stats.ks_2samp(dev, cur)
    assert ks.statistic == pytest.approx(kr.statistic) and ks.p_value == pytest.approx(kr.pvalue)
    assert stability_wasserstein(dev, cur).metric == pytest.approx(stats.wasserstein_distance(dev, cur))
