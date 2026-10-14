"""RV-720 to RV-723: the fit of the Vasicek distribution to a series of yearly loss or default rates (dossier
04 C.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.engines.vasicek import conditional_pd, implied_factor, lhp_pdf, lhp_quantile
from riskvalidation.portfolio.asrf_fit import (
    AsrfFit,
    bootstrap_interval,
    fit_ml,
    fit_moments,
    implied_factors,
    log_likelihood,
    profile_interval,
    profile_likelihood,
)

T_QUASI = 2000
QUASI = np.asarray(lhp_quantile((np.arange(1, T_QUASI + 1) - 0.5) / T_QUASI, 0.02, 0.15))


def _series(t: int = 30, pd: float = 0.02, rho: float = 0.15, seed: int = 1) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal(t)
    return np.asarray(conditional_pd(pd, rho, z))


def test_log_likelihood_golden():
    rates = np.array([0.01, 0.02, 0.05])
    dens = lhp_pdf(rates, 0.02, 0.15)
    assert dens == pytest.approx([34.661219, 18.004184, 3.517955], abs=5e-7)
    assert log_likelihood(rates, 0.02, 0.15) == pytest.approx(7.694105, abs=5e-7)
    assert log_likelihood(rates, 0.02, 0.15) == pytest.approx(float(np.log(dens).sum()), abs=1e-12)
    assert log_likelihood(rates[::-1], 0.02, 0.15) == pytest.approx(log_likelihood(rates, 0.02, 0.15), abs=1e-12)
    for bad in (np.array([0.0, 0.02]), np.array([0.02, 1.0]), np.array([-0.1, 0.02]), np.array([0.02, np.nan]),
                np.array([])):
        with pytest.raises(ValueError):
            log_likelihood(bad, 0.02, 0.15)
    with pytest.raises(ValueError):
        log_likelihood(rates, 0.0, 0.15)


def test_fit_ml_recovers_the_quasi_sample():
    f = fit_ml(QUASI)
    assert isinstance(f, AsrfFit)
    assert f.pd == pytest.approx(0.019995, abs=5e-5) and f.rho == pytest.approx(0.149917, abs=5e-5)
    assert abs(f.pd - 0.02) < 1e-3 and abs(f.rho - 0.15) < 1e-3  # recovers the generating pair
    assert f.converged and f.method == "maximum-likelihood" and f.n_years == T_QUASI and f.n_missing == 0
    assert 0 < f.pd < 1 and 0 < f.rho < 1
    assert f.log_likelihood == pytest.approx(log_likelihood(QUASI, f.pd, f.rho), abs=1e-9)
    g = fit_ml(QUASI)
    assert (g.pd, g.rho, g.log_likelihood) == (f.pd, f.rho, f.log_likelihood)  # deterministic
    # a short series from the model: the maximum is not below its neighbours
    r = _series(40, 0.02, 0.15, seed=3)
    h = fit_ml(r)
    assert h.converged
    best = h.log_likelihood
    for dp in (-0.1, -0.05, 0.0, 0.05, 0.1):
        for dr in (-0.1, -0.05, 0.0, 0.05, 0.1):
            pd, rho = h.pd * (1 + dp), min(0.99, h.rho * (1 + dr))
            assert log_likelihood(r, pd, rho) <= best + 1e-9


def test_fit_ml_missing_years_refusals_and_neighbours():
    r = _series(25, seed=5)
    base = fit_ml(r)
    with_nan = np.insert(r, 7, np.nan)
    f = fit_ml(np.append(with_nan, np.nan))
    assert f.n_missing == 2 and f.n_years == 25
    assert (f.pd, f.rho) == pytest.approx((base.pd, base.rho), rel=1e-9)
    assert fit_ml(r[::-1]).pd == pytest.approx(base.pd, rel=1e-6)  # permutation invariant
    with pytest.raises(ValueError, match="3"):
        fit_ml(np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="3"):
        fit_ml(np.array([0.01, 0.02, np.nan, np.nan]))
    with pytest.raises(ValueError, match="spread"):
        fit_ml(np.full(10, 0.02))
    for bad in (np.array([0.01, 0.02, 0.0]), np.array([0.01, 0.02, 1.0]), np.array([0.01, 0.02, -0.5])):
        with pytest.raises(ValueError):
            fit_ml(bad)
    # 25 neighbouring points of the quasi-sample fit
    q = fit_ml(QUASI)
    for dp in (-2e-4, -1e-4, 0.0, 1e-4, 2e-4):
        for dr in (-2e-3, -1e-3, 0.0, 1e-3, 2e-3):
            assert log_likelihood(QUASI, q.pd + dp, q.rho + dr) <= q.log_likelihood + 1e-9


def test_fit_moments_golden_and_refusal():
    rates = np.array([0.01, 0.02, 0.03])
    f = fit_moments(rates)
    assert f.pd == pytest.approx(0.02, abs=1e-15) and f.rho == pytest.approx(0.026880, abs=5e-7)
    assert float(np.mean(rates**2)) == pytest.approx(0.000467, abs=5e-7)
    assert f.method == "moments" and f.n_years == 3 and f.n_missing == 0 and f.converged
    assert f.log_likelihood == pytest.approx(log_likelihood(rates, f.pd, f.rho), abs=1e-12)
    # the second moment of the fit equals the second moment of the data (the joint default probability)
    from scipy import integrate, stats

    joint, _ = integrate.quad(lambda z: conditional_pd(f.pd, f.rho, z) ** 2 * stats.norm.pdf(z), -12, 12, epsabs=1e-15)
    assert joint == pytest.approx(float(np.mean(rates**2)), abs=1e-10)
    g = fit_moments(np.array([0.01, np.nan, 0.02, 0.03]))
    assert g.n_missing == 1 and g.rho == pytest.approx(f.rho, rel=1e-12)
    with pytest.raises(ValueError, match="dispersion"):
        fit_moments(np.full(5, 0.02))
    with pytest.raises(ValueError):
        fit_moments(np.array([0.01, 0.02]))
    with pytest.raises(ValueError):
        fit_moments(np.array([0.01, 0.02, 1.0]))


def test_profile_likelihood():
    grid = np.round(np.arange(0.02, 0.3001, 0.02), 10)
    ll, pd_at = profile_likelihood(QUASI, grid)
    full = fit_ml(QUASI)
    assert ll.shape == grid.shape == pd_at.shape
    assert ll.max() <= full.log_likelihood + 1e-9
    top = int(np.argmax(ll))
    assert grid[top] in (0.14, 0.16)  # the grid points nearest 0.15
    d = np.diff(ll)
    assert (d[:top] > 0).all() and (d[top:] < 0).all()  # unimodal
    g2 = np.array([0.1, full.rho, 0.2])
    ll2, pd2 = profile_likelihood(QUASI, g2)
    assert ll2[1] == pytest.approx(full.log_likelihood, abs=1e-6) and pd2[1] == pytest.approx(full.pd, abs=1e-5)
    assert pd2[0] != pd2[2]
    with pytest.raises(ValueError):
        profile_likelihood(QUASI, np.array([0.2, 0.1]))
    with pytest.raises(ValueError):
        profile_likelihood(QUASI, np.array([0.1, 1.0]))
    with pytest.raises(ValueError):
        profile_likelihood(QUASI, np.array([]))


def test_profile_interval_cutoff_and_coverage():
    assert math.isclose(2 * 1.920729, 3.841459, abs_tol=2e-6)
    iv = profile_interval(QUASI, 0.95)
    assert set(iv) == {"pd", "rho"}
    assert iv["rho"][0] < 0.15 < iv["rho"][1] and iv["pd"][0] < 0.02 < iv["pd"][1]
    full = fit_ml(QUASI)
    assert iv["rho"][0] < full.rho < iv["rho"][1]
    # at an interior endpoint the profile log-likelihood is at the cutoff max - 1.920729
    ll, _ = profile_likelihood(QUASI, np.array(iv["rho"]))
    assert ll == pytest.approx([full.log_likelihood - 1.920729] * 2, abs=2e-3)
    # a short realistic series: a wider interval that contains the estimate, deterministic
    r = _series(22, 0.02, 0.15, seed=11)
    f = fit_ml(r)
    a = profile_interval(r, 0.95)
    b = profile_interval(r, 0.95)
    assert a == b
    assert a["rho"][0] <= f.rho <= a["rho"][1] and a["pd"][0] <= f.pd <= a["pd"][1]
    assert (a["rho"][1] - a["rho"][0]) > (iv["rho"][1] - iv["rho"][0])
    wide = profile_interval(r, 0.99)
    assert wide["rho"][0] <= a["rho"][0] and wide["rho"][1] >= a["rho"][1]
    with pytest.raises(ValueError):
        profile_interval(r, 1.0)
    with pytest.raises(ValueError):
        profile_interval(r, 0.0)


def test_bootstrap_interval_reproducible():
    r = _series(22, 0.02, 0.15, seed=11)
    a = bootstrap_interval(r, "maximum-likelihood", 100, seed=3)
    b = bootstrap_interval(r, "maximum-likelihood", 100, seed=3)
    c = bootstrap_interval(r, "maximum-likelihood", 100, seed=4)
    assert dict(a) == dict(b) and dict(a) != dict(c)
    assert a["pd"][0] < a["pd"][1] and a["rho"][0] < a["rho"][1]
    assert 0 < a["pd"][0] and a["pd"][1] < 1 and 0 < a["rho"][0] and a["rho"][1] < 1
    assert a.n_dropped >= 0 and a.n_boot == 100
    m = bootstrap_interval(r, "moments", 100, seed=3)
    assert m["rho"][0] < m["rho"][1] and m.n_dropped >= 0
    # no excess dispersion in the sample: every refit of the moments estimator may fail and is counted, never invented
    with pytest.raises(ValueError):
        bootstrap_interval(r, "least-squares", 100, seed=1)
    with pytest.raises(ValueError):
        bootstrap_interval(r, "moments", 99, seed=1)
    with pytest.raises(ValueError):
        bootstrap_interval(r, "moments", 100, seed=1, level=1.0)


def test_implied_factors():
    z = implied_factors(np.array([0.05]), 0.02, 0.15)
    assert z[0] == pytest.approx(-1.387222, abs=5e-7)
    assert conditional_pd(0.02, 0.15, z[0]) == pytest.approx(0.05, abs=1e-12)
    r = np.array([0.01, np.nan, 0.03, 0.08])
    out = implied_factors(r, 0.02, 0.15)
    assert math.isnan(out[1]) and out[0] > out[2] > out[3]  # decreasing in the rate
    assert out[0] == implied_factor(0.01, 0.02, 0.15)
    for bad in (np.array([0.0]), np.array([1.0]), np.array([-0.1, 0.2])):
        with pytest.raises(ValueError):
            implied_factors(bad, 0.02, 0.15)
    with pytest.raises(ValueError):
        implied_factors(np.array([0.1]), 0.0, 0.15)
