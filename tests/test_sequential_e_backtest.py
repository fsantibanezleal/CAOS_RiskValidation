"""U20 (BL-121): the e-backtests of VaR and of expected shortfall of Wang, Wang and Ziegel (2022), arXiv 2209.00991.

Every definition is the paper's. The gates: the e-statistics on hand-computed values and their expectation by quadrature
and by Monte Carlo (Definitions 1 and 2, equations (1) and (3), Theorem 1); the e-process (7) and the betting processes
GREE, GREL and GREM (Section 5.1, equation (15), Appendix A) against loops written from the equations and, for VaR,
against the reference function of the Contraste manuscript; the exact martingale property by full enumeration (Theorem
2); the size at any stopping time and the detection of a planted under-forecast; the edge cases and the record."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from riskvalidation.core.result import Light
from riskvalidation.validation.sequential import (
    EBacktestResult,
    EProcessResult,
    betting_process,
    e_backtest_es,
    e_backtest_var,
    es_backtest_e_statistic,
    var_backtest_e_statistic,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("GREE", "GREL", "GREM")


# ------------------------------------------------------------------------------------------------ the oracles

def wwz_log_e(h: np.ndarray, p: float, gamma: float = 0.5) -> np.ndarray:
    """The reference of the Contraste manuscript `e-process-sequential-monitors`, copied from CAOS_Contraste,
    `manuscripts/e-process-sequential-monitors/experiments/common.py`, function `wwz_log_e` (the VaR hit e-statistic
    `hit / p` with `p` the hit probability, the bet `1 - lam + lam e`, `lam` by the growth-rate betting process of the
    paper's equation (15) on all past days, capped at gamma, no bet on the first day). Returns `log M_t` per series
    and day."""
    n = h.shape[1]
    x = h.astype(float)
    hits_before = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(x, axis=1)[:, :-1]], axis=1)
    t_before = np.arange(0, n, dtype=float)[None, :]
    num = hits_before / p - t_before
    den = hits_before * (1.0 / p - 1.0) ** 2 + (t_before - hits_before)
    lam = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), 0.0, gamma)
    return np.cumsum(np.log1p(lam * (x / p - 1.0)), axis=1)


def e_var(x: float, r: float, alpha: float) -> float:
    """Equation (1) of the paper for one day, `1{x > r} / (1 - p)` with `1 - p = alpha`."""
    return (1.0 if x > r else 0.0) / alpha


def e_es(x: float, r: float, z: float, alpha: float) -> float:
    """Equation (3) of the paper for one day, with the conventions 0/0 = 1 and 1/0 = infinity."""
    num, den = max(x - z, 0.0), alpha * (r - z)
    if den == 0.0:
        return 1.0 if num == 0.0 else math.inf
    return num / den


def loop_lambda(loss, r, z, alpha, kind, method, gamma=0.5, window=None):
    """The bets of GREE or GREL by the closed form (15), written as a loop over days and over each day's past sample:
    `0 v (sum (e - 1)) / (sum (e - 1)^2) ^ gamma`, the first day and an empty or neutral sample betting 0."""
    n = len(loss)
    out = np.zeros(n)
    for t in range(n):
        lo = 0 if window is None else max(0, t - window)
        idx = range(lo, t)
        if not idx:
            continue
        es = []
        for s in idx:
            rr, zz = (r[s], z[s]) if method == "GREE" else (r[t], z[t])
            es.append(e_var(loss[s], rr, alpha) if kind == "var" else e_es(loss[s], rr, zz, alpha))
        if any(math.isinf(e) for e in es):
            out[t] = gamma
            continue
        s1 = sum(e - 1.0 for e in es)
        s2 = sum((e - 1.0) ** 2 for e in es)
        out[t] = 0.0 if s2 == 0.0 else min(max(s1 / s2, 0.0), gamma)
    return out


def loop_log_e(loss, r, z, alpha, kind, lam):
    """Equation (7) as a loop."""
    total, path = 0.0, []
    for t in range(len(loss)):
        x = e_var(loss[t], r[t], alpha) if kind == "var" else e_es(loss[t], r[t], z[t], alpha)
        if lam[t] > 0.0:
            total += math.inf if math.isinf(x) else math.log1p(lam[t] * (x - 1.0))
        path.append(total)
    return np.array(path)


def normal_var_es(alpha: float) -> tuple[float, float]:
    """The VaR and the ES at level 1 - alpha of a standard normal loss."""
    z = float(stats.norm.ppf(1.0 - alpha))
    return z, float(stats.norm.pdf(z) / alpha)


def _series(seed: int, n: int, alpha: float = 0.05, scale_var: float = 1.0, scale_es: float = 1.0, varying=False):
    """Standard normal losses (times a known volatility when `varying`) and the true VaR and ES forecasts scaled."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261116]))
    sigma = 1.0 + 0.5 * np.sin(np.arange(n) / 7.0) if varying else np.ones(n)
    loss = sigma * rng.standard_normal(n)
    z, es = normal_var_es(alpha)
    return loss, scale_var * z * sigma, scale_es * es * sigma


# ------------------------------------------------------------------------------------- the e-statistics (1), (3)

def test_var_e_statistic_values_and_refusals():
    e = var_backtest_e_statistic([0.5, 2.0, -1.0, 3.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0], 0.25)
    np.testing.assert_array_equal(e, [0.0, 4.0, 0.0, 4.0, 0.0])  # a loss equal to the VaR is not a hit: x > r
    assert var_backtest_e_statistic([2.0, 0.0], [1.0, 1.0], 0.5).tolist() == [2.0, 0.0]
    with pytest.raises(ValueError, match="alpha"):
        var_backtest_e_statistic([1.0], [0.5], 1.0)
    with pytest.raises(ValueError, match="alpha"):
        var_backtest_e_statistic([1.0], [0.5], 0.0)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="loss"):
            var_backtest_e_statistic([1.0, bad], [0.5, 0.5], 0.1)
        with pytest.raises(ValueError, match="var"):
            var_backtest_e_statistic([1.0, 1.0], [0.5, bad], 0.1)
    with pytest.raises(ValueError, match="same length"):
        var_backtest_e_statistic([1.0, 2.0], [0.5], 0.1)
    with pytest.raises(ValueError, match="one-dimensional"):
        var_backtest_e_statistic([[1.0, 2.0]], [[0.5, 0.5]], 0.1)


def test_es_e_statistic_values_conventions_and_refusals():
    # alpha = 0.1, VaR z = 1, ES r = 3: e = (x - 1)+ / (0.1 * 2)
    e = es_backtest_e_statistic([0.0, 1.0, 1.5, 4.0], [1.0] * 4, [3.0] * 4, 0.1)
    np.testing.assert_allclose(e, [0.0, 0.0, 2.5, 15.0], rtol=0, atol=1e-14)
    # es == var: 0/0 = 1 when the loss does not exceed the VaR, 1/0 = infinity when it does (the paper, Section 3.3)
    e0 = es_backtest_e_statistic([0.5, 1.0, 1.5], [1.0] * 3, [1.0] * 3, 0.1)
    assert e0[0] == 1.0 and e0[1] == 1.0 and math.isinf(e0[2])
    with pytest.raises(ValueError, match="es"):
        es_backtest_e_statistic([1.0], [2.0], [1.0], 0.1)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="es"):
            es_backtest_e_statistic([1.0], [0.5], [bad], 0.1)
    with pytest.raises(ValueError, match="same length"):
        es_backtest_e_statistic([1.0, 2.0], [0.5, 0.5], [1.0], 0.1)
    with pytest.raises(ValueError, match="alpha"):
        es_backtest_e_statistic([1.0], [0.5], [1.0], 1.5)


def _quad_mean_es(alpha: float, scale_var: float, scale_es: float) -> float:
    """`E[e^ES(X, r, z)]` for a standard normal X, by quadrature of the definition (3)."""
    z0, es0 = normal_var_es(alpha)
    z, r = scale_var * z0, scale_es * es0
    value, _ = integrate.quad(lambda x: (x - z) * stats.norm.pdf(x), z, np.inf, epsabs=1e-13, epsrel=1e-13)
    return value / (alpha * (r - z))


def test_expectations_by_quadrature_and_by_monte_carlo():
    for alpha in (0.1, 0.05, 0.025, 0.01):
        assert _quad_mean_es(alpha, 1.0, 1.0) == pytest.approx(1.0, abs=1e-9)  # equality in Theorem 1
        assert _quad_mean_es(alpha, 1.0, 1.2) < 1.0 - 1e-3  # an over-forecast of the ES
        assert _quad_mean_es(alpha, 1.0, 0.9) > 1.0 + 1e-3  # an under-forecast of the ES (Definition 2)
        z, _ = normal_var_es(alpha)
        assert (1.0 - stats.norm.cdf(z)) / alpha == pytest.approx(1.0, abs=1e-12)  # the VaR e-statistic, Example 3
        assert (1.0 - stats.norm.cdf(0.9 * z)) / alpha > 1.0  # an under-forecast of the VaR
        assert (1.0 - stats.norm.cdf(1.1 * z)) / alpha < 1.0
    alpha, n = 0.05, 200_000
    z, es = normal_var_es(alpha)
    x = np.random.default_rng(np.random.SeedSequence([1, 20261116])).standard_normal(n)
    for e, sd in ((var_backtest_e_statistic(x, np.full(n, z), alpha), math.sqrt((1 - alpha) / alpha)),
                  (es_backtest_e_statistic(x, np.full(n, z), np.full(n, es), alpha), None)):
        se = (sd if sd is not None else float(e.std())) / math.sqrt(n)
        assert abs(float(e.mean()) - 1.0) <= 4.0 * se, (float(e.mean()), se)


@pytest.mark.sizepower
def test_expectation_monte_carlo_large():
    alpha, chunk, reps = 0.025, 1_000_000, 20
    z, es = normal_var_es(alpha)
    rng = np.random.default_rng(np.random.SeedSequence([2, 20261116]))
    sums = {"var": [0.0, 0.0], "es": [0.0, 0.0]}
    for _ in range(reps):
        x = rng.standard_normal(chunk)
        for key, e in (("var", var_backtest_e_statistic(x, np.full(chunk, z), alpha)),
                       ("es", es_backtest_e_statistic(x, np.full(chunk, z), np.full(chunk, es), alpha))):
            sums[key][0] += float(e.sum())
            sums[key][1] += float((e**2).sum())
    n = chunk * reps
    for key, (s, s2) in sums.items():
        mean = s / n
        se = math.sqrt((s2 / n - mean**2) / n)
        assert abs(mean - 1.0) <= 4.0 * se, (key, mean, se)


# ------------------------------------------------------------------------------------- the e-process (7)

def test_the_e_process_on_a_hand_computed_series():
    # alpha = 0.2: e = 5 on a hit and 0 otherwise. Bets by (15) on the past days: day 1 none; day 2 one hit in one
    # day: 4 / 16 = 1/4; day 3 (hit, miss): 3 / 17; day 4 (hit, miss, hit): 7 / 33; day 5: 6 / 34.
    loss = np.array([2.0, 0.0, 2.0, 0.0, 0.0])
    var = np.ones(5)
    lam = [Fraction(0), Fraction(1, 4), Fraction(3, 17), Fraction(7, 33), Fraction(6, 34)]
    factors = [Fraction(1), 1 - lam[1], 1 + lam[2] * 4, 1 - lam[3], 1 - lam[4]]
    expected = np.cumsum([math.log(f.numerator / f.denominator) for f in factors])
    res = e_backtest_var(loss, var, 0.2, method="GREE")
    np.testing.assert_allclose(res.betting, [float(x) for x in lam], rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.log_e, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(res.e_statistic, [5.0, 0.0, 5.0, 0.0, 0.0])
    assert res.betting[0] == 0.0 and np.all((res.betting >= 0.0) & (res.betting <= 0.5))
    # the cap: a hit every day wants the bet alpha / (1 - alpha) = 1/4 at alpha 0.2; a cap of 0.1 binds
    capped = e_backtest_var(np.full(6, 2.0), var[:1].repeat(6), 0.2, method="GREE", gamma=0.1)
    np.testing.assert_allclose(capped.betting, [0.0] + [0.1] * 5, rtol=0, atol=1e-15)
    assert betting_process(loss, var, 0.2, method="GREE").tolist() == res.betting.tolist()


# ----------------------------------------------------------------------- GREE and GREL: closed form (15), Appendix A

@pytest.mark.parametrize("kind", ["var", "es"])
@pytest.mark.parametrize("method", ["GREE", "GREL"])
@pytest.mark.parametrize("window", [None, 25])
def test_gree_and_grel_equal_a_loop_from_the_equations(kind, method, window):
    n, alpha = 140, 0.05
    # an under-reporting forecaster with a time-varying forecast and a tail that sometimes sits below the VaR
    loss, var, es = _series(3, n, alpha, scale_var=0.85, scale_es=0.9, varying=True)
    r = var if kind == "var" else es
    res = (e_backtest_var(loss, var, alpha, method=method, window=window) if kind == "var"
           else e_backtest_es(loss, var, es, alpha, method=method, window=window))
    lam = loop_lambda(loss, r, var, alpha, kind, method, window=window)
    np.testing.assert_allclose(res.betting, lam, rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.log_e, loop_log_e(loss, r, var, alpha, kind, lam), rtol=0, atol=1e-11)
    assert res.betting.max() > 0.0  # the series makes the bet positive, so the test is not vacuous


def test_gree_equals_grel_for_constant_forecasts():
    n, alpha = 200, 0.05
    loss, var, es = _series(4, n, alpha, scale_var=0.9, scale_es=0.9)
    for res_e, res_l in (
        (e_backtest_var(loss, var, alpha, method="GREE"), e_backtest_var(loss, var, alpha, method="GREL")),
        (e_backtest_es(loss, var, es, alpha, method="GREE"), e_backtest_es(loss, var, es, alpha, method="GREL")),
        (e_backtest_var(loss, var, alpha, method="GREE", window=30),
         e_backtest_var(loss, var, alpha, method="GREL", window=30)),
    ):
        np.testing.assert_allclose(res_e.betting, res_l.betting, rtol=0, atol=1e-15)
        np.testing.assert_allclose(res_e.log_e, res_l.log_e, rtol=0, atol=1e-13)
    # with forecasts that move, the two differ (the paper's comparison of GREE and GREL, Example 7)
    loss2, var2, es2 = _series(4, n, alpha, scale_var=0.9, scale_es=0.9, varying=True)
    a = e_backtest_es(loss2, var2, es2, alpha, method="GREE").betting
    b = e_backtest_es(loss2, var2, es2, alpha, method="GREL").betting
    assert np.abs(a - b).max() > 1e-3


@pytest.mark.parametrize("p_hit", [0.01, 0.03, 0.08])
@pytest.mark.parametrize("alpha", [0.01, 0.025])
def test_the_var_e_process_equals_the_manuscripts_reference(p_hit, alpha):
    reps, n = 6, 750
    rng = np.random.default_rng(np.random.SeedSequence([int(p_hit * 1000), int(alpha * 1000), 20261116]))
    hits = rng.random((reps, n)) < p_hit
    ref = wwz_log_e(hits, alpha)
    for h, ref_h in zip(hits, ref, strict=True):
        loss = np.where(h, 1.0, -1.0)  # a hit is loss > var with var = 0
        var = np.zeros(n)
        for method in ("GREE", "GREL"):  # constant forecasts: the two coincide
            res = e_backtest_var(loss, var, alpha, method=method)
            np.testing.assert_allclose(res.log_e, ref_h, rtol=0, atol=1e-12)
        # GREE reads only the hit flags, so a VaR that moves with the same flags gives the same path
        var_moving = np.linspace(-0.5, 0.5, n)
        loss_moving = np.where(h, var_moving + 1.0, var_moving - 1.0)
        res = e_backtest_var(loss_moving, var_moving, alpha, method="GREE")
        np.testing.assert_allclose(res.log_e, ref_h, rtol=0, atol=1e-12)
    # Appendix A, VaR: lambda = (1 - p)((t - 1) p - k) / ((t - 1) p^2 + (1 - 2p) k) with k the non-hits of the past
    h = hits[0]
    res = e_backtest_var(np.where(h, 1.0, -1.0), np.zeros(n), alpha, method="GREE")
    p = 1.0 - alpha
    t1 = np.arange(n, dtype=float)  # t - 1 past days
    k = np.concatenate([[0.0], np.cumsum(~h)[:-1]])
    num, den = (1.0 - p) * (t1 * p - k), t1 * p**2 + (1.0 - 2.0 * p) * k
    lam = np.clip(np.divide(num, den, out=np.zeros(n), where=den != 0.0), 0.0, 0.5)
    np.testing.assert_allclose(res.betting, lam, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------ the exact solver

def growth(lam: float, sample: list) -> float:
    """The empirical growth rate `mean log(1 - lam + lam e)` of a sample of e-statistics (equations (13) and (14))."""
    return float(np.mean(np.log1p(lam * (np.array(sample) - 1.0))))


def test_the_exact_solver_maximises_the_growth_rate():
    n, alpha = 160, 0.05
    # a hit series: the maximiser of h log(1 + lam (1 - alpha) / alpha) + (1 - h) log(1 - lam) is
    # (h - alpha) / (1 - alpha)
    rng = np.random.default_rng(np.random.SeedSequence([5, 20261116]))
    hits = rng.random(n) < 0.12
    loss, var = np.where(hits, 1.0, -1.0), np.zeros(n)
    res = e_backtest_var(loss, var, alpha, method="GREE", solver="exact")
    h = np.concatenate([[0.0], np.cumsum(hits)[:-1]]) / np.maximum(np.arange(n), 1)
    closed = np.clip((h - alpha) / (1.0 - alpha), 0.0, 0.5)
    closed[0] = 0.0
    np.testing.assert_allclose(res.betting, closed, rtol=0, atol=1e-9)
    # ES series with moving forecasts: a bounded scalar minimiser of the negative empirical growth rate
    loss, var, es = _series(6, 90, alpha, scale_var=0.9, scale_es=0.9, varying=True)
    for method in ("GREE", "GREL"):
        res = e_backtest_es(loss, var, es, alpha, method=method, solver="exact")
        for t in (10, 40, 89):
            sample = [e_es(loss[s], es[s] if method == "GREE" else es[t], var[s] if method == "GREE" else var[t], alpha)
                      for s in range(t)]
            sol = optimize.minimize_scalar(lambda lam, sample=sample: -growth(lam, sample), bounds=(0.0, 0.5),
                                           method="bounded", options={"xatol": 1e-13})
            assert res.betting[t] == pytest.approx(sol.x, abs=1e-7), (method, t)
            taylor = e_backtest_es(loss, var, es, alpha, method=method).betting[t]
            assert growth(res.betting[t], sample) >= growth(taylor, sample) - 1e-12
    # a neutral or adverse sample bets nothing, and an infinite e-statistic in the sample bets the cap
    flat = e_backtest_var(np.full(8, -1.0), np.zeros(8), alpha, method="GREE", solver="exact")
    assert flat.betting.tolist() == [0.0] * 8
    inf_case = e_backtest_es([0.5, 1.5, 0.5, 0.5], [1.0] * 4, [1.0] * 4, alpha, method="GREE", solver="exact")
    assert inf_case.betting.tolist() == [0.0, 0.0, 0.5, 0.5]


# ----------------------------------------------------------------------------------------------------- GREM

@pytest.mark.parametrize("solver", ["taylor", "exact"])
def test_grem_is_the_wealth_mixture_with_the_paper_betting_process(solver):
    n, alpha = 120, 0.05
    loss, var, es = _series(7, n, alpha, scale_var=0.85, scale_es=0.88, varying=True)
    gree = e_backtest_es(loss, var, es, alpha, method="GREE", solver=solver)
    grel = e_backtest_es(loss, var, es, alpha, method="GREL", solver=solver)
    grem = e_backtest_es(loss, var, es, alpha, method="GREM", solver=solver)
    m_e, m_l = np.exp(gree.log_e), np.exp(grel.log_e)
    np.testing.assert_allclose(np.exp(grem.log_e), (m_e + m_l) / 2.0, rtol=1e-12)
    prev_e, prev_l = np.concatenate([[1.0], m_e[:-1]]), np.concatenate([[1.0], m_l[:-1]])
    lam = (prev_e * gree.betting + prev_l * grel.betting) / (prev_e + prev_l)
    np.testing.assert_allclose(grem.betting, lam, rtol=0, atol=1e-13)
    # the product (7) of the mixture betting process is the mixture path
    x = grem.e_statistic
    np.testing.assert_allclose(np.cumsum(np.log1p(grem.betting * (x - 1.0))), grem.log_e, rtol=0, atol=1e-10)
    assert set(grem.component_log_e) == {"GREE", "GREL"}
    np.testing.assert_array_equal(grem.component_log_e["GREE"], gree.log_e)
    assert gree.component_log_e == {} and grel.component_log_e == {}
    assert e_backtest_var(loss, var, alpha).extras["method"] == "GREM"  # the paper's recommended default


# ------------------------------------------------------------------------------------------ the rolling window

def test_the_rolling_window():
    n, alpha = 150, 0.05
    # an over-forecast first half then an under-forecast second half: the window forgets the first (Remark 8)
    loss, var, _ = _series(8, n, alpha)
    var = np.where(np.arange(n) < 75, var * 1.4, var * 0.7)
    for method in ("GREE", "GREL"):
        res = e_backtest_var(loss, var, alpha, method=method, window=20)
        np.testing.assert_allclose(res.betting, loop_lambda(loss, var, var, alpha, "var", method, window=20),
                                   rtol=0, atol=1e-13)
        assert res.extras["window"] == 20
        full = e_backtest_var(loss, var, alpha, method=method)
        for big in (n, n + 50):
            same = e_backtest_var(loss, var, alpha, method=method, window=big)
            np.testing.assert_allclose(same.betting, full.betting, rtol=0, atol=1e-15)
    # Remark 8 by hand: no hit for 100 days, then a hit every day, alpha = 0.2. On day 111 (110 days before it) the
    # whole past has 10 hits (e = 5, e - 1 = 4) and 100 non-hits (e - 1 = -1): S1 = 40 - 100 < 0 and the bet is 0; the
    # last 20 days have 10 non-hits and 10 hits: S1 = 40 - 10 = 30, S2 = 10 * 16 + 10 = 170, the bet is 30 / 170
    swing = np.where(np.arange(150) < 100, -1.0, 1.0)
    for method in ("GREE", "GREL"):
        assert e_backtest_var(swing, np.zeros(150), 0.2, method=method).betting[110] == 0.0
        windowed = e_backtest_var(swing, np.zeros(150), 0.2, method=method, window=20)
        assert windowed.betting[110] == pytest.approx(30.0 / 170.0, abs=1e-15)
    one = e_backtest_var(loss, var, alpha, method="GREE", window=1)
    np.testing.assert_allclose(one.betting, loop_lambda(loss, var, var, alpha, "var", "GREE", window=1), atol=1e-15)


# ------------------------------------------------------------------------- the martingale property, exactly

def _enumerate_var(n_days: int, alpha: float, method, solver, window, gamma, scale):
    """`E[M_T]` of the VaR e-process by summing over all 2^n hit sequences with their null probabilities."""
    total = 0.0
    for bits in itertools.product([0, 1], repeat=n_days):
        k = sum(bits)
        prob = alpha**k * (1.0 - alpha) ** (n_days - k)
        loss = np.where(np.array(bits, dtype=bool), 2.0, 0.0)
        res = e_backtest_var(loss, np.full(n_days, scale), alpha, method=method, solver=solver, window=window,
                             gamma=gamma)
        total += prob * math.exp(res.log_e[-1])
    return total


def test_the_e_process_has_mean_one_by_enumeration():
    for method, solver, window, gamma in (("GREE", "taylor", None, 0.5), ("GREL", "taylor", None, 0.5),
                                          ("GREM", "taylor", None, 0.5), ("GREM", "taylor", 4, 0.3),
                                          ("GREM", "exact", None, 0.5), ("GREE", "exact", 5, 0.9)):
        n_days = 8 if solver == "exact" else 10
        assert _enumerate_var(n_days, 0.2, method, solver, window, gamma, 1.0) == pytest.approx(1.0, abs=1e-12), \
            (method, solver, window)
    # an over-forecast (a VaR above the true quantile: hit probability below alpha) is a strict supermartingale
    over = 0.0
    p_hit = 0.1
    for bits in itertools.product([0, 1], repeat=10):
        k = sum(bits)
        prob = p_hit**k * (1.0 - p_hit) ** (10 - k)
        res = e_backtest_var(np.where(np.array(bits, dtype=bool), 2.0, 0.0), np.ones(10), 0.2)
        over += prob * math.exp(res.log_e[-1])
    assert over < 1.0 - 1e-3
    # ES: the three-point loss law P(0) = 0.7, P(1) = 0.2, P(3) = 0.1 has VaR_0.9 = 1 and ES_0.9 = 3 (alpha = 0.1),
    # so e = (x - 1)+ / 0.2 has mean 0.1 * 10 = 1 exactly
    values, probs = np.array([0.0, 1.0, 3.0]), np.array([0.7, 0.2, 0.1])
    assert float(np.dot(probs, es_backtest_e_statistic(values, np.ones(3), np.full(3, 3.0), 0.1))) == pytest.approx(1.0)
    for method, window in (("GREE", None), ("GREL", None), ("GREM", None), ("GREM", 3)):
        total = 0.0
        for idx in itertools.product(range(3), repeat=7):
            prob = float(np.prod(probs[list(idx)]))
            res = e_backtest_es(values[list(idx)], np.ones(7), np.full(7, 3.0), 0.1, method=method, window=window)
            total += prob * math.exp(res.log_e[-1])
        assert total == pytest.approx(1.0, abs=1e-12), (method, window)
    # forecasts that move with the day (GREL then differs from GREE) keep the mean at 1: here the scale cycles 1, 2, 3
    total = 0.0
    scale = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    for idx in itertools.product(range(3), repeat=7):
        prob = float(np.prod(probs[list(idx)]))
        res = e_backtest_es(values[list(idx)] * scale, scale, 3.0 * scale, 0.1, method="GREM")
        total += prob * math.exp(res.log_e[-1])
    assert total == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------------------- size and power by simulation

def _null_rejection_shares(reps: int, n: int, alpha: float, kind: str, varying: bool, seed: int, levels):
    ever = {lv: 0 for lv in levels}
    for i in range(reps):
        loss, var, es = _series(seed * 100_000 + i, n, alpha, varying=varying)
        res = e_backtest_var(loss, var, alpha) if kind == "var" else e_backtest_es(loss, var, es, alpha)
        sup = float(res.log_e.max())
        for lv in levels:
            ever[lv] += sup >= -math.log(lv)
    return {lv: c / reps for lv, c in ever.items()}


def _assert_size(shares, reps):
    for lv, share in shares.items():
        bound = lv + 3.0 * math.sqrt(lv * (1.0 - lv) / reps)
        assert share <= bound, (lv, share, bound)


def test_size_at_any_stopping_time():
    levels = (0.5, 0.2, 0.1, 0.05)
    for kind in ("var", "es"):
        for varying, n in ((False, 250), (True, 150)):
            _assert_size(_null_rejection_shares(400, n, 0.05, kind, varying, 1, levels), 400)


@pytest.mark.sizepower
def test_size_at_any_stopping_time_large():
    levels = (0.5, 0.2, 0.1, 0.05)
    for kind in ("var", "es"):
        _assert_size(_null_rejection_shares(4000, 1000, 0.025, kind, False, 2, levels), 4000)
        _assert_size(_null_rejection_shares(4000, 250, 0.05, kind, True, 3, levels), 4000)


def test_a_planted_under_forecast_is_detected_and_an_over_forecast_is_not():
    reps, n, alpha = 100, 1500, 0.05
    detect_var = detect_es = over_var = over_es = prop1 = 0
    for i in range(reps):
        loss, var, es = _series(9000 + i, n, alpha)
        detect_var += e_backtest_var(loss, 0.7 * var, alpha).rejected
        detect_es += e_backtest_es(loss, 0.7 * var, 0.7 * es, alpha).rejected
        over_var += e_backtest_var(loss, 1.3 * var, alpha).rejected
        over_es += e_backtest_es(loss, 1.3 * var, 1.3 * es, alpha).rejected
        prop1 += e_backtest_es(loss, 1.2 * var, 1.5 * es, alpha, alpha_level=0.1).rejected
    assert detect_var >= 90 and detect_es >= 90, (detect_var, detect_es)
    assert over_var == 0 and over_es == 0, (over_var, over_es)
    assert prop1 == 0  # both forecasts above the truth by a factor (Proposition 1, Example 6 (ii))


# -------------------------------------------------------------------------------------- edge cases and refusals

def test_edge_cases():
    alpha = 0.05
    empty = e_backtest_var([], [], alpha)
    assert empty.n == 0 and empty.log_e.size == 0 and empty.betting.size == 0 and empty.e_statistic.size == 0
    assert empty.rejection_time is None and empty.to_test_result().light is Light.NOT_EVALUATED
    assert e_backtest_es([], [], [], alpha).n == 0
    one = e_backtest_var([5.0], [1.0], alpha)
    assert one.log_e.tolist() == [0.0] and one.betting.tolist() == [0.0] and one.e_statistic.tolist() == [20.0]
    assert e_backtest_es([5.0], [1.0], [2.0], alpha).log_e.tolist() == [0.0]
    for method in METHODS:  # no hit at all: every bet is 0 and M stays 1 (Remark 8)
        res = e_backtest_var(np.full(50, -1.0), np.zeros(50), alpha, method=method)
        assert res.log_e.tolist() == [0.0] * 50 and not res.betting.any()
    for method in ("GREE", "GREL"):  # a hit every day: lambda = alpha / (1 - alpha), M_t = 2^(t - 1)
        res = e_backtest_var(np.full(30, 1.0), np.zeros(30), alpha, method=method)
        np.testing.assert_allclose(res.betting[1:], alpha / (1.0 - alpha), rtol=1e-13)
        np.testing.assert_allclose(res.log_e, np.arange(30) * math.log(2.0), rtol=0, atol=1e-12)
    # an ES forecast equal to the VaR forecast: e = 1 below the VaR, infinity above it. An infinite e-value that was
    # not bet on (lambda = 0) changes nothing; the infinite value in the past sample makes the later bets the cap
    res = e_backtest_es([0.5, 0.2, 1.5, 0.3, 0.2], [1.0] * 5, [1.0] * 5, alpha, method="GREE")
    assert res.e_statistic[0] == 1.0 and math.isinf(res.e_statistic[2])
    assert res.betting.tolist() == [0.0, 0.0, 0.0, 0.5, 0.5]
    assert res.log_e.tolist() == [0.0] * 5 and not res.rejected
    # a bet already on when the infinite e-value arrives: it is paid in full, the e-process is infinite and stays so
    res = e_backtest_es([0.5, 1.5, 1.5, 0.2], [1.0] * 4, [1.0] * 4, alpha, method="GREE")
    assert res.betting.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert math.isinf(res.log_e[2]) and math.isinf(res.log_e[3])
    assert res.rejected and res.rejection_time == 3 and res.anytime_p_value == 0.0
    tr = res.to_test_result()
    assert tr.light is Light.RED and tr.p_value == 0.0


def test_refusals():
    ok = dict(loss=[1.0, 2.0], var=[0.5, 0.5], alpha=0.1)
    with pytest.raises(ValueError, match="gamma"):
        e_backtest_var(**ok, gamma=1.0)
    with pytest.raises(ValueError, match="gamma"):
        e_backtest_var(**ok, gamma=0.0)
    with pytest.raises(ValueError, match="alpha_level"):
        e_backtest_var(**ok, alpha_level=1.0)
    with pytest.raises(ValueError, match="alpha_level"):
        e_backtest_var(**ok, alpha_level=0.0)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="window"):
            e_backtest_var(**ok, window=bad)
    with pytest.raises(ValueError, match="method"):
        e_backtest_var(**ok, method="GRO")
    with pytest.raises(ValueError, match="solver"):
        e_backtest_var(**ok, solver="newton")
    with pytest.raises(ValueError, match="loss"):
        e_backtest_var([1.0, float("nan")], [0.5, 0.5], 0.1)
    with pytest.raises(ValueError, match="es"):
        e_backtest_es([1.0, 2.0], [0.5, 0.9], [0.6, 0.8], 0.1)
    with pytest.raises(ValueError, match="alpha"):
        betting_process([1.0], [0.5], 1.2)
    with pytest.raises(ValueError, match="same length"):
        betting_process([1.0, 2.0], [0.5, 0.5], 0.1, es=[1.0])


# ------------------------------------------------------------------------------------------------- the record

def test_the_record_and_the_test_result():
    loss, var, es = _series(11, 300, 0.05, scale_var=0.7, scale_es=0.7)
    res = e_backtest_es(loss, var, es, 0.05, method="GREM")
    assert isinstance(res, EBacktestResult) and isinstance(res, EProcessResult)
    assert res.monitor_id == "sequential.e_backtest_es"
    for arr in (res.log_e, res.e_statistic, res.betting, res.component_log_e["GREE"]):
        assert not arr.flags.writeable
    with pytest.raises(AttributeError):
        res.alpha_level = 0.1  # frozen
    again = e_backtest_es(loss, var, es, 0.05, method="GREM")
    assert again.inputs_hash == res.inputs_hash and again.log_e.tolist() == res.log_e.tolist()
    other = e_backtest_es(loss, var, es, 0.05, method="GREL")
    assert other.inputs_hash != res.inputs_hash
    assert res.extras["alpha"] == 0.05 and res.extras["gamma"] == 0.5 and res.extras["solver"] == "taylor"
    assert res.extras["window"] is None and res.extras["measure"] == "es" and res.extras["n_hits"] >= 0
    assert "arXiv 2209.00991" in res.reference
    tr = res.to_test_result(model_id="m", segment="desk")
    assert tr.test_id == "sequential.e_backtest_es" and tr.n == 300
    assert tr.light is (Light.RED if res.rejected else Light.GREEN)
    assert res.rejected  # a 30% under-forecast over 300 days at alpha 5% crosses 20
    var_res = e_backtest_var(loss, var, 0.05)
    assert var_res.monitor_id == "sequential.e_backtest_var" and var_res.extras["measure"] == "var"
    assert var_res.to_test_result().test_id == "sequential.e_backtest_var"
    from riskvalidation.core.registry import catalogue

    ids = {s.test_id: s for s in catalogue()}
    for tid in ("sequential.e_backtest_var", "sequential.e_backtest_es"):
        assert ids[tid].family == "sequential" and ids[tid].decision == "metric" and "2209.00991" in ids[tid].reference
    green = e_backtest_var(*_series(12, 300, 0.05)[:2], 0.05)
    assert not green.rejected and green.to_test_result().light is Light.GREEN


def test_the_page_carries_the_equations_and_the_choices():
    page = (ROOT / "docs" / "sequential" / "02_e-backtest.md").read_text(encoding="utf-8")
    assert page.count("$$") >= 12, "the page carries its display equations"
    for needle in ("equation (1)", "equation (3)", "Theorem 1", "Theorem 2", "equation (7)", "equation (15)", "GREE",
                   "GREL", "GREM", "arXiv 2209.00991", "`sequential.e_backtest_var`", "`sequential.e_backtest_es`",
                   "gamma", "window", "Remark 7"):
        assert needle in page, needle
    index = (ROOT / "docs" / "sequential" / "README.md").read_text(encoding="utf-8")
    assert "(02_e-backtest.md)" in index
