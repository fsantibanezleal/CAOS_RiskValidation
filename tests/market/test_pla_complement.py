"""BS-13: the scale-and-level complement to the FRTB P&L attribution test (`riskvalidation.market.pla_complement`).

Known truth: the Mincer-Zarnowitz Wald statistic is checked against an independent sandwich computation, the Spearman
statistics against `scipy.stats.spearmanr`, the calibration against the size it promises, and the defining behaviour (a
mis-scaled model is seen by the Wald statistic, a volatility that lags is seen by the rolling-mean Spearman statistic,
the null is not flagged above its size) on a generator with a known truth."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from pla_complement_sim import pair
from scipy import stats

from riskvalidation.market import frtb
from riskvalidation.market import pla_complement as pc


def _null_pairs(start: int, count: int):
    return [pair(start + i) for i in range(count)]


def _replace(obj, **kw):
    return obj.__class__(**{**obj.__dict__, **kw})


def test_module_is_public_and_documented():
    mod = importlib.import_module("riskvalidation.market.pla_complement")
    assert sorted(mod.__all__) == sorted(set(mod.__all__))
    for name in mod.__all__:
        assert getattr(mod, name).__doc__, name
    from riskvalidation import market
    assert "pla_complement" in market.__all__


def test_mz_wald_matches_independent_sandwich():
    h, r = pair(1, lag=20)
    res = pc.pla_mz_wald(h, r)
    x = np.column_stack([np.ones(250), r])
    beta, *_ = np.linalg.lstsq(x, h, rcond=None)
    e = h - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    lev = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    xs = x * (e / (1 - lev))[:, None]
    cov = xtx_inv @ (xs.T @ xs) @ xtx_inv
    d = beta - np.array([0.0, 1.0])
    wald = float(d @ np.linalg.solve(cov, d))
    assert res.intercept == pytest.approx(beta[0], rel=1e-9, abs=1e-12)
    assert res.slope == pytest.approx(beta[1], rel=1e-9)
    assert res.wald == pytest.approx(wald, rel=1e-8)
    assert res.p_value == pytest.approx(stats.chi2.sf(wald, 2), rel=1e-6, abs=1e-12)


def test_mz_wald_sees_scale_and_level():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(250) * 0.01
    noise = rng.standard_normal(250) * 0.001
    exact = pc.pla_mz_wald(r + noise, r)
    assert exact.wald < 12 and abs(exact.slope - 1.0) < 0.05
    scaled = pc.pla_mz_wald(2.0 * r + noise, r)
    assert scaled.wald > 1000 and scaled.slope == pytest.approx(2.0, abs=0.05)
    shifted = pc.pla_mz_wald(r + noise + 0.002, r)
    assert shifted.wald > 100 and shifted.intercept == pytest.approx(0.002, abs=3e-4)


def test_abs_spearman_matches_scipy():
    h, r = pair(2)
    assert pc.pla_abs_spearman(h, r) == pytest.approx(stats.spearmanr(np.abs(h), np.abs(r))[0], abs=1e-12)


def test_rolling_abs_spearman_matches_scipy_and_window():
    h, r = pair(2, lag=10)
    w = 20
    k = np.ones(w) / w
    hm, rm = np.convolve(np.abs(h), k, mode="valid"), np.convolve(np.abs(r), k, mode="valid")
    assert hm.size == 250 - w + 1
    assert pc.pla_rolling_abs_spearman(h, r, window=w) == pytest.approx(stats.spearmanr(hm, rm)[0], abs=1e-12)
    assert pc.pla_rolling_abs_spearman(h, r, window=5) != pc.pla_rolling_abs_spearman(h, r, window=40)


def test_spearman_statistics_are_scale_free_and_wald_is_not():
    h, r = pair(4)
    a, b = pc.pla_complement_statistics(h, r), pc.pla_complement_statistics(h, 3.0 * r)
    assert a.spearman_abs == pytest.approx(b.spearman_abs, abs=1e-12)
    assert a.spearman_roll == pytest.approx(b.spearman_roll, abs=1e-12)
    assert b.wald > a.wald


def test_statistics_record_and_determinism():
    h, r = pair(5, lag=20)
    s1, s2 = pc.pla_complement_statistics(h, r), pc.pla_complement_statistics(h.copy(), r.copy())
    assert s1 == s2 and s1.window == 20 and s1.n == 250
    assert s1.spearman_roll < pc.pla_complement_statistics(*pair(5)).spearman_roll


@pytest.mark.parametrize("bad", ["short", "nan", "flat", "ndim"])
def test_missing_or_degenerate_input_is_refused(bad):
    h, r = pair(6)
    if bad == "short":
        h = h[:200]
    elif bad == "nan":
        r = r.copy()
        r[3] = np.nan
    elif bad == "flat":
        r = np.zeros(250)
    else:
        h = h.reshape(10, 25)
    for fn in (pc.pla_mz_wald, pc.pla_abs_spearman, pc.pla_rolling_abs_spearman, pc.pla_complement_statistics):
        with pytest.raises(ValueError):
            fn(h, r)


def test_window_is_validated():
    h, r = pair(6)
    for w in (1, 0, 250, 300, 2.5):
        with pytest.raises(ValueError):
            pc.pla_rolling_abs_spearman(h, r, window=w)


def test_calibrate_gives_the_promised_size_on_its_own_null():
    null = [pc.pla_complement_statistics(*p) for p in _null_pairs(1000, 600)]
    th = pc.pla_complement_calibrate(null, size=0.05, correction="none")
    dec = [pc.pla_complement_decide(s, th) for s in null]
    for field in ("wald", "spearman_abs", "spearman_roll"):
        assert np.mean([getattr(d, f"flag_{field}") for d in dec]) <= 0.05 + 1e-12
    assert th.size == 0.05 and th.n_null == 600 and th.per_statistic_size == 0.05
    bon = pc.pla_complement_calibrate(null, size=0.05, correction="bonferroni")
    assert bon.per_statistic_size == pytest.approx(0.05 / 3)
    assert bon.wald >= th.wald and bon.spearman_abs <= th.spearman_abs and bon.spearman_roll <= th.spearman_roll
    union = np.mean([pc.pla_complement_decide(s, bon).flagged for s in null])
    assert union <= 0.05 + 1e-12


def test_calibrate_refuses_bad_input():
    null = [pc.pla_complement_statistics(*p) for p in _null_pairs(2000, 30)]
    with pytest.raises(ValueError):
        pc.pla_complement_calibrate(null, size=0.01)   # 30 null replications cannot resolve a 1 percent size
    with pytest.raises(ValueError):
        pc.pla_complement_calibrate(null, size=0.0)
    with pytest.raises(ValueError):
        pc.pla_complement_calibrate(null, size=0.2, correction="holm")
    with pytest.raises(ValueError):
        pc.pla_complement_calibrate([], size=0.05)


def test_decide_logic():
    th = pc.PLAComplementThresholds(wald=10.0, spearman_abs=0.5, spearman_roll=0.6, size=0.01,
                                    per_statistic_size=0.01, n_null=1000, window=20, correction="none")
    ok = pc.PLAComplementStatistics(wald=1.0, p_value=0.6, intercept=0.0, slope=1.0, spearman_abs=0.9,
                                    spearman_roll=0.95, window=20, n=250)
    assert not pc.pla_complement_decide(ok, th).flagged
    d = pc.pla_complement_decide(_replace(ok, spearman_roll=0.3), th)
    assert d.flagged and d.flag_spearman_roll and not d.flag_wald and not d.flag_spearman_abs
    assert pc.pla_complement_decide(_replace(ok, wald=11.0), th).flag_wald
    with pytest.raises(ValueError):
        pc.pla_complement_decide(_replace(ok, window=10), th)


def test_complement_is_added_to_not_substituted_for_the_pla_zone():
    null = [pc.pla_complement_statistics(*p) for p in _null_pairs(1000, 400)]
    th = pc.pla_complement_calibrate(null, size=0.05)
    h, r = pair(7, lag=40)
    out = pc.pla_complement(h, r, th)
    assert out.pla_zone == frtb.pla_zone_exact(h, r)
    assert out.decision == pc.pla_complement_decide(out.statistics, th)


def test_defining_behaviour_hypothesis_on_known_truth():
    """The plan's hypothesis in a test's time: at a size of 1 percent per statistic the rolling-mean statistic flags a
    20-day-stale volatility in at least 10 percent of replications (the refutation bound; the 30 percent hypothesis is
    decided in docs/market/08_pla-complement.md at 6,000 replications) while the regulatory zone flags fewer, and the
    null stays at its size."""
    null = [pc.pla_complement_statistics(*p) for p in _null_pairs(10_000, 800)]
    th = pc.pla_complement_calibrate(null, size=0.01, correction="none")
    stale = [pair(50_000 + i, lag=20) for i in range(200)]
    power = np.mean([pc.pla_complement_decide(pc.pla_complement_statistics(*p), th).flag_spearman_roll for p in stale])
    pla = np.mean([frtb.pla_zone_exact(*p) != "green" for p in stale])
    fresh = [pc.pla_complement_statistics(*pair(90_000 + i)) for i in range(400)]
    size = np.mean([pc.pla_complement_decide(s, th).flag_spearman_roll for s in fresh])
    assert power >= 0.10
    assert pla < power
    assert size <= 0.04   # 1 percent nominal, 400 replications: a generous bound on Monte Carlo error
