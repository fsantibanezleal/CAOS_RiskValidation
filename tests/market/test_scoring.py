"""RV-813 to RV-815: the FZ0 and tick losses and the comparison tests (contract `engine-market-api`, module
market.scoring; dossier 05 A.1 and B.6)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from riskvalidation.market import scoring as s


def a(*x):
    return np.array(x, dtype=float)


def general_fz(y, v, e, alpha, g1, g2, cal_g2):
    """The Fissler-Ziegel loss of arXiv 1707.05108 eq. 4, written independently of the engine."""
    hit = (y <= v).astype(float)
    return (hit - alpha) * (g1(v) - g1(y)) + g2(e) * (e - v + hit * (v - y) / alpha) - cal_g2(e)


def test_fz0_loss_goldens_and_reduction():
    assert s.fz0_loss(a(-0.03), a(-0.02), a(-0.025), 0.01)[0] == pytest.approx(36.111121, abs=5e-7)
    assert s.fz0_loss(a(0.01), a(-0.02), a(-0.03), 0.025)[0] == pytest.approx(-3.839891, abs=5e-7)
    assert s.fz0_loss(a(0.0), a(-0.02), a(-0.02), 0.01)[0] == pytest.approx(-3.912023, abs=5e-7)
    assert s.fz0_loss(a(0.0), a(-0.02), a(-0.02), 0.01)[0] == pytest.approx(math.log(0.02), abs=1e-12)
    rng = np.random.default_rng(1)
    y = rng.standard_t(5, size=400) * 0.01
    v = -np.abs(rng.normal(0.02, 0.005, size=400)) - 0.001
    e = v - np.abs(rng.normal(0.005, 0.002, size=400))
    for alpha in (0.01, 0.025, 0.05):
        ref = general_fz(y, v, e, alpha, lambda x: 0.0 * x, lambda x: -1.0 / x, lambda x: -np.log(-x))
        assert np.max(np.abs(s.fz0_loss(y, v, e, alpha) - ref)) < 1e-12
    # the shape follows the input
    assert s.fz0_loss(y, v, e, 0.025).shape == y.shape


def test_fz0_loss_refusals():
    ok = (a(-0.03), a(-0.02), a(-0.025), 0.01)
    s.fz0_loss(*ok)
    for bad in ((a(-0.03), a(0.02), a(-0.025), 0.01), (a(-0.03), a(-0.02), a(0.025), 0.01),
                (a(-0.03), a(-0.02), a(-0.01), 0.01), (a(-0.03), a(0.0), a(-0.025), 0.01),
                (a(-0.03), a(-0.02), a(-0.025), 0.0)):
        with pytest.raises(ValueError):
            s.fz0_loss(*bad)
    with pytest.raises(ValueError):
        s.fz0_loss(a(-0.03, 0.0), a(-0.02), a(-0.025), 0.01)


def test_tick_loss():
    assert s.tick_loss(a(-0.03), a(-0.02), 0.01)[0] == pytest.approx(0.0099, abs=1e-12)
    assert s.tick_loss(a(-0.01), a(-0.02), 0.01)[0] == pytest.approx(0.0001, abs=1e-12)
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.01, 300)
    v = np.full(300, -0.02)
    t = s.tick_loss(y, v, 0.05)
    assert np.all(t >= 0.0) and s.tick_loss(a(-0.02), a(-0.02), 0.05)[0] == 0.0
    # the minimiser over a constant is the empirical quantile
    qs = np.quantile(y, 0.05)
    assert s.tick_loss(y, np.full(300, qs), 0.05).sum() <= s.tick_loss(y, np.full(300, qs + 0.002), 0.05).sum()
    assert s.tick_loss(y, np.full(300, qs), 0.05).sum() <= s.tick_loss(y, np.full(300, qs - 0.002), 0.05).sum()


def test_mean_scores():
    y = a(-0.03, 0.01, -0.01, 0.0)
    v = a(-0.02, -0.02, -0.02, -0.02)
    e = a(-0.025, -0.03, np.nan, -0.03)
    r = s.mean_scores(y, v, e, 0.025)
    assert isinstance(r, s.Scores) and r.n == 4 and r.n_fz0 == 3
    ok = np.isfinite(e)
    assert r.fz0_mean == pytest.approx(s.fz0_loss(y[ok], v[ok], e[ok], 0.025).mean(), abs=1e-12)
    assert r.tick_mean == pytest.approx(s.tick_loss(y, v, 0.025).mean(), abs=1e-12)
    none = s.mean_scores(y, v, np.full(4, np.nan), 0.025)
    assert none.n_fz0 == 0 and none.fz0_mean is None and none.n == 4 and math.isfinite(none.tick_mean)
    with pytest.raises(ValueError):
        s.mean_scores(a(0.0, np.nan), a(-0.02, -0.02), a(-0.03, -0.03), 0.025)
    with pytest.raises(ValueError):
        s.mean_scores(y, v[:3], e, 0.025)


def hac_reference(d, lags):
    t = len(d)
    mean = sum(d) / t
    c = [x - mean for x in d]

    def gamma(j):
        return sum(c[i] * c[i - j] for i in range(j, t)) / t

    lrv = gamma(0) + 2.0 * sum((1.0 - j / (lags + 1.0)) * gamma(j) for j in range(1, lags + 1))
    z = mean / math.sqrt(lrv / t)
    return z, 2.0 * (1.0 - NormalDist().cdf(abs(z)))


def test_diebold_mariano():
    d = a(0.1, -0.05, 0.2, 0.1, 0.0, 0.15)
    r = s.diebold_mariano(d, np.zeros(6), 0)
    assert isinstance(r, s.DmResult)
    assert r.statistic == pytest.approx(2.401922, abs=5e-7) and r.p_value == pytest.approx(0.016309, abs=5e-7)
    assert r.mean_diff == pytest.approx(d.mean()) and r.n == 6 and r.lags == 0
    rng = np.random.default_rng(3)
    la = rng.gamma(2, 1, 300)
    lb = la + 0.1 + rng.normal(0, 0.2, 300)
    for lags in (0, 1, 5):
        r = s.diebold_mariano(la, lb, lags)
        z, p = hac_reference(list(la - lb), lags)
        assert r.statistic == pytest.approx(z, rel=1e-9) and r.p_value == pytest.approx(p, abs=1e-9) and r.lags == lags
    assert s.diebold_mariano(la, lb, 0).mean_diff < 0  # model A has the lower loss
    assert s.diebold_mariano(lb, la, 0).statistic == pytest.approx(-s.diebold_mariano(la, lb, 0).statistic)
    with pytest.raises(ValueError):
        s.diebold_mariano(la, la + 0.5, 0)  # a constant difference has no spread
    with pytest.raises(ValueError):
        s.diebold_mariano(a(1, 2), a(2, 1), 1)  # T <= lags + 1
    with pytest.raises(ValueError):
        s.diebold_mariano(la, lb[:-1], 0)
    with pytest.raises(ValueError):
        s.diebold_mariano(la, lb, -1)


def test_model_confidence_set():
    pytest.importorskip("arch")
    rng = np.random.default_rng(4)
    t = 600
    base = rng.gamma(2, 0.5, size=t)
    losses = np.column_stack([base + rng.normal(0, 0.05, t), base + 0.002 + rng.normal(0, 0.05, t),
                              base + 0.5 + rng.normal(0, 0.05, t)])
    r = s.model_confidence_set(losses, confidence=0.90, n_bootstrap=300, block_length=5, seed=1)
    assert isinstance(r, s.McsResult) and r.p_values.shape == (3,)
    assert 2 not in r.members and {0, 1} <= set(r.members)
    assert r.confidence == 0.90 and r.statistic == "TR"
    assert all(r.p_values[i] >= 0.1 for i in r.members) and r.p_values[2] < 0.1
    again = s.model_confidence_set(losses, confidence=0.90, n_bootstrap=300, block_length=5, seed=1)
    assert again.members == r.members and np.array_equal(again.p_values, r.p_values)
    tmax = s.model_confidence_set(losses, statistic="Tmax", n_bootstrap=300, block_length=5, seed=1)
    assert tmax.statistic == "Tmax" and 2 not in tmax.members
    one = s.model_confidence_set(losses[:, :1], n_bootstrap=100, seed=1)
    assert one.members == [0] and one.p_values.tolist() == [1.0]
    assert len(r.members) >= 1
    with pytest.raises(ValueError):
        s.model_confidence_set(losses, confidence=1.2)
    with pytest.raises(ValueError):
        s.model_confidence_set(losses, statistic="X")
