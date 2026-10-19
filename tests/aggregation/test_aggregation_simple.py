"""RV-1301 to RV-1306: the first rungs of the aggregation ladder and the closed-form allocation quantities (contract
`engine-aggregation-api`, module aggregation.simple)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.aggregation import simple as s


def arr(*x):
    return np.array(x, dtype=float)


def test_simple_sum_golden_and_refusals():
    assert s.simple_sum(arr(3, 4)) == 7.0
    ec = arr(1.5, 0.0, 4.25, 9.0)
    assert s.simple_sum(ec) == pytest.approx(s.variance_covariance(ec, np.ones((4, 4))), rel=1e-12)
    with pytest.raises(ValueError, match="ec"):
        s.simple_sum(arr())
    with pytest.raises(ValueError, match="ec"):
        s.simple_sum(arr(1.0, float("nan")))
    with pytest.raises(ValueError, match="ec"):
        s.simple_sum(arr(1.0, -0.1))
    with pytest.raises(ValueError, match="ec"):
        s.simple_sum(np.ones((2, 2)))


def test_check_correlation_golden_and_refusals():
    assert s.check_correlation(np.array([[1, 0.5], [0.5, 1]])) is None
    assert s.check_correlation(np.ones((3, 3))) is None  # singular positive semi-definite is accepted
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    assert np.linalg.eigvalsh(bad).min() == pytest.approx(-0.8)
    with pytest.raises(ValueError, match="positive semi-definite"):
        s.check_correlation(bad)
    with pytest.raises(ValueError, match="square"):
        s.check_correlation(np.ones((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        s.check_correlation(np.array([[1, 0.5], [0.4, 1]]))
    with pytest.raises(ValueError, match="diagonal"):
        s.check_correlation(np.array([[1.1, 0.5], [0.5, 1]]))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        s.check_correlation(np.array([[1, 1.5], [1.5, 1]]))
    with pytest.raises(ValueError, match="corr"):
        s.check_correlation(np.array([[1, float("nan")], [float("nan"), 1]]))


def test_variance_covariance_golden_and_invariants():
    ec = arr(3, 4)
    assert s.variance_covariance(ec, np.eye(2)) == pytest.approx(5.0, abs=1e-12)
    assert s.variance_covariance(ec, np.ones((2, 2))) == pytest.approx(7.0, abs=1e-12)
    r = np.array([[1, 0.5], [0.5, 1]])
    assert s.variance_covariance(ec, r) == pytest.approx(6.0828, abs=5e-5)
    assert s.variance_covariance(ec, r) == pytest.approx(math.sqrt(37), abs=1e-12)
    # the check is called first
    with pytest.raises(ValueError, match="positive semi-definite"):
        s.variance_covariance(arr(1, 1, 1), np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]))
    with pytest.raises(ValueError, match="same length"):
        s.variance_covariance(arr(1, 2, 3), r)
    rng = np.random.default_rng(1)
    for _ in range(20):
        k = 4
        w, phi = rng.uniform(0.0, 1.0, 2)
        ar = phi ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))  # AR(1) correlation, nonnegative entries
        corr = 0.5 * (w * np.ones((k, k)) + (1.0 - w) * np.eye(k)) + 0.5 * ar
        e = rng.uniform(0.0, 10.0, k)
        v = s.variance_covariance(e, corr)
        assert math.sqrt(float(np.sum(e**2))) - 1e-9 <= v <= float(e.sum()) + 1e-9
        perm = rng.permutation(k)
        assert s.variance_covariance(e[perm], corr[np.ix_(perm, perm)]) == pytest.approx(v, rel=1e-12)
        assert s.variance_covariance(3.0 * e, corr) == pytest.approx(3.0 * v, rel=1e-12)


def test_hybrid_quantile_golden_and_invariants():
    x = np.linspace(0.0, 1.0, 1001) ** 2
    q = 0.99
    qx = float(np.sort(x)[math.ceil(q * x.size - 1e-9) - 1])
    got = s.hybrid_quantile(np.column_stack([x, 2.0 * x]), q)
    assert got == pytest.approx(3.0 * qx, rel=1e-12)
    # two columns of exactly zero Pearson correlation: x and a symmetric function of a centred x
    t = np.linspace(-1.0, 1.0, 2001)
    cols = np.column_stack([t + 1.0, t**2])
    assert np.corrcoef(cols.T)[0, 1] == pytest.approx(0.0, abs=1e-12)
    v1 = float(np.sort(cols[:, 0])[math.ceil(q * 2001 - 1e-9) - 1])
    v2 = float(np.sort(cols[:, 1])[math.ceil(q * 2001 - 1e-9) - 1])
    assert s.hybrid_quantile(cols, q) == pytest.approx(math.hypot(v1, v2), rel=1e-12)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(5000, 3)) @ np.linalg.cholesky(np.array([[1, .6, .2], [.6, 1, .1], [.2, .1, 1]])).T
    z = np.exp(z)
    v = [float(np.sort(z[:, k])[math.ceil(0.999 * 5000 - 1e-9) - 1]) for k in range(3)]
    assert s.hybrid_quantile(z, 0.999) <= sum(v) + 1e-9
    with pytest.raises(ValueError, match="samples"):
        s.hybrid_quantile(np.ones((1, 2)), 0.9)
    with pytest.raises(ValueError, match="constant"):
        s.hybrid_quantile(np.column_stack([np.arange(10.0), np.ones(10)]), 0.9)
    with pytest.raises(ValueError, match="q"):
        s.hybrid_quantile(z, 1.0)


def test_diversification_factor_golden():
    assert s.diversification_factor(6.0828, [3, 4]) == pytest.approx(0.8690, abs=5e-5)
    assert s.diversification_factor(6.0828, [3, 4]) == pytest.approx(6.0828 / 7.0, rel=1e-12)
    assert s.diversification_factor(9.0, [3, 4]) == pytest.approx(9.0 / 7.0)  # above 1 is not repaired
    with pytest.raises(ValueError, match="rho_marginals"):
        s.diversification_factor(1.0, [0.0, 0.0])
    with pytest.raises(ValueError, match="rho_marginals"):
        s.diversification_factor(1.0, [-3.0, 1.0])


def test_euler_diversification_benefit_golden():
    assert s.euler_diversification_benefit(6.0, 8.0) == pytest.approx(0.25, abs=1e-12)
    assert s.euler_diversification_benefit(8.0, 8.0) == 0.0
    with pytest.raises(ValueError, match="rho_standalone"):
        s.euler_diversification_benefit(1.0, 0.0)
    with pytest.raises(ValueError, match="rho_standalone"):
        s.euler_diversification_benefit(1.0, -2.0)


def test_std_contributions_golden_and_full_allocation():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    rc = s.std_contributions(cov)
    assert rc[0] == pytest.approx(1.4552, abs=5e-5)
    assert rc[1] == pytest.approx(2.6679, abs=5e-5)
    assert rc.sum() == pytest.approx(math.sqrt(17.0), rel=1e-12)
    assert rc[0] == pytest.approx(6.0 / math.sqrt(17.0), rel=1e-12)
    rng = np.random.default_rng(7)
    a = rng.normal(size=(5, 8))
    big = a @ a.T
    for c in (1.0, 2.5):
        r = s.std_contributions(big, c)
        assert r.shape == (5,)
        assert r.sum() == pytest.approx(c * math.sqrt(big.sum()), rel=1e-12)
    # the allocation of the variance-covariance total: cov = D R D with D = diag(EC)
    ec = arr(3, 4)
    corr = np.array([[1, 0.5], [0.5, 1]])
    assert s.std_contributions(np.outer(ec, ec) * corr).sum() == pytest.approx(s.variance_covariance(ec, corr),
    rel=1e-12)
    with pytest.raises(ValueError, match="symmetric"):
        s.std_contributions(np.array([[4.0, 2.0], [1.0, 9.0]]))
    with pytest.raises(ValueError, match="positive semi-definite"):
        s.std_contributions(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="variance"):
        s.std_contributions(np.zeros((2, 2)))
