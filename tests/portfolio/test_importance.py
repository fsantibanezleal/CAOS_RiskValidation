"""RV-715 to RV-717: importance sampling for the Gaussian copula (dossier 04 H.7, ladder rung C6).

The exponential twist and the factor mean shift are UNVERIFIED in dossier 04 H.7: these tests accept the estimator
only by its unbiasedness against plain Monte Carlo, never by the form of the twist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _portfolio_helpers import identical

from riskvalidation.portfolio.copula import mc_standard_errors, simulate_gaussian
from riskvalidation.portfolio.importance import (
    ImportanceResult,
    importance_sample_gaussian,
    solve_theta,
    twisted_pd,
    variance_ratio,
    weighted_var_es,
)
from riskvalidation.portfolio.portfolio import Portfolio, es_from_sample, expected_loss, var_from_sample


def test_twisted_pd_golden_and_invariants():
    assert twisted_pd(np.array([0.1]), np.array([1.0]), math.log(2.0))[0] == pytest.approx(0.181818, abs=5e-7)
    p = np.array([0.001, 0.05, 0.4, 0.9])
    c = np.array([1.0, 3.0, 0.5, 10.0])
    assert np.array_equal(twisted_pd(p, c, 0.0), p)
    prev = p
    for theta in (0.1, 0.5, 2.0, 10.0, 1000.0):
        t = twisted_pd(p, c, theta)
        assert (t > prev - 1e-15).all() and (t > 0).all() and (t <= 1.0).all() and np.isfinite(t).all()
        prev = t
    # the extremes of the conditional PD stay at the extremes
    assert list(twisted_pd(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 3.0)) == [0.0, 1.0]
    # the formula p e^{theta c} / (1 + p (e^{theta c} - 1))
    th = 0.7
    assert np.allclose(twisted_pd(p, c, th), p * np.exp(th * c) / (1 + p * (np.exp(th * c) - 1)), rtol=1e-12)
    with pytest.raises(ValueError):
        twisted_pd(p, c, -0.1)
    with pytest.raises(ValueError):
        twisted_pd(p, c[:2], 0.1)


def test_solve_theta_golden_and_refusals():
    p, c = np.array([0.1]), np.array([1.0])
    assert solve_theta(p, c, 2.0 / 11.0) == pytest.approx(math.log(2.0), abs=1e-10)
    assert solve_theta(p, c, 0.181818) == pytest.approx(0.693146, abs=5e-7)
    p3, c3 = np.array([0.02, 0.05, 0.1]), np.array([1.0, 2.0, 3.0])
    th = solve_theta(p3, c3, 3.0)
    assert th > 0 and float((c3 * twisted_pd(p3, c3, th)).sum()) == pytest.approx(3.0, abs=1e-9)
    assert solve_theta(p3, c3, 0.1) == 0.0 and solve_theta(p3, c3, float((c3 * p3).sum())) == 0.0
    assert solve_theta(p3, c3, 3.0) == solve_theta(p3, c3, 3.0)  # deterministic
    with pytest.raises(ValueError):
        solve_theta(p3, c3, float(c3.sum()))
    with pytest.raises(ValueError):
        solve_theta(p3, c3, float(c3.sum()) + 1.0)


_PLAIN: dict = {}


def _book() -> Portfolio:
    return identical(30, pd=0.02, rho=0.3, lgd=1.0)


def _plain_and_is(threshold: float, shift, s: int = 150_000):
    if "plain" not in _PLAIN:
        _PLAIN["plain"] = simulate_gaussian(_book(), s, seed=31, chunk_scenarios=50_000)
    res = importance_sample_gaussian(_book(), threshold, s, seed=32, factor_shift=shift, chunk_scenarios=50_000)
    return _PLAIN["plain"], res


@pytest.mark.parametrize("shift", [None, np.array([0.0]), np.array([-1.0])])
@pytest.mark.parametrize("threshold", [1.5, 4.0])
def test_importance_sampling_is_unbiased_against_plain_mc(shift, threshold):
    p = _book()
    plain, res = _plain_and_is(threshold, shift)
    assert isinstance(res, ImportanceResult)
    s = res.losses.size
    assert res.weights.shape == (s,)
    assert (res.weights > 0).all()
    assert res.n_scenarios == s
    assert res.factor_shift.shape == (1,)
    # the first and second unbiasedness identities
    se_w = res.weights.std(ddof=1) / math.sqrt(s)
    assert abs(res.weights.mean() - 1.0) <= 4 * se_w
    wl = res.weights * res.losses
    assert abs(wl.mean() - expected_loss(p)) <= 4 * wl.std(ddof=1) / math.sqrt(s)
    # the tail probability at several loss levels against plain Monte Carlo
    for x in (np.quantile(plain.losses, 0.95), np.quantile(plain.losses, 0.99), 6.0):
        pp = float((plain.losses > x).mean())
        se_p = math.sqrt(max(pp * (1 - pp), 1e-12) / plain.n_scenarios)
        ind = res.weights * (res.losses > x)
        assert abs(ind.mean() - pp) <= 4 * math.sqrt(se_p**2 + ind.var(ddof=1) / s), (x, pp, ind.mean())
    # the weighted VaR and ES against the plain ones
    v_is, e_is, sv_is, se_is = weighted_var_es(res, 0.99)
    v_p, e_p = var_from_sample(plain.losses, 0.99), es_from_sample(plain.losses, 0.99)
    sv_p, se_p_ = mc_standard_errors(plain.losses, 0.99)
    assert abs(e_is - e_p) <= 4 * math.hypot(se_is, se_p_)
    assert abs(v_is - v_p) <= 4 * math.hypot(sv_is, sv_p) + 1.0  # one default is one unit of loss


def test_importance_sampling_reduces_the_variance_of_a_tail_probability():
    plain, res = _plain_and_is(4.0, None)
    x = float(np.quantile(plain.losses, 0.99))
    pp = float((plain.losses > x).mean())
    se_plain = math.sqrt(pp * (1 - pp) / plain.n_scenarios)
    se_is = float((res.weights * (res.losses > x)).std(ddof=1) / math.sqrt(res.losses.size))
    # the default twist and shift are worth their cost at the 99 percent level
    assert variance_ratio(se_plain, se_is) > 1.5


def test_vectorised_theta_solver_agrees_with_solve_theta():
    """The sampler solves theta(z) for every scenario at once; each row must equal the scalar root of `solve_theta`."""
    from riskvalidation.portfolio.importance import _logit, _theta_rows

    rng = np.random.default_rng(14)
    c = np.array([1.0, 2.0, 3.0, 0.5, 4.0])
    x = 3.0
    rows = np.vstack([rng.uniform(0.001, 0.4, size=5) for _ in range(40)])
    rows[0] = 0.9  # a row whose mean loss is already above the threshold: theta = 0
    theta = _theta_rows(_logit(rows), c, x, rows @ c)
    expected = np.array([solve_theta(r, c, x) for r in rows])
    assert theta[0] == 0.0 and expected[0] == 0.0
    assert theta == pytest.approx(expected, abs=1e-7)


def test_importance_weights_integrate_to_one_exactly():
    """The contract's likelihood ratio is exact: E_Q[w] = 1 and E_Q[w L] = EL by quadrature over z and enumeration
    of the default patterns of three names, for several thresholds and factor shifts (no sampling noise)."""
    from scipy.special import ndtr, ndtri

    pd = np.array([0.05, 0.10, 0.20])
    beta = np.sqrt(np.array([0.3, 0.3, 0.5]))
    c = np.array([1.0, 2.0, 3.0])
    nodes, wts = np.polynomial.hermite_e.hermegauss(100)
    patterns = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T
    for x, mu in ((2.5, 0.0), (2.5, -0.7), (4.5, -1.5), (1.0, 0.5)):
        e_w = e_wl = 0.0
        for zq, wq in zip(nodes, wts, strict=True):
            z = mu + zq
            pc = ndtr((ndtri(pd) - beta * z) / np.sqrt(1 - beta**2))
            th = solve_theta(pc, c, x)
            tw = twisted_pd(pc, c, th)
            psi = np.sum(np.log(1 + pc * (np.exp(th * c) - 1)))
            q = np.prod(np.where(patterns == 1, tw, 1 - tw), axis=1)
            loss = patterns @ c
            lr = np.exp(-th * loss + psi) * math.exp(-mu * z + mu * mu / 2)
            e_w += wq * float(q @ lr) / math.sqrt(2 * math.pi)
            e_wl += wq * float(q @ (lr * loss)) / math.sqrt(2 * math.pi)
        assert e_w == pytest.approx(1.0, abs=1e-9)
        assert e_wl == pytest.approx(float((c * pd).sum()), abs=1e-9)


def test_importance_sampling_golden_one_name():
    p = identical(1, pd=0.1, rho=0.3)
    res = importance_sample_gaussian(p, 0.5, 1_000_000, seed=7)
    s = res.losses.size
    se_w = res.weights.std(ddof=1) / math.sqrt(s)
    assert abs(res.weights.mean() - 1.0) <= 4 * se_w
    wl = res.weights * res.losses
    assert abs(wl.mean() - 0.1) <= 4 * wl.std(ddof=1) / math.sqrt(s)
    ind = res.weights * (res.losses > 0.5)  # the same event as the default itself
    assert abs(ind.mean() - 0.1) <= 4 * ind.std(ddof=1) / math.sqrt(s)


def test_importance_reproducible_and_refusals():
    p = identical(30, pd=0.03, rho=0.25)
    a = importance_sample_gaussian(p, 10.0, 20_000, seed=2, chunk_scenarios=5_000)
    b = importance_sample_gaussian(p, 10.0, 20_000, seed=2, chunk_scenarios=5_000)
    c = importance_sample_gaussian(p, 10.0, 20_000, seed=3, chunk_scenarios=5_000)
    assert np.array_equal(a.losses, b.losses) and np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.losses, c.losses)
    assert a.seed == 2 and a.chunk_scenarios == 5_000
    # the default shift is deterministic and lowers the factor (a bad state of the economy)
    assert a.factor_shift[0] < 0.0 and np.array_equal(a.factor_shift, b.factor_shift)
    shifted = importance_sample_gaussian(p, 10.0, 1_000, seed=2, factor_shift=np.array([-0.5]))
    assert shifted.factor_shift[0] == -0.5
    two = Portfolio(
        ead=np.ones(2), pd=np.full(2, 0.02), lgd=np.ones(2), loadings=np.array([[0.4, 0.2], [0.2, 0.4]]),
        factor_corr=np.array([[1.0, 0.5], [0.5, 1.0]]), group=np.zeros(2, dtype=np.int64), group_ids=("a",),
        lgd_gamma=0.25, currency="USD", label="illustrative",
    )
    with pytest.raises(ValueError, match="correlated"):
        importance_sample_gaussian(two, 1.0, 100, seed=1)
    ok_two = Portfolio(**{**{f: getattr(two, f) for f in two.__dataclass_fields__}, "factor_corr": np.eye(2)})
    res2 = importance_sample_gaussian(ok_two, 1.0, 2_000, seed=1, factor_shift=np.array([-0.3, -0.3]))
    assert res2.factor_shift.shape == (2,)
    with pytest.raises(ValueError):
        importance_sample_gaussian(p, 1000.0, 100, seed=1)   # threshold above the maximum loss
    with pytest.raises(ValueError):
        importance_sample_gaussian(p, -1.0, 100, seed=1)
    with pytest.raises(ValueError):
        importance_sample_gaussian(p, 5.0, 0, seed=1)
    with pytest.raises(ValueError):
        importance_sample_gaussian(p, 5.0, 100, seed=1, factor_shift=np.zeros(3))


def test_weighted_var_es_golden_and_unit_weights():
    res = ImportanceResult(losses=np.array([0.0, 1.0, 2.0, 3.0]), weights=np.ones(4), factor_shift=np.zeros(1),
                           n_scenarios=4, seed=0, chunk_scenarios=4)
    v, e, sv, se = weighted_var_es(res, 0.5)
    assert v == 1.0 and e == pytest.approx(2.5, abs=1e-12)
    assert math.isnan(sv) and math.isnan(se)  # four scenarios cannot form 20 batches of 10
    rng = np.random.default_rng(6)
    x = rng.gamma(2.0, 1.0, size=40_000)
    unit = ImportanceResult(losses=x, weights=np.ones(x.size), factor_shift=np.zeros(1), n_scenarios=x.size, seed=0,
                            chunk_scenarios=x.size)
    for q in (0.9, 0.99):
        v, e, sv, se = weighted_var_es(unit, q)
        assert v == var_from_sample(x, q) and e == pytest.approx(es_from_sample(x, q), rel=1e-12)
        assert (sv, se) == pytest.approx(mc_standard_errors(x, q), rel=1e-12)
    # the tail is normalised by the scenario count S, never by the realised sum of the weights: weights that sum to
    # half of S carry half of the tail mass, so the VaR at q = 0.8 (tail mass 0.2) is the plain VaR at 0.6, and a
    # level whose tail mass (0.6) exceeds the weight mass is refused
    half = ImportanceResult(losses=x, weights=np.full(x.size, 0.5), factor_shift=np.zeros(1), n_scenarios=x.size,
                            seed=0, chunk_scenarios=x.size)
    assert weighted_var_es(half, 0.8)[0] == var_from_sample(x, 0.6)
    with pytest.raises(ValueError, match="tail mass"):
        weighted_var_es(half, 0.4)
    for q in (0.0, 1.0):
        with pytest.raises(ValueError):
            weighted_var_es(unit, q)


def test_weighted_var_es_at_the_production_threshold():
    """Review blocker B1: with the threshold at the plain 99.9 percent VaR the realised mean weight is far below 1, and
    the tail estimators must still agree with plain Monte Carlo (the tail probability of the contract is
    sum(w 1{L > x}) / S, normalised by the scenario count)."""
    book = identical(60, pd=0.02, rho=0.3, lgd=1.0)
    plain = simulate_gaussian(book, 1_000_000, seed=41, chunk_scenarios=100_000)
    v_p, e_p = var_from_sample(plain.losses, 0.999), es_from_sample(plain.losses, 0.999)
    sv_p, se_p = mc_standard_errors(plain.losses, 0.999)
    res = importance_sample_gaussian(book, v_p, 200_000, seed=42, chunk_scenarios=50_000)
    assert res.weights.mean() < 0.5  # the realised mean weight is not 1 at this threshold (0.03 in the review)
    v_is, e_is, sv_is, se_is = weighted_var_es(res, 0.999)
    assert abs(v_is - v_p) <= 4 * math.hypot(sv_is, sv_p) + 1.0
    assert abs(e_is - e_p) <= 4 * math.hypot(se_is, se_p) + 0.3
    assert abs(e_is - e_p) < 2.0  # the normalised estimator gave an ES of about 39 against 25.6
    # the tail probability the VaR rests on
    tail = float((res.weights * (res.losses > v_p)).sum()) / res.losses.size
    assert abs(tail - 0.001) < 0.0005


def test_variance_ratio():
    assert variance_ratio(0.002, 0.0005) == pytest.approx(16.0, abs=1e-12)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            variance_ratio(0.002, bad)
    with pytest.raises(ValueError):
        variance_ratio(-0.002, 0.001)
