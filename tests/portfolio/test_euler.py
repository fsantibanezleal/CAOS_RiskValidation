"""RV-718 and RV-719: Euler contributions for ES and VaR, the ASRF anchor and the concentration measures (dossier 04
H.2 and H.6; dossier 05 G.4)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.engines.vasicek import lhp_quantile
from riskvalidation.portfolio.euler import (
    asrf_var_contributions,
    effective_names,
    es_contributions,
    hhi,
    var_contributions_kernel,
)
from riskvalidation.portfolio.portfolio import es_from_sample, var_from_sample


def test_es_contributions_golden_and_full_allocation():
    g1 = np.arange(10.0)
    g2 = 2.0 * np.arange(10.0)
    gl = np.column_stack([g1, g2])
    losses = 3.0 * np.arange(10.0)
    rc = es_contributions(gl, losses, 0.9)
    assert es_from_sample(losses, 0.9) == pytest.approx(27.0, abs=1e-12)
    assert rc == pytest.approx([9.0, 18.0], abs=1e-12) and rc.sum() == pytest.approx(27.0, abs=1e-12)
    # full allocation: the contributions add up to the sample ES exactly, for any level, including ties at the boundary
    rng = np.random.default_rng(10)
    parts = rng.gamma(1.5, 2.0, size=(2000, 4))
    total = parts.sum(axis=1)
    for q in (0.5, 0.9, 0.95, 0.99, 0.999, 0.9003, 0.91337, 0.99917):  # the last three leave a fractional boundary mass
        c = es_contributions(parts, total, q)
        assert c.sum() == pytest.approx(es_from_sample(total, q), rel=1e-12)
        assert (c >= 0).all()
    ties = rng.integers(0, 4, size=(500, 3)).astype(float)
    t_total = ties.sum(axis=1)
    for q in (0.3, 0.55, 0.9, 0.99):
        assert es_contributions(ties, t_total, q).sum() == pytest.approx(es_from_sample(t_total, q), rel=1e-12)
    # independent of the order of the scenarios (no ties)
    perm = rng.permutation(2000)
    shuffled = es_contributions(parts[perm], total[perm], 0.99)
    assert shuffled == pytest.approx(es_contributions(parts, total, 0.99), rel=1e-12)
    # the inputs are untouched
    assert gl[0, 1] == 0.0 and gl[9, 1] == 18.0
    with pytest.raises(ValueError, match="add up"):
        es_contributions(gl, losses + 1.0, 0.9)
    with pytest.raises(ValueError):
        es_contributions(gl[:5], losses, 0.9)
    with pytest.raises(ValueError):
        es_contributions(gl, losses, 1.0)
    with pytest.raises(ValueError):
        es_contributions(np.empty((0, 2)), np.empty(0), 0.9)
    with pytest.raises(ValueError):
        es_contributions(np.column_stack([g1, [np.nan] * 10]), losses, 0.9)


def test_var_contributions_kernel_properties():
    rng = np.random.default_rng(7)
    s = 1_000_000
    base = rng.gamma(3.0, 1.0, size=s)
    shares = np.array([0.2, 0.3, 0.5])
    const = base[:, None] * shares[None, :]
    rc = var_contributions_kernel(const, base, 0.99)
    var = var_from_sample(base, 0.99)
    assert (rc >= 0).all()
    assert rc == pytest.approx(shares * var, rel=0.01)  # constant shares: a_g VaR
    # independent components: the contributions add up to the VaR within the document tolerance 0.01
    parts = rng.gamma(2.0, 1.0, size=(s, 3)) * np.array([1.0, 2.0, 0.5])
    total = parts.sum(axis=1)
    rc2 = var_contributions_kernel(parts, total, 0.99)
    assert (rc2 >= 0).all()
    assert abs(rc2.sum() - var_from_sample(total, 0.99)) <= 0.01 * var_from_sample(total, 0.99)
    # the larger, higher-variance component carries more of the tail
    assert rc2[1] > rc2[0] > rc2[2]
    # invariant to the order of the scenarios, to a given bandwidth equal to the default, and the input is not mutated
    small = parts[:20_000]
    st = total[:20_000]
    perm = rng.permutation(20_000)
    a = var_contributions_kernel(small, st, 0.95)
    b = var_contributions_kernel(small[perm], st[perm], 0.95)
    assert a == pytest.approx(b, rel=1e-9)
    h = 1.06 * st.std(ddof=1) * 20_000 ** (-0.2)
    assert var_contributions_kernel(small, st, 0.95, bandwidth=h) == pytest.approx(a, rel=1e-9)
    wide = var_contributions_kernel(small, st, 0.95, bandwidth=5 * h)
    assert wide.shape == (3,) and (wide >= 0).all()
    with pytest.raises(ValueError):
        var_contributions_kernel(small, st, 0.95, bandwidth=0.0)
    with pytest.raises(ValueError):
        var_contributions_kernel(small, np.full(20_000, 3.0), 0.95)  # no spread: the default bandwidth is zero
    with pytest.raises(ValueError, match="add up"):
        var_contributions_kernel(small, st + 1.0, 0.95)


def test_asrf_var_contributions_golden():
    rc = asrf_var_contributions(np.array([60.0, 40.0]), np.array([0.01, 0.05]), 0.45, 0.15, 0.999)
    assert rc == pytest.approx([2.977148, 5.643106], abs=5e-7) and rc.sum() == pytest.approx(8.620255, abs=5e-7)
    # identical names: the contributions add up to EAD * LGD * the LHP quantile of the engine
    n = 8
    rc_i = asrf_var_contributions(np.full(n, 2.0), np.full(n, 0.02), np.full(n, 0.4), 0.15, 0.999)
    assert rc_i.sum() == pytest.approx(n * 2.0 * 0.4 * lhp_quantile(0.999, 0.02, 0.15), rel=1e-12)
    # a per-name correlation, equal to the scalar when it is constant
    vec = asrf_var_contributions(np.array([60.0, 40.0]), np.array([0.01, 0.05]), np.array([0.45, 0.45]),
                                 np.array([0.15, 0.15]), 0.999)
    assert vec == pytest.approx(rc, rel=1e-12)
    diff = asrf_var_contributions(np.array([60.0, 40.0]), np.array([0.01, 0.05]), 0.45, np.array([0.10, 0.20]), 0.999)
    assert diff[0] != rc[0] and diff[1] != rc[1]
    for bad in (
        dict(ead=np.array([-1.0, 1.0])),
        dict(pd=np.array([0.0, 0.05])),
        dict(rho=1.0),
        dict(q=1.0),
        dict(lgd=1.5),
    ):
        kw = dict(ead=np.array([60.0, 40.0]), pd=np.array([0.01, 0.05]), lgd=0.45, rho=0.15, q=0.999)
        kw.update(bad)
        with pytest.raises(ValueError):
            asrf_var_contributions(**kw)


def test_hhi_and_effective_names():
    e = np.array([50.0, 30.0, 20.0])
    assert hhi(e) == pytest.approx(0.38, abs=5e-7) and effective_names(e) == pytest.approx(2.631579, abs=5e-7)
    assert hhi(np.full(7, 3.0)) == pytest.approx(1.0 / 7.0, abs=1e-15)
    assert effective_names(np.full(7, 3.0)) == pytest.approx(7.0, abs=1e-12)
    assert hhi(np.array([5.0])) == 1.0 and effective_names(np.array([5.0])) == 1.0
    assert hhi(1000.0 * e) == pytest.approx(hhi(e), rel=1e-14)  # invariant to the scale
    rng = np.random.default_rng(2)
    x = rng.gamma(1.0, 1.0, size=200)
    assert 1.0 / 200 <= hhi(x) <= 1.0 and 1.0 <= effective_names(x) <= 200
    # the fixture portfolio of the contract: 10, 50 and 1000 names
    fixture = np.concatenate([np.full(10, 4e6), np.full(50, 6e5), np.full(1000, 3e4)])
    assert hhi(fixture) == pytest.approx(0.01789, abs=5e-6)
    for bad in (np.array([0.0, 0.0]), np.array([]), np.array([-1.0, 2.0])):
        with pytest.raises(ValueError):
            hhi(bad)
