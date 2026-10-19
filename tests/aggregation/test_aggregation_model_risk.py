"""RV-1317 and RV-1318: the model risk measures of dossier 06 B.12 (contract `engine-aggregation-api`, module
aggregation.model_risk)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.aggregation import model_risk as mr


def test_barrieu_scandolo_golden_and_invariants():
    out = mr.barrieu_scandolo((100.0, 130.0, 90.0), 0)
    assert out.am == pytest.approx(0.3, abs=1e-12)
    assert out.rm == pytest.approx(0.75, abs=1e-12)
    assert (out.rho_x0, out.rho_max, out.rho_min) == (100.0, 130.0, 90.0)
    assert out.am == pytest.approx(out.rho_max / out.rho_x0 - 1.0, rel=1e-12)
    assert out.rm == pytest.approx((out.rho_max - out.rho_x0) / (out.rho_max - out.rho_min), rel=1e-12)
    flat = mr.barrieu_scandolo((50.0, 50.0, 50.0), 1)
    assert flat.am == 0.0 and flat.rm == 0.0  # the convention for 0 / 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        rhos = rng.uniform(1.0, 100.0, 6)
        k = int(rng.integers(0, 6))
        r = mr.barrieu_scandolo(rhos, k)
        assert r.am >= 0.0 and 0.0 <= r.rm <= 1.0
        assert r.rho_x0 == rhos[k]
    top = mr.barrieu_scandolo((100.0, 130.0), 1)  # the reference is the maximum
    assert top.am == 0.0 and top.rm == 0.0
    bottom = mr.barrieu_scandolo((100.0, 130.0), 0)
    assert bottom.rm == 0.0 and bottom.am == pytest.approx(0.3)
    with pytest.raises(ValueError, match="rho_x0|rhos"):
        mr.barrieu_scandolo((0.0, 130.0), 0)
    with pytest.raises(ValueError, match="rho_x0|rhos"):
        mr.barrieu_scandolo((-5.0, 130.0), 0)
    with pytest.raises(ValueError, match="reference_index"):
        mr.barrieu_scandolo((100.0, 130.0), 2)
    with pytest.raises(ValueError, match="reference_index"):
        mr.barrieu_scandolo((100.0, 130.0), -1)
    with pytest.raises(ValueError, match="rhos"):
        mr.barrieu_scandolo((), 0)
    with pytest.raises(ValueError, match="rhos"):
        mr.barrieu_scandolo((100.0, float("nan")), 0)


def test_glasserman_xu_golden_and_recovery():
    v = np.array([0.0, 1.0])
    e = math.exp(1.0)
    m = np.array([1.0, e]) / ((1.0 + e) / 2.0)
    assert m == pytest.approx([0.5379, 1.4621], abs=5e-5)
    entropy = float(np.mean(m * np.log(m)))
    assert entropy == pytest.approx(0.1109, abs=5e-5)
    # the golden route: the budget of theta = 1 is recovered
    out = mr.glasserman_xu_worst_case(v, entropy)
    assert out.theta == pytest.approx(1.0, abs=1e-3)
    assert out.relative_entropy == pytest.approx(entropy, abs=1e-9)
    assert out.baseline == pytest.approx(0.5, abs=1e-12)
    assert out.worst_case == pytest.approx(0.7311, abs=5e-5)
    assert out.worst_case == pytest.approx(float(np.mean(m * v)), abs=1e-8)
    printed = mr.glasserman_xu_worst_case(v, 0.1109)
    assert printed.theta == pytest.approx(1.0, abs=1e-3)
    # a general sample: the entropy binds and the weights are the exponential tilt
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    res = mr.glasserman_xu_worst_case(x, 0.2)
    assert res.relative_entropy == pytest.approx(0.2, abs=1e-9)
    w = np.exp(res.theta * (x - x.max()))
    mm = w / w.mean()
    assert float(np.mean(mm * np.log(mm))) == pytest.approx(0.2, abs=1e-9)
    assert res.worst_case == pytest.approx(float(np.mean(mm * x)), rel=1e-9)
    assert res.worst_case >= res.baseline
    # the tilt is invariant to a shift of V and scales theta with the scale of V
    shifted = mr.glasserman_xu_worst_case(x + 100.0, 0.2)
    assert shifted.theta == pytest.approx(res.theta, rel=1e-6)
    assert shifted.worst_case == pytest.approx(res.worst_case + 100.0, rel=1e-9)
    scaled = mr.glasserman_xu_worst_case(10.0 * x, 0.2)
    assert scaled.theta == pytest.approx(res.theta / 10.0, rel=1e-6)


def test_glasserman_xu_monotone_and_refusals():
    rng = np.random.default_rng(4)
    x = rng.lognormal(size=400)
    wc = [mr.glasserman_xu_worst_case(x, bu).worst_case for bu in (0.01, 0.05, 0.2, 1.0, 3.0)]
    assert all(a < b for a, b in zip(wc, wc[1:], strict=False))
    assert wc[-1] <= x.max() + 1e-12
    with pytest.raises(ValueError, match="budget"):
        mr.glasserman_xu_worst_case(x, math.log(x.size) + 0.01)
    with pytest.raises(ValueError, match="budget"):
        mr.glasserman_xu_worst_case(x, math.log(x.size))
    with pytest.raises(ValueError, match="budget"):
        mr.glasserman_xu_worst_case(x, 0.0)
    with pytest.raises(ValueError, match="budget"):
        mr.glasserman_xu_worst_case(x, -1.0)
    with pytest.raises(ValueError, match="constant"):
        mr.glasserman_xu_worst_case(np.ones(10), 0.1)
    with pytest.raises(ValueError, match="v"):
        mr.glasserman_xu_worst_case(np.array([1.0, float("nan")]), 0.1)
    with pytest.raises(ValueError, match="v"):
        mr.glasserman_xu_worst_case(np.array([1.0]), 0.1)
    # a sample whose maximum is attained twice cannot reach log S: the attainable entropy is log(S / 2)
    tied = np.array([0.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="budget"):
        mr.glasserman_xu_worst_case(tied, math.log(2.0) + 1e-3)
