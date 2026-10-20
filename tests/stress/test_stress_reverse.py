"""RV-1313 and RV-1314: the reverse stress (contract `engine-stress-api`, module stress.reverse)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.stress import reverse as r

COV = np.diag([1.0, 4.0])
GRAD = np.array([1.0, 1.0])


def test_reverse_stress_linear_golden_and_minimality():
    sol = r.reverse_stress_linear(GRAD, COV, 5.0)
    assert sol.method == "closed-form-linear" and sol.converged
    assert sol.x_star == pytest.approx([1.0, 4.0], abs=1e-12)
    assert sol.z_star == pytest.approx([1.0, 2.0], abs=1e-12)
    assert sol.distance == pytest.approx(5.0 / math.sqrt(5.0), abs=1e-12) and sol.distance == pytest.approx(
        2.2361, abs=5e-5
    )
    assert sol.loss_at_x == pytest.approx(5.0, rel=1e-9) and sol.threshold == 5.0
    # a general covariance: the loss at x equals the threshold and random feasible points are never closer
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T + 0.5 * np.eye(4)
    g = rng.normal(size=4)
    s = r.reverse_stress_linear(g, cov, 7.5)
    assert abs(g @ s.x_star - 7.5) / 7.5 <= 1e-9
    prec = np.linalg.inv(cov)
    assert s.distance == pytest.approx(math.sqrt(s.x_star @ prec @ s.x_star), rel=1e-10)
    for _ in range(2000):
        x = rng.normal(scale=3.0, size=4)
        if g @ x < 7.5:
            x = x * (7.5 / (g @ x)) if g @ x > 0 else x
        if g @ x >= 7.5 - 1e-12:
            assert math.sqrt(x @ prec @ x) >= s.distance - 1e-9
    # the distance in z units: z_star is x_star over the factor sd
    assert s.z_star == pytest.approx(s.x_star / np.sqrt(np.diag(cov)), abs=1e-12)
    assert r.reverse_stress_linear(GRAD, COV, 0.0).distance == 0.0


def test_reverse_stress_linear_refusals():
    with pytest.raises(ValueError, match="cov"):
        r.reverse_stress_linear(GRAD, np.array([[1.0, 2.0], [2.0, 1.0]]), 5.0)
    with pytest.raises(ValueError, match="cov"):
        r.reverse_stress_linear(GRAD, np.array([[1.0, 0.0], [0.0, 0.0]]), 5.0)
    with pytest.raises(ValueError, match="cov"):
        r.reverse_stress_linear(GRAD, np.array([[1.0, 0.1], [0.0, 1.0]]), 5.0)
    with pytest.raises(ValueError, match="gradient"):
        r.reverse_stress_linear(np.zeros(2), COV, 5.0)
    with pytest.raises(ValueError, match="threshold"):
        r.reverse_stress_linear(GRAD, COV, -1.0)
    with pytest.raises(ValueError, match="gradient"):
        r.reverse_stress_linear(np.ones(3), COV, 5.0)
    with pytest.raises(ValueError, match="threshold"):
        r.reverse_stress_linear(GRAD, COV, float("nan"))


def test_plausibility_goldens_and_invariants():
    d = math.sqrt(5.0)
    assert r.plausibility(d, 2) == pytest.approx(0.0821, abs=5e-5) and r.plausibility(d, 2) == pytest.approx(
        math.exp(-2.5), abs=1e-12
    )
    assert r.plausibility(d, 3) == pytest.approx(0.1718, abs=5e-5)
    # chi-square with 3 degrees of freedom by its closed-form survival function
    x = 5.0
    sf3 = math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    assert r.plausibility(d, 3) == pytest.approx(sf3, abs=1e-12)
    ds = np.linspace(0.0, 6.0, 25)
    vals = [r.plausibility(float(v), 4) for v in ds]
    assert vals[0] == 1.0 and all(b < a for a, b in zip(vals, vals[1:], strict=False))
    with pytest.raises(ValueError, match="df"):
        r.plausibility(1.0, 0)
    with pytest.raises(ValueError, match="distance"):
        r.plausibility(-1.0, 2)


def _linear(x):
    return float(x[0] + x[1])


STARTS = [np.array([3.0, 1.0]), np.array([-2.0, 6.0]), np.array([0.5, 0.5]), np.array([10.0, -3.0])]


def test_reverse_stress_matches_the_closed_form_from_every_start():
    closed = r.reverse_stress_linear(GRAD, COV, 5.0)
    best, per_start = r.reverse_stress(_linear, COV, 5.0, STARTS)
    assert best.method == "scipy-optimize" and best.converged
    assert len(per_start) == len(STARTS)
    for sol in per_start:
        assert sol.converged
        assert np.max(np.abs(sol.x_star - closed.x_star)) <= 1e-6
        assert sol.distance == pytest.approx(closed.distance, abs=1e-6)
        assert abs(sol.loss_at_x - 5.0) / 5.0 <= 1e-6
        assert sol.threshold == 5.0
    assert best.distance == min(s.distance for s in per_start if s.converged)
    assert best.z_star == pytest.approx(best.x_star / np.sqrt(np.diag(COV)), abs=1e-12)


def test_reverse_stress_quadratic_loss():
    def loss(x):
        return float(x[0] ** 2 + x[1] ** 2)

    starts = [np.array([3.0, 0.5]), np.array([-1.0, 2.5]), np.array([0.7, -2.0])]
    best, per_start = r.reverse_stress(loss, np.eye(2), 4.0, starts)
    assert best.distance == pytest.approx(2.0, abs=1e-5)
    assert np.linalg.norm(best.x_star) == pytest.approx(2.0, abs=1e-5)
    assert abs(best.loss_at_x - 4.0) / 4.0 <= 1e-6
    for s in per_start:
        if s.converged:
            assert abs(s.loss_at_x - 4.0) / 4.0 <= 1e-6 and s.distance >= best.distance - 1e-12


def test_reverse_stress_refusals():
    with pytest.raises(ValueError, match="starts"):
        r.reverse_stress(_linear, COV, 5.0, STARTS[:2])
    with pytest.raises(ValueError, match="start"):
        r.reverse_stress(_linear, COV, 5.0, [np.zeros(3)] * 3)
    with pytest.raises(ValueError, match="threshold"):
        r.reverse_stress(_linear, COV, 0.0, STARTS)
    with pytest.raises(ValueError, match="cov"):
        r.reverse_stress(_linear, np.array([[1.0, 0.0], [0.0, -1.0]]), 5.0, STARTS)
    # a loss that can never reach the threshold: no start converges
    with pytest.raises(ValueError, match="converge"):
        r.reverse_stress(lambda x: float(1.0 - np.exp(-np.sum(x * x))), COV, 5.0, STARTS)
    with pytest.raises(ValueError, match="tolerance"):
        r.reverse_stress(_linear, COV, 5.0, STARTS, tolerance=0.0)


def test_reverse_stress_two_solutions_returns_the_minimum_distance_with_its_plausibility():
    """A loss that is satisfied by two disjoint regions with different distances: the nearer one must be returned."""

    def loss(x):  # |x0| >= 2 with a cheap branch at x0 = +2 and an expensive one at x0 = -3 (scaled)
        return float(max(x[0], -x[0] / 1.5))

    starts = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.5, 0.2]), np.array([-0.5, -0.2])]
    best, per_start = r.reverse_stress(loss, np.eye(2), 2.0, starts)
    conv = [s for s in per_start if s.converged]
    assert len({round(s.distance, 3) for s in conv}) == 2  # both solutions were found
    assert best.distance == pytest.approx(2.0, abs=1e-4)
    assert best.distance == min(s.distance for s in conv)
    assert max(s.distance for s in conv) == pytest.approx(3.0, abs=1e-4)
    assert r.plausibility(best.distance, 2) == pytest.approx(math.exp(-2.0), abs=1e-4)
