"""RV-501 to RV-505: the term-structure identities, competing risks and the one-factor conversion, each function at
its golden value or invariant from the provisions API contract (dossier 02 B.2, B.4, C.1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.provisions import term_structure as ts

MONTHS = np.array([3, 6])


def test_from_hazard_golden_and_identities():
    out = ts.from_hazard(np.array([0.01, 0.01]), MONTHS)
    np.testing.assert_allclose(out.survival, [0.99, 0.9801], atol=1e-15)
    np.testing.assert_allclose(out.marginal, [0.01, 0.0099], atol=1e-15)
    np.testing.assert_allclose(out.cumulative, [0.01, 0.0199], atol=1e-15)
    assert out.months.dtype == np.int64
    assert dataclasses.is_dataclass(out)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.months = np.array([1, 2])  # type: ignore[misc]
    # identities for a batch of random hazards, to 1e-12
    rng = np.random.default_rng(5)
    h = rng.uniform(0.001, 0.2, size=(4, 3, 8))
    m = np.arange(1, 9) * 3
    r = ts.from_hazard(h, m)
    assert r.survival.shape == h.shape
    np.testing.assert_allclose(r.survival, np.cumprod(1 - h, axis=-1), atol=1e-12)
    s_prev = np.concatenate([np.ones(h.shape[:-1] + (1,)), r.survival[..., :-1]], axis=-1)
    np.testing.assert_allclose(r.marginal, s_prev * h, atol=1e-12)
    np.testing.assert_allclose(r.cumulative, 1 - r.survival, atol=1e-12)
    assert np.all(np.diff(r.cumulative, axis=-1) > 0)
    assert np.all((r.cumulative > 0) & (r.cumulative < 1))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2, np.nan])
def test_hazard_outside_open_interval_is_refused(bad):
    with pytest.raises(ValueError, match="hazard"):
        ts.from_hazard(np.array([0.01, bad]), MONTHS)


def test_months_are_validated():
    with pytest.raises(ValueError, match="months"):
        ts.from_hazard(np.array([0.01, 0.01]), np.array([6, 3]))
    with pytest.raises(ValueError, match="months"):
        ts.from_hazard(np.array([0.01, 0.01]), np.array([3, 3]))
    with pytest.raises(ValueError, match="months"):
        ts.from_hazard(np.array([0.01, 0.01]), np.array([3.5, 6.0]))
    with pytest.raises(ValueError, match="months"):
        ts.from_hazard(np.array([0.01, 0.01, 0.01]), MONTHS)


def test_from_marginal_and_cumulative_round_trip_and_refusals():
    a = ts.from_marginal(np.array([0.01, 0.0099]), MONTHS)
    np.testing.assert_allclose(a.hazard, [0.01, 0.01], atol=1e-12)
    b = ts.from_cumulative(np.array([0.01, 0.0199]), MONTHS)
    np.testing.assert_allclose(b.hazard, [0.01, 0.01], atol=1e-12)
    rng = np.random.default_rng(6)
    h = rng.uniform(0.001, 0.1, size=(5, 6))
    m = np.arange(1, 7) * 3
    base = ts.from_hazard(h, m)
    for rebuilt in (ts.from_marginal(base.marginal, m), ts.from_cumulative(base.cumulative, m)):
        for f in ("hazard", "survival", "marginal", "cumulative"):
            np.testing.assert_allclose(getattr(rebuilt, f), getattr(base, f), atol=1e-12)
    with pytest.raises(ValueError, match="marginal"):
        ts.from_marginal(np.array([0.6, 0.4]), MONTHS)
    with pytest.raises(ValueError, match="marginal"):
        ts.from_marginal(np.array([0.5, 0.6]), MONTHS)
    with pytest.raises(ValueError, match="marginal"):
        ts.from_marginal(np.array([0.1, 0.0]), MONTHS)
    with pytest.raises(ValueError, match="cumulative"):
        ts.from_cumulative(np.array([0.02, 0.01]), MONTHS)
    with pytest.raises(ValueError, match="cumulative"):
        ts.from_cumulative(np.array([0.02, 0.02]), MONTHS)
    with pytest.raises(ValueError, match="cumulative"):
        ts.from_cumulative(np.array([0.5, 1.0]), MONTHS)


def test_forward_pd_and_annualise_pd():
    c = ts.from_hazard(np.full(3, 0.01), np.array([3, 6, 9]))
    assert ts.forward_pd(c, 3, 9) == pytest.approx(1 - 0.99**2, abs=1e-12)
    assert ts.forward_pd(c, 3, 9) == pytest.approx(0.0199, abs=1e-12)
    # from month zero it is the cumulative PD
    assert ts.forward_pd(c, 0, 9) == pytest.approx(c.cumulative[-1], abs=1e-12)
    batch = ts.from_hazard(np.full((2, 3), 0.02), np.array([3, 6, 9]))
    assert ts.forward_pd(batch, 0, 6).shape == (2,)
    with pytest.raises(ValueError, match="end_month"):
        ts.forward_pd(c, 6, 6)
    with pytest.raises(ValueError, match="month"):
        ts.forward_pd(c, 3, 7)
    with pytest.raises(ValueError, match="month"):
        ts.forward_pd(c, -3, 6)
    # competing risks: default incidence given survival of both exits, never default-or-prepayment
    cr = ts.competing_risks(np.array([0.01, 0.01]), np.array([0.02, 0.02]), np.array([3, 6]))
    assert ts.forward_pd(cr, 0, 6) == pytest.approx(0.0197, abs=1e-12)
    assert abs(ts.forward_pd(cr, 0, 6) - 0.0591) > 0.03
    assert ts.forward_pd(cr, 3, 6) == pytest.approx(0.0097 / 0.97, abs=1e-12)
    # annualisation
    assert ts.annualise_pd(0.19, 2.0) == pytest.approx(0.1, abs=1e-12)
    p = np.array([0.01, 0.2, 0.7])
    np.testing.assert_allclose(ts.annualise_pd(p, 1.0), p, atol=1e-15)
    assert np.all(np.diff(ts.annualise_pd(np.linspace(0.01, 0.9, 20), 3.0)) > 0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="horizon_years"):
            ts.annualise_pd(0.1, bad)
    with pytest.raises(ValueError, match="pd"):
        ts.annualise_pd(1.0, 2.0)


def test_competing_risks_golden_and_limits():
    cr = ts.competing_risks(np.array([0.01, 0.01]), np.array([0.02, 0.02]), np.array([3, 6]))
    np.testing.assert_allclose(cr.survival_all, [0.97, 0.9409], atol=1e-12)
    np.testing.assert_allclose(cr.marginal_default, [0.01, 0.0097], atol=1e-12)
    np.testing.assert_allclose(cr.cumulative_default, [0.01, 0.0197], atol=1e-12)
    assert not isinstance(cr, ts.TermStructure)
    assert np.all(cr.cumulative_default < 1 - cr.survival_all)
    # no prepayment: the arrays equal the single-risk structure
    h = np.array([0.01, 0.02, 0.03])
    m = np.array([3, 6, 9])
    solo = ts.from_hazard(h, m)
    nop = ts.competing_risks(h, np.zeros(3), m)
    np.testing.assert_allclose(nop.survival_all, solo.survival, atol=1e-12)
    np.testing.assert_allclose(nop.cumulative_default, solo.cumulative, atol=1e-12)
    # prepayment never raises the default incidence
    more = ts.competing_risks(h, np.full(3, 0.05), m)
    assert np.all(more.cumulative_default <= nop.cumulative_default + 1e-15)
    with pytest.raises(ValueError, match="hazard_default"):
        ts.competing_risks(np.array([0.6]), np.array([0.5]), np.array([3]))
    with pytest.raises(ValueError, match="hazard_prepay"):
        ts.competing_risks(np.array([0.01]), np.array([-0.1]), np.array([3]))


def test_ttc_to_pit_goldens_and_invariants():
    pit = ts.ttc_to_pit(0.02, 0.12, np.array([1.0, 0.0, -2.0]))
    np.testing.assert_allclose(pit, [0.005255059, 0.014287387, 0.073424082], atol=1e-8)
    z = np.linspace(-4, 4, 41)
    assert np.all(np.diff(ts.ttc_to_pit(0.02, 0.12, z)) < 0)
    assert ts.ttc_to_pit(0.02, 0.12, 0.0) < 0.02
    # the average over the systematic factor is the TTC PD
    nodes, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / np.sqrt(2 * np.pi)
    assert np.sum(w * ts.ttc_to_pit(0.07, 0.2, nodes)) == pytest.approx(0.07, abs=1e-9)
    for pd, rho in ((0.0, 0.1), (1.0, 0.1), (0.02, 0.0), (0.02, 1.0)):
        with pytest.raises(ValueError):
            ts.ttc_to_pit(pd, rho, np.array([0.0]))


def test_implied_z_inverts_ttc_to_pit():
    assert ts.implied_z(np.array([0.014287387]), 0.02, 0.12) == pytest.approx(0.0, abs=1e-6)
    z = np.array([-3.0, -0.5, 0.0, 1.3, 2.2])
    back = ts.implied_z(ts.ttc_to_pit(0.03, 0.15, z), 0.03, 0.15)
    np.testing.assert_allclose(back, z, atol=1e-9)
    with pytest.raises(ValueError, match="default_rate"):
        ts.implied_z(np.array([0.0]), 0.02, 0.12)


Z3 = np.array([1.0, 0.0, -2.0])
W3 = np.array([0.25, 0.5, 0.25])


def test_weighted_pit_pd_non_linearity_golden():
    weighted = ts.weighted_pit_pd(0.02, 0.12, Z3, W3)
    at_mean = ts.ttc_to_pit(0.02, 0.12, float(np.sum(W3 * Z3)))
    assert weighted == pytest.approx(0.026813479, abs=1e-8)
    assert at_mean == pytest.approx(0.017997448, abs=1e-8)
    assert weighted / at_mean == pytest.approx(1.48985, abs=1e-5)
    # LGD 40 per cent: 1.07% against 0.72%
    assert 0.4 * weighted == pytest.approx(0.0107, abs=5e-5)
    assert 0.4 * at_mean == pytest.approx(0.0072, abs=5e-5)
    # Jensen in the convex regime (strongly adverse scenarios)
    rng = np.random.default_rng(1)
    for _ in range(25):
        z = rng.uniform(-3, 0.5, size=4)
        w = rng.dirichlet(np.ones(4))
        assert ts.weighted_pit_pd(0.03, 0.15, z, w) >= ts.ttc_to_pit(0.03, 0.15, float(w @ z)) - 1e-15
    # zero and maximal weights are valid
    assert ts.weighted_pit_pd(0.02, 0.12, Z3, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.014287387, abs=1e-8)


def test_gauss_hermite_recovers_ttc_pd():
    assert ts.gauss_hermite_pit_pd(0.02, 0.12, 64) == pytest.approx(0.02, abs=1e-9)
    for n in (32, 48, 64, 128):
        assert ts.gauss_hermite_pit_pd(0.02, 0.12, n) == pytest.approx(0.02, abs=1e-9)
    with pytest.raises(ValueError, match="n_nodes"):
        ts.gauss_hermite_pit_pd(0.02, 0.12, 1)


def test_weights_are_validated():
    bad = (
        np.array([0.5, 0.6]),  # sums to 1.1
        np.array([-0.1, 1.1]),  # outside [0, 1]
        np.array([1.0]),  # one scenario only (B5.5.42)
        np.array([0.5, 0.25, 0.25, 0.1]),  # wrong length for z
        np.array([0.5, np.nan]),
    )
    for w in bad:
        with pytest.raises(ValueError):
            ts.weighted_pit_pd(0.02, 0.12, np.array([0.0, 1.0]), w)
