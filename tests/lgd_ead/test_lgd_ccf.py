"""RV-612 to RV-615: CCF and EAD (dossier 04 G.1 to G.3), in the EU convention (the share of the undrawn amount drawn at
default), the treatments of a CCF outside [0, 1], the two EAD models (contract `engine-lgd-ead-api`, module ccf)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.lgd_ead import ccf as c


def a(*v):
    return np.array(v, dtype=float)


def facility():
    """The constructed facility of the contract: limit 1000, EAD 900 at default month 21."""
    drawn = np.full(22, 100.0)
    drawn[[9, 12, 15, 18]] = [420, 500, 560, 640]
    return drawn, np.full(22, 1000.0), 900.0


# ---------------------------------------------------------------- RV-612


def test_reference_indices():
    assert c.reference_indices("fixed-horizon", 21).tolist() == [9]
    assert c.reference_indices("cohort", 21, cohort_start_index=12).tolist() == [12]
    out = c.reference_indices("variable-horizon", 21)
    assert out.tolist() == [9, 12, 15, 18] and out.dtype == np.int64
    assert c.reference_indices("variable-horizon", 30, horizon=6, n_reference=3, step=2).tolist() == [24, 26, 28]
    assert c.reference_indices("fixed-horizon", 30, horizon=6).tolist() == [24]
    with pytest.raises(ValueError, match="cohort_start_index"):
        c.reference_indices("cohort", 21)
    with pytest.raises(ValueError, match="negative|precedes"):
        c.reference_indices("fixed-horizon", 5)  # 5 - 12 < 0
    with pytest.raises(ValueError, match="negative|precedes"):
        c.reference_indices("variable-horizon", 10)
    with pytest.raises(ValueError, match="design"):
        c.reference_indices("moral", 21)
    with pytest.raises(ValueError, match="before"):
        c.reference_indices("cohort", 21, cohort_start_index=22)  # the reference must precede the default
    with pytest.raises(ValueError):
        c.reference_indices("fixed-horizon", 21, horizon=0)


def test_ccf_observations_golden_and_undefined():
    drawn, limit, ead = facility()
    idx = np.array([9, 12, 15, 18])
    ccf, undrawn = c.ccf_observations(drawn, limit, ead, idx)
    np.testing.assert_allclose(undrawn, [580, 500, 440, 360])
    np.testing.assert_allclose(ccf, [0.827586, 0.8, 0.772727, 0.722222], atol=5e-7)
    np.testing.assert_allclose(ccf, (ead - drawn[idx]) / (limit[idx] - drawn[idx]), atol=1e-15)
    # the fixed-horizon design reads 0.827586 and the cohort design 0.8
    assert c.ccf_observations(drawn, limit, ead, np.array([9]))[0][0] == pytest.approx(0.827586, abs=5e-7)
    assert c.ccf_observations(drawn, limit, ead, np.array([12]))[0][0] == pytest.approx(0.8, abs=1e-15)
    # EU convention, not the US one: ead / drawn would be 2.142857 at month 9
    assert abs(ccf[0] - ead / drawn[9]) > 1.0
    # a fully drawn or overdrawn line: undefined (NaN), never 0 and never 1
    d2 = drawn.copy()
    d2[9] = 1000.0
    d2[12] = 1100.0
    ccf2, und2 = c.ccf_observations(d2, limit, ead, idx)
    assert np.isnan(ccf2[0]) and np.isnan(ccf2[1]) and np.isfinite(ccf2[2])
    np.testing.assert_allclose(und2[:2], [0.0, -100.0])
    # outside [0, 1] is returned as it comes
    ccf3, _ = c.ccf_observations(drawn, limit, 1200.0, np.array([9]))
    assert ccf3[0] > 1.0
    ccf4, _ = c.ccf_observations(drawn, limit, 300.0, np.array([9]))
    assert ccf4[0] < 0.0
    with pytest.raises(ValueError, match="reference_index"):
        c.ccf_observations(drawn, limit, ead, np.array([22]))
    with pytest.raises(ValueError, match="reference_index"):
        c.ccf_observations(drawn, limit, ead, np.array([-1]))
    with pytest.raises(ValueError, match="same length"):
        c.ccf_observations(drawn, limit[:5], ead, idx)
    with pytest.raises(ValueError, match="non-finite"):
        c.ccf_observations(drawn, limit, float("nan"), idx)


def test_aggregate_ccf():
    drawn, limit, ead = facility()
    ccf, undrawn = c.ccf_observations(drawn, limit, ead, np.array([9, 12, 15, 18]))
    assert c.aggregate_ccf(ccf, undrawn, "mean") == pytest.approx(0.780634, abs=5e-7)
    assert c.aggregate_ccf(ccf, undrawn, "pooled") == pytest.approx(1480 / 1880, abs=1e-12)
    assert c.aggregate_ccf(ccf, undrawn, "pooled") == pytest.approx(0.787234, abs=5e-7)
    # the pooled ratio equals sum(EAD - E) / sum(L - E)
    assert c.aggregate_ccf(ccf, undrawn, "pooled") == pytest.approx((4 * ead - (420 + 500 + 560 + 640)) / undrawn.sum())
    # NaN entries are excluded
    cn = np.append(ccf, np.nan)
    un = np.append(undrawn, 0.0)
    assert c.aggregate_ccf(cn, un, "mean") == pytest.approx(0.780634, abs=5e-7)
    assert c.aggregate_ccf(cn, un, "pooled") == pytest.approx(0.787234, abs=5e-7)
    # not clipped
    assert c.aggregate_ccf(a(1.5, 1.3), a(10, 10), "mean") == pytest.approx(1.4)
    with pytest.raises(ValueError, match="finite"):
        c.aggregate_ccf(a(np.nan, np.nan), a(0, 0), "mean")
    with pytest.raises(ValueError, match="aggregator"):
        c.aggregate_ccf(ccf, undrawn, "median")
    with pytest.raises(ValueError, match="same"):
        c.aggregate_ccf(ccf, undrawn[:2], "mean")


# ---------------------------------------------------------------- RV-613


def test_treat_ccf():
    x = a(-0.1, 0.0, 0.2, 0.6, 1.3)
    none = c.treat_ccf(x, "none")
    assert isinstance(none, c.TreatedCcf) and dataclasses.is_dataclass(none)
    assert none.values.mean() == pytest.approx(0.4) and none.n_dropped == 0
    cen = c.treat_ccf(x, "censor")
    np.testing.assert_allclose(cen.values, [0.0, 0.2, 0.6])
    assert cen.values.mean() == pytest.approx(0.266667, abs=5e-7) and cen.n_dropped == 2
    fc = c.treat_ccf(x, "floor-cap")
    np.testing.assert_allclose(fc.values, [0.0, 0.0, 0.2, 0.6, 1.0])
    assert fc.values.mean() == pytest.approx(0.36) and fc.n_dropped == 0
    bs = c.treat_ccf(x, "bounded-scale")
    np.testing.assert_allclose(bs.values, [0.0, 0.0, 0.2, 0.6, 1.0])
    for r in (none, cen, fc, bs):  # counted on the input whatever the treatment
        assert (r.n_below_zero, r.n_above_one) == (1, 1)
    # NaN entries are dropped first and counted, and never counted as outside [0, 1]
    nn = c.treat_ccf(a(np.nan, -0.5, 0.5), "none")
    np.testing.assert_allclose(nn.values, [-0.5, 0.5])
    assert nn.n_dropped == 1 and nn.n_below_zero == 1 and nn.n_above_one == 0
    assert c.treat_ccf(a(np.nan, -0.5, 1.5, 0.5), "censor").n_dropped == 3
    with pytest.raises(ValueError, match="treatment"):
        c.treat_ccf(x, "winsorise")
    with pytest.raises(ValueError, match="non-finite"):
        c.treat_ccf(a(0.1, np.inf), "none")
    assert c.treat_ccf(a(), "none").values.size == 0


# ---------------------------------------------------------------- RV-614


def test_ead_from_ccf_and_round_trip():
    assert c.ead_from_ccf(a(600), a(1000), a(0.75))[0] == pytest.approx(900.0, abs=1e-12)
    # not clipped: a CCF outside [0, 1] gives an EAD below the drawn amount or above the limit
    out = c.ead_from_ccf(a(600, 600), a(1000, 1000), a(-0.5, 1.5))
    np.testing.assert_allclose(out, [400.0, 1200.0])
    drawn, limit, ead = facility()
    ccf, _ = c.ccf_observations(drawn, limit, ead, np.array([9, 12, 15, 18]))
    back = c.ead_from_ccf(drawn[[9, 12, 15, 18]], limit[[9, 12, 15, 18]], ccf)
    np.testing.assert_allclose(back, [900.0] * 4, atol=1e-9)
    with pytest.raises(ValueError, match="same"):
        c.ead_from_ccf(a(1, 2), a(3), a(0.5))


def test_to_us_terms_keeps_the_conventions_apart():
    leq, ccf_us, eadf = c.to_us_terms(a(600), a(1000), a(900))
    assert (leq[0], ccf_us[0], eadf[0]) == (pytest.approx(0.75), pytest.approx(1.5), pytest.approx(0.9))
    # the EU CCF is the LEQ, equal to the observation function, and differs from the US CCF
    drawn, limit, ead = facility()
    eu, _ = c.ccf_observations(drawn, limit, ead, np.array([12]))
    l2, us2, _ = c.to_us_terms(drawn[[12]], limit[[12]], a(ead))
    assert l2[0] == pytest.approx(eu[0], abs=1e-15) and abs(us2[0] - eu[0]) > 0.5
    # undefined entries are NaN
    leq, ccf_us, eadf = c.to_us_terms(a(1000, 0), a(1000, 1000), a(900, 500))
    assert np.isnan(leq[0]) and np.isfinite(ccf_us[0]) and np.isfinite(eadf[0])
    assert np.isnan(ccf_us[1]) and np.isfinite(leq[1])


def test_small_undrawn_rule():
    sw, ratio = c.small_undrawn_rule(a(980, 600), a(1000, 1000), a(990, 900), 50.0)
    assert sw.tolist() == [True, False] and sw.dtype == bool
    np.testing.assert_allclose(ratio, [0.99, 0.75])
    # the threshold is a parameter: the same facility is not switched at a threshold of 10
    sw2, ratio2 = c.small_undrawn_rule(a(980), a(1000), a(990), 10.0)
    assert sw2.tolist() == [False] and ratio2[0] == pytest.approx(0.5)
    # strict: an undrawn amount equal to the threshold is not switched
    assert c.small_undrawn_rule(a(950), a(1000), a(990), 50.0)[0].tolist() == [False]
    # a fully drawn line is switched (undrawn 0 < threshold), so the ratio is defined
    sw3, r3 = c.small_undrawn_rule(a(1000), a(1000), a(1000), 50.0)
    assert sw3.tolist() == [True] and r3[0] == pytest.approx(1.0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="threshold"):
            c.small_undrawn_rule(a(1), a(2), a(1), bad)


def test_ccf_bias():
    assert c.ccf_bias(0.83, 0.60) == pytest.approx(0.23, abs=1e-12)
    assert c.ccf_bias(0.5, 0.6) == pytest.approx(-0.1, abs=1e-12)
    assert c.ccf_bias(1.4, 0.6) == pytest.approx(0.8, abs=1e-12)  # not clipped
    with pytest.raises(ValueError):
        c.ccf_bias(float("nan"), 0.6)


# ---------------------------------------------------------------- RV-615


def test_fit_direct_ead_golden_and_clip_counts():
    X = np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=float)
    fit = c.fit_direct_ead(X, a(0, 500, 500, 1000), a(1000, 1000, 1000, 1000))
    assert isinstance(fit, c.EadFit) and fit.kind == "direct-ead"
    np.testing.assert_allclose(fit.coefficients, [-1.098612, 2.197225], atol=5e-7)
    assert fit.extras == {"n_clipped_below": 0, "n_clipped_above": 0}
    # an overlimit EAD is clipped to the limit for the fit and counted, never silently
    fit2 = c.fit_direct_ead(X, a(0, 500, 500, 1300), a(1000, 1000, 1000, 1000))
    assert fit2.extras["n_clipped_above"] == 1 and fit2.extras["n_clipped_below"] == 0
    np.testing.assert_allclose(fit2.coefficients, fit.coefficients, atol=1e-6)
    fit3 = c.fit_direct_ead(X, a(-20, 500, 500, 1000), a(1000, 1000, 1000, 1000))
    assert fit3.extras["n_clipped_below"] == 1
    with pytest.raises(ValueError, match="limit"):
        c.fit_direct_ead(X, a(0, 500, 500, 1000), a(1000, 1000, 1000, 0))
    with pytest.raises(ValueError, match="non-finite"):
        c.fit_direct_ead(X, a(0, 500, np.nan, 1000), a(1000, 1000, 1000, 1000))


def test_fit_utilisation_change_and_predict_ead():
    X = np.array([[1], [1], [1]], dtype=float)
    fit = c.fit_utilisation_change(X, a(400, 500, 600), a(700, 800, 900), a(1000, 1000, 1000))
    assert fit.kind == "utilisation-change"
    np.testing.assert_allclose(fit.coefficients, [0.3], atol=1e-12)
    assert c.predict_ead(fit, np.array([[1.0]]), a(600), a(1000))[0] == pytest.approx(900.0, abs=1e-9)
    # not clipped: a large fitted change can exceed the limit
    over = c.fit_utilisation_change(X, a(400, 500, 600), a(1400, 1500, 1600), a(1000, 1000, 1000))
    assert c.predict_ead(over, np.array([[1.0]]), a(600), a(1000))[0] > 1000.0
    # the direct prediction: limit times the fitted EADF, inside [0, limit]
    X4 = np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=float)
    d = c.fit_direct_ead(X4, a(0, 500, 500, 1000), a(1000, 1000, 1000, 1000))
    assert c.predict_ead(d, np.array([[1.0, 1.0]]), a(600), a(1000))[0] == pytest.approx(750.0, abs=1e-6)
    assert c.predict_ead(d, np.array([[1.0, 0.0]]), a(100), a(2000))[0] == pytest.approx(500.0, abs=1e-6)
    far = c.predict_ead(d, np.array([[1.0, 40.0], [1.0, -40.0]]), a(0, 0), a(1000, 1000))
    assert np.all((far >= 0) & (far <= 1000))
    with pytest.raises(ValueError, match="limit_ref"):
        c.fit_utilisation_change(X, a(400, 500, 600), a(700, 800, 900), a(1000, 0, 1000))
    with pytest.raises(ValueError, match="columns"):
        c.predict_ead(d, np.array([[1.0]]), a(1), a(1))
