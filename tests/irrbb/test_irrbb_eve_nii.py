"""RV-1013 to RV-1017: the repricing and duration gaps, the standardised EVE and NII, the CMF pair and Annex 1 measures,
the outlier tests (contract `engine-irrbb-api`, module eve_nii). Every golden is recomputed in the test by plain loops
and
`math`, and the offset rule is checked against a literal transcription of the formula of dossier 03 section F.2."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from riskvalidation.irrbb import eve_nii as e

MIDS = np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5,
                 17.5, 25.0])
SHOCKS = ("parallel-up", "parallel-down", "steepener", "flattener", "short-up", "short-down")


def arr(*x):
    return np.array(x, dtype=float)


def sheet():
    """The three-bucket sheet: +100 in bucket 4, -30 in bucket 9, -70 in bucket 16."""
    cf = np.zeros(19)
    cf[3], cf[8], cf[15] = 100.0, -30.0, -70.0
    return cf


def shock_bp(name, par, short, long_, t):
    a_s = math.exp(-t / 4.0)
    a_l = 1.0 - a_s
    return {
        "parallel-up": par, "parallel-down": -par, "short-up": short * a_s, "short-down": -short * a_s,
        "steepener": -0.65 * short * a_s + 0.9 * long_ * a_l, "flattener": 0.8 * short * a_s - 0.6 * long_ * a_l,
    }[name]


def eve_loop(cf, rate, mids):
    return sum(cf[k] * math.exp(-rate[k] * mids[k]) for k in range(len(mids)))


# ---------------------------------------------------------------------------------------------------- RV-1013


def test_repricing_gap_and_delta_nii_gap():
    a, liab = np.zeros(19), np.zeros(19)
    a[2] = 100.0
    assert e.repricing_gap(a, liab)[2] == 100.0 and e.repricing_gap(liab, a)[2] == -100.0
    rng = np.random.default_rng(1)
    a, liab = rng.uniform(0, 50, 19), rng.uniform(0, 50, 19)
    assert np.array_equal(e.repricing_gap(a, liab), -e.repricing_gap(liab, a))
    assert e.repricing_gap(a, liab).sum() == pytest.approx(a.sum() - liab.sum(), abs=1e-10)
    with pytest.raises(ValueError):
        e.repricing_gap(-a, liab)
    with pytest.raises(ValueError):
        e.repricing_gap(a[:5], liab)
    gap = np.zeros(19)
    gap[2] = 100.0
    assert e.delta_nii_gap(gap, MIDS, 100.0) == pytest.approx(-100 * 0.01 * (1 - 0.1667), abs=1e-12)
    assert e.delta_nii_gap(gap, MIDS, 100.0) == pytest.approx(-0.8333, abs=5e-5)
    g = rng.normal(0, 100, 19)
    ref = -sum(g[k] * 0.005 * (1 - MIDS[k]) for k in range(19) if MIDS[k] < 1)
    assert e.delta_nii_gap(g, MIDS, 50.0) == pytest.approx(ref, abs=1e-12)
    assert e.delta_nii_gap(g, MIDS, 0.0) == 0.0
    assert e.delta_nii_gap(g, MIDS, 100.0) == pytest.approx(2 * e.delta_nii_gap(g, MIDS, 50.0), abs=1e-12)
    far = np.zeros(19)
    far[6:] = 50.0
    assert e.delta_nii_gap(far, MIDS, 100.0) == 0.0
    with pytest.raises(ValueError):
        e.delta_nii_gap(g, MIDS[:5], 100.0)


def test_duration_gap_golden():
    a, liab = np.zeros(19), np.zeros(19)
    a[8], liab[5] = 100.0, 80.0
    mids = MIDS.copy()
    mids[8], mids[5] = 2.0, 1.0
    r = e.duration_gap(a, liab, np.zeros(19), mids)
    assert (r.asset_value, r.liability_value) == (100.0, 80.0)
    assert (r.asset_duration_years, r.liability_duration_years) == (2.0, 1.0)
    assert r.duration_gap_years == pytest.approx(2 - 0.8 * 1, abs=1e-12)
    assert r.equity_change_100bp_up == pytest.approx(-1.2, abs=1e-12)
    # at a positive rate the values are present values, and the first-order change matches a finite difference of the
    # equity
    rates = np.full(19, 0.03)
    r2 = e.duration_gap(a, liab, rates, mids)
    pv_a, pv_l = 100 * math.exp(-0.03 * 2), 80 * math.exp(-0.03 * 1)
    assert (r2.asset_value, r2.liability_value) == pytest.approx((pv_a, pv_l), abs=1e-12)
    eq = lambda d: 100 * math.exp(-(0.03 + d) * 2) - 80 * math.exp(-(0.03 + d) * 1)  # noqa: E731
    assert r2.equity_change_100bp_up == pytest.approx(eq(0.01) - eq(0.0), abs=2e-2)
    assert dataclasses.is_dataclass(r2)
    # no liabilities: the gap is the asset duration
    none = e.duration_gap(a, np.zeros(19), np.zeros(19), mids)
    assert none.duration_gap_years == none.asset_duration_years == 2.0
    with pytest.raises(ValueError):
        e.duration_gap(np.zeros(19), liab, np.zeros(19), mids)


# ---------------------------------------------------------------------------------------------------- RV-1014


def test_eve_golden_and_additivity():
    cf, flat = sheet(), np.full(19, 0.02)
    assert e.eve(cf, flat, MIDS) == pytest.approx(100 * math.exp(-0.02 * 0.375) - 30 * math.exp(-0.02 * 2.5)
                                                 - 70 * math.exp(-0.02 * 9.5), abs=1e-12)
    assert e.eve(cf, flat, MIDS) == pytest.approx(12.828783, abs=5e-7)
    assert e.eve(cf, np.zeros(19), MIDS) == 0.0
    rng = np.random.default_rng(4)
    items = rng.normal(0, 40, (6, 19))
    rates = rng.uniform(0.0, 0.06, 19)
    parts = sum(e.eve(row, rates, MIDS) for row in items)
    assert e.eve(items.sum(axis=0), rates, MIDS) == pytest.approx(parts, abs=1e-10)
    assert e.eve(items.sum(axis=0), rates, MIDS) == pytest.approx(eve_loop(items.sum(axis=0), rates, MIDS), abs=1e-10)
    assert e.eve(3.0 * cf, flat, MIDS) == pytest.approx(3.0 * e.eve(cf, flat, MIDS), abs=1e-12)
    with pytest.raises(ValueError):
        e.eve(cf, flat[:5], MIDS)
    with pytest.raises(ValueError):
        e.eve(cf, np.full(19, float("nan")), MIDS)


def test_delta_eve_six_shocks():
    cf, base = sheet(), np.full(19, 0.02)
    par, short, long_ = 200.0, 300.0, 225.0  # USD, SRP31.90 Table 2
    expected = {"parallel-up": -10.666989, "parallel-down": 12.828783, "steepener": -8.944982,
                "flattener": 6.076550, "short-up": -1.625807, "short-down": 1.702071}
    for name in SHOCKS:
        shocked = base + np.array([shock_bp(name, par, short, long_, t) for t in MIDS]) / 1e4
        got = e.delta_eve(cf, cf, base, shocked, MIDS)
        assert got == pytest.approx(expected[name], abs=5e-7), name
        assert got == pytest.approx(eve_loop(cf, base, MIDS) - eve_loop(cf, shocked, MIDS), abs=1e-10)
    # the add-on is added after the difference, and nothing moves when the shocked inputs equal the base
    assert e.delta_eve(cf, cf, base, base, MIDS) == 0.0
    assert e.delta_eve(cf, cf, base, base, MIDS, kao=2.5) == 2.5
    # shocked flows (a behavioural change) enter the shocked EVE only
    cf2 = cf.copy()
    cf2[3] += 10.0
    assert e.delta_eve(cf, cf2, base, base, MIDS) == pytest.approx(-10.0 * math.exp(-0.02 * 0.375), abs=1e-12)


def test_eve_risk_measure_and_addon():
    # the contract golden: the three rows [[3, -1], [2, 4], [-5, -2]] completed to the six shocks with rows of zeros
    m = np.vstack([np.array([[3.0, -1.0], [2.0, 4.0], [-5.0, -2.0]]), np.zeros((3, 2))])
    assert e.eve_risk_measure(m) == 6.0
    # a gain in one currency never offsets a loss in another: the all-gain matrix gives zero
    assert e.eve_risk_measure(-np.abs(m)) == 0.0
    assert e.eve_risk_measure(np.array([[1.0], [7.0], [-3.0], [0.0], [0.0], [0.0]])) == 7.0
    assert e.eve_risk_measure(np.full((6, 1), -7.0)) == 0.0
    rng = np.random.default_rng(8)
    big = rng.normal(0, 10, (6, 3))
    assert e.eve_risk_measure(big) == pytest.approx(max(0.0, *[sum(x for x in row if x > 0) for row in big]), abs=1e-12)
    assert e.eve_risk_measure(big) >= 0
    with pytest.raises(ValueError):
        e.eve_risk_measure(np.array([1.0, 2.0]))
    # a transposed (C, S) matrix is refused instead of giving another number silently; so is any row count other than 6
    assert e.eve_risk_measure(big) != e.eve_risk_measure(big.T) if big.T.shape[0] == 6 else True
    with pytest.raises(ValueError, match="six|6"):
        e.eve_risk_measure(big.T)
    for rows in (1, 3, 5, 7):
        with pytest.raises(ValueError):
            e.eve_risk_measure(np.ones((rows, 2)))
    assert e.automatic_option_addon(arr(3, 2), arr(1)) == 4.0
    assert e.automatic_option_addon(arr(), arr()) == 0.0
    assert e.automatic_option_addon(arr(1), arr(3, 2)) == -e.automatic_option_addon(arr(3, 2), arr(1))


def test_delta_nii_golden():
    cf = np.zeros(19)
    cf[4] = 100.0
    got = e.delta_nii(cf, np.full(19, 0.03), np.full(19, 0.06), MIDS)
    assert got == pytest.approx(100 * (0.03 - 0.06) * (1 - 0.625), abs=1e-12)
    assert e.delta_nii(cf, np.full(19, 0.03), np.full(19, 0.06), MIDS) == pytest.approx(-1.125, abs=1e-12)
    base = np.full(19, 0.03)
    assert e.delta_nii(cf, base, base, MIDS) == 0.0
    far = np.zeros(19)
    far[6:] = 100.0  # bucket 7 (1.25 years) and beyond do not contribute
    assert e.delta_nii(far, base, base + 0.02, MIDS) == 0.0
    rng = np.random.default_rng(6)
    flows = rng.normal(0, 50, 19)
    sh = base + rng.uniform(-0.01, 0.01, 19)
    ref = sum(flows[k] * (base[k] - sh[k]) * (1 - MIDS[k]) for k in range(19) if MIDS[k] < 1)
    assert e.delta_nii(flows, base, sh, MIDS) == pytest.approx(ref, abs=1e-12)
    assert e.delta_nii(2 * flows, base, sh, MIDS) == pytest.approx(2 * e.delta_nii(flows, base, sh, MIDS), abs=1e-12)


# ---------------------------------------------------------------------------------------------------- RV-1015


def cmf_ref(clp, clf, offset=True):
    """The formula of dossier 03 F.2, transcribed literally."""
    if offset and clp * clf < 0:
        if clp <= 0:
            return max(0.5 * clp + clf, 0.0)
        return max(clp + 0.5 * clf, 0.0)
    return max(clp, 0.0) + max(clf, 0.0)


def test_cmf_pair_eight_cases_both_signs():
    table = [(10, 20, 30), (-10, -20, 0), (30, -20, 20), (-20, 30, 20), (10, -40, 0), (-40, 10, 0), (0, 25, 25),
             (25, 0, 25)]
    for clp, clf, pair in table:
        assert e.cmf_pair(clp, clf) == pytest.approx(pair, abs=1e-12), (clp, clf)
        assert cmf_ref(clp, clf) == pytest.approx(pair, abs=1e-12)
    # the 50 percent applies to the gain leg, on both sign branches (the wrong branch would give 30 + 0.5 * -20 on the
    # first)
    assert e.cmf_pair(30, -20) == 30 - 10 and e.cmf_pair(-20, 30) == 30 - 10
    assert e.cmf_pair(30, -20) != 0.5 * 30 + -20
    # offset off: the positive parts add
    assert e.cmf_pair(30, -20, offset=False) == 30.0 and e.cmf_pair(-20, 30, offset=False) == 30.0
    assert e.cmf_pair(10, -40, offset=False) == 10.0
    # a different share
    assert e.cmf_pair(30, -20, share=0.25) == 30 - 5
    assert e.cmf_pair(-20, 30, share=0.25) == 30 - 5  # the share on the delta_clp <= 0 branch too
    assert e.cmf_pair(-40, 30, share=0.25) == max(0.25 * -40 + 30, 0.0)
    with pytest.raises(ValueError):
        e.cmf_pair(1.0, 1.0, share=1.5)
    with pytest.raises(ValueError):
        e.cmf_pair(float("nan"), 1.0)


def test_cmf_pair_properties():
    rng = np.random.default_rng(12)
    pairs = rng.normal(0, 30, (4000, 2))
    for clp, clf in pairs:
        on, off = e.cmf_pair(float(clp), float(clf)), e.cmf_pair(float(clp), float(clf), offset=False)
        assert on >= 0 and off >= 0 and off >= on - 1e-12
        assert on == pytest.approx(cmf_ref(float(clp), float(clf)), abs=1e-12)
        if clp > 0 and clf > 0:
            assert on == pytest.approx(e.cmf_pair(float(clf), float(clp)), abs=1e-12)
    for a, b in itertools.product((0.0, 25.0, -25.0), repeat=2):
        assert e.cmf_pair(a, b) == pytest.approx(cmf_ref(a, b), abs=1e-12)


# ---------------------------------------------------------------------------------------------------- RV-1016


def test_annex1_delta_nii_golden_and_charge_once():
    r = e.annex1_delta_nii(arr(30, -10), arr(-20, 5), -200.0)
    assert r.pairs.tolist() == [20.0, 0.0] and r.max_pair == 20.0
    assert r.indexation_charge == 5.0 and r.delta_nii == 25.0 and r.worst_shock == "parallel-up"
    assert r.delta_nii == r.max_pair + 0.025 * 200
    # the charge does not depend on the sign of the position and is added once, after the maximum
    assert e.annex1_delta_nii(arr(30, -10), arr(-20, 5), 200.0).delta_nii == 25.0
    both = e.annex1_delta_nii(arr(30, 30), arr(-20, -20), -200.0)
    assert both.delta_nii == 20.0 + 5.0 and both.worst_shock == "parallel-up"
    # a larger loss under the down shock is the worst one
    down = e.annex1_delta_nii(arr(-10, 30), arr(5, -20), 0.0)
    assert down.worst_shock == "parallel-down" and down.delta_nii == 20.0 and down.indexation_charge == 0.0
    # no loss in either shock: the maximum is zero, no worst shock, only the charge remains
    zero = e.annex1_delta_nii(arr(-10, -10), arr(-5, -5), 40.0)
    assert zero.max_pair == 0.0 and zero.worst_shock is None and zero.delta_nii == 1.0
    # the other currencies contribute their positive parts to each shock
    other = np.array([[4.0, -3.0], [1.0, 7.0]])  # (2, C): one row per shock, one column per currency
    o = e.annex1_delta_nii(arr(30, -10), arr(-20, 5), -200.0, other_currencies=other)
    assert o.max_pair == max(4.0 + 20.0, 1.0 + 7.0 + 0.0) == 24.0 and o.delta_nii == o.max_pair + 5.0
    assert o.pairs.tolist() == [20.0, 0.0]
    # offset off adds the positive parts
    off = e.annex1_delta_nii(arr(30, -10), arr(-20, 5), -200.0, offset=False)
    assert off.pairs.tolist() == [30.0, 5.0] and off.delta_nii == 35.0
    assert dataclasses.is_dataclass(r)
    with pytest.raises(ValueError):
        e.annex1_delta_nii(arr(1, 2, 3), arr(1, 2), 0.0)
    with pytest.raises(ValueError):
        e.annex1_delta_nii(arr(1, 2), arr(1, 2), 0.0, other_currencies=np.zeros((3, 2)))


def test_annex1_delta_eve_golden():
    r = e.annex1_delta_eve(arr(10, -20, 0, 5, -30, 40), arr(20, 30, -10, -2, 50, -15))
    assert r.pairs.tolist() == pytest.approx([30, 20, 0, 4, 35, 32.5], abs=1e-12)
    assert r.delta_eve == 35.0 and r.worst_shock == "short-up"
    assert e.annex1_delta_eve(arr(-1, -1, -1, -1, -1, -1), arr(-2, -2, -2, -2, -2, -2)).worst_shock is None
    assert e.annex1_delta_eve(arr(-1, -1, -1, -1, -1, -1), arr(-2, -2, -2, -2, -2, -2)).delta_eve == 0.0
    # offset off never gives less
    off = e.annex1_delta_eve(arr(10, -20, 0, 5, -30, 40), arr(20, 30, -10, -2, 50, -15), offset=False)
    assert off.delta_eve >= r.delta_eve and off.pairs.tolist() == [30, 30, 0, 5, 50, 40]
    # there is no indexation charge in the EVE measure; the other currencies add their positive parts
    other = np.full((6, 1), 2.0)
    assert e.annex1_delta_eve(arr(10, -20, 0, 5, -30, 40), arr(20, 30, -10, -2, 50, -15),
                              other_currencies=other).delta_eve == 37.0
    with pytest.raises(ValueError):
        e.annex1_delta_eve(arr(1, 2), arr(1, 2))


def test_indexation_charge():
    assert e.indexation_charge(-200.0) == 5.0 and e.indexation_charge(200.0) == 5.0 and e.indexation_charge(0.0) == 0.0
    assert e.indexation_charge(1000.0, rate=0.01) == pytest.approx(10.0, abs=1e-12)
    for x in (-1e6, -3.3, 0.0, 7.7, 1e6):
        assert e.indexation_charge(x) == e.indexation_charge(-x) >= 0
        assert e.indexation_charge(x) == pytest.approx(0.025 * abs(x), abs=1e-9)
    with pytest.raises(ValueError):
        e.indexation_charge(float("inf"))
    with pytest.raises(ValueError):
        e.indexation_charge(1.0, rate=-0.01)


# ---------------------------------------------------------------------------------------------------- RV-1017


def test_outlier_test_strict():
    hi = e.outlier_test(16.0, 100.0)
    assert (hi.ratio, hi.threshold, hi.breached) == (0.16, 0.15, True)
    eq = e.outlier_test(15.0, 100.0)
    assert (eq.ratio, eq.breached) == (0.15, False)
    assert not e.outlier_test(-5.0, 100.0).breached and not e.outlier_test(0.0, 100.0).breached
    assert e.outlier_test(16.0, 100.0, threshold=0.20).breached is False
    assert e.outlier_test(20.5, 100.0, threshold=0.20).breached is True
    assert dataclasses.is_dataclass(hi)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            e.outlier_test(1.0, bad)


def test_prioritised_bank_golden():
    r = e.prioritised_bank(6.0, 10.0, 100.0, 40.0)
    assert [t["id"] for t in r.tests] == ["nii-tier1", "eve-tier1", "nii-margin"]
    assert [t["ratio"] for t in r.tests] == pytest.approx([0.06, 0.10, 0.15], abs=1e-12)
    assert [t["threshold"] for t in r.tests] == [0.05, 0.15, 0.18]
    assert [t["breached"] for t in r.tests] == [True, False, False] and r.prioritised is True
    assert [t["base"] for t in r.tests] == [100.0, 100.0, 40.0]
    assert [t["measure"] for t in r.tests] == [6.0, 10.0, 6.0]
    # strict inequalities: exactly on a threshold does not breach
    assert e.prioritised_bank(18.0, 0.0, 1000.0, 100.0).tests[2]["breached"] is False  # 18 / 100 = 0.18, not above
    exact = e.prioritised_bank(5.0, 15.0, 100.0, 1000.0)  # 0.05 and 0.15 exactly
    assert [t["breached"] for t in exact.tests] == [False, False, False] and exact.prioritised is False
    assert e.prioritised_bank(1.0, 16.0, 100.0, 1000.0).prioritised is True
    assert e.prioritised_bank(1.0, 1.0, 100.0, 4.0).prioritised is True
    for args in ((1, 1, 0.0, 10.0), (1, 1, 100.0, 0.0), (1, 1, -1.0, 10.0)):
        with pytest.raises(ValueError):
            e.prioritised_bank(*args)
