"""RV-1010 to RV-1012: the 19 buckets, the slotting of cash flows, the prepayment and term-deposit multipliers, SMM
and CPR,
the PSA curve (contract `engine-irrbb-api`, module slotting). The expected values are recomputed with `math` and plain
loops."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.irrbb import slotting as sl

MIDS = [0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5,
        25.0]
SHOCKS = ("parallel-up", "parallel-down", "steepener", "flattener", "short-up", "short-down")


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-1010


def test_bucket_grid_midpoints_and_edges():
    g = sl.bucket_grid()
    assert g.midpoints.tolist() == MIDS
    assert len(g.labels) == 19 and g.labels[0] == "overnight" and g.labels[-1] == "over 20Y"
    edges = [0, 1 / 360, 1 / 12, 3 / 12, 6 / 12, 9 / 12, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    assert g.lower.tolist() == pytest.approx(edges, abs=1e-15)
    assert g.upper[:-1].tolist() == g.lower[1:].tolist()
    assert math.isinf(g.upper[-1]) and g.lower.shape == (19,) and g.upper.shape == (19,)
    # for buckets 3 to 18 the printed midpoint is the middle of the bucket (bucket 2 and 19 use their own conventions)
    for k in range(2, 18):
        assert g.midpoints[k] == pytest.approx((g.lower[k] + g.upper[k]) / 2, abs=6e-5)
    assert dataclasses.is_dataclass(g)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.labels = ()  # type: ignore[misc]
    # the grid returned is a copy: editing it does not change the next call
    g.midpoints[0] = 99.0
    assert sl.bucket_grid().midpoints[0] == 0.0028


def test_bucket_of_goldens_and_monotone():
    got = [sl.bucket_of(t) for t in (0, 0.5, 0.75, 20, 20.5, 25)]
    assert got == [1, 4, 5, 18, 19, 19]
    g = sl.bucket_grid()
    # the intervals are (lower, upper]: an upper edge belongs to its own bucket, just beyond it to the next
    for k in range(18):
        assert sl.bucket_of(float(g.upper[k]), g) == k + 1
        assert sl.bucket_of(float(g.upper[k]) + 1e-9, g) == k + 2
    ts = np.linspace(0, 30, 3001)
    b = [sl.bucket_of(float(t)) for t in ts]
    assert all(x <= y for x, y in zip(b, b[1:], strict=False)) and min(b) == 1 and max(b) == 19
    with pytest.raises(ValueError):
        sl.bucket_of(-0.001)
    with pytest.raises(ValueError):
        sl.bucket_of(float("nan"))


def test_slot_cash_flows_golden_and_additive():
    flows = sl.slot_cash_flows(arr(0.5, 2.5, 9.5), arr(100, -30, -70))
    expect = np.zeros(19)
    expect[3], expect[8], expect[15] = 100, -30, -70
    assert np.array_equal(flows, expect)
    assert flows.sum() == 0.0
    rng = np.random.default_rng(2)
    t1, a1 = rng.uniform(0, 30, 40), rng.normal(0, 100, 40)
    t2, a2 = rng.uniform(0, 30, 25), rng.normal(0, 100, 25)
    both = sl.slot_cash_flows(np.r_[t1, t2], np.r_[a1, a2])
    assert np.allclose(both, sl.slot_cash_flows(t1, a1) + sl.slot_cash_flows(t2, a2), atol=1e-12)
    assert both.sum() == pytest.approx(np.r_[a1, a2].sum(), abs=1e-10)
    ref = np.zeros(19)
    for t, a in zip(np.r_[t1, t2], np.r_[a1, a2], strict=True):
        ref[sl.bucket_of(float(t)) - 1] += a
    assert np.allclose(both, ref, atol=1e-12)
    assert np.array_equal(sl.slot_cash_flows(np.array([]), np.array([])), np.zeros(19))
    t_in, a_in = arr(0.5, 2.5), arr(1, 2)
    sl.slot_cash_flows(t_in, a_in)
    assert t_in.tolist() == [0.5, 2.5] and a_in.tolist() == [1, 2]
    with pytest.raises(ValueError):
        sl.slot_cash_flows(arr(0.5, 2.5), arr(1))
    with pytest.raises(ValueError):
        sl.slot_cash_flows(arr(-1.0), arr(1))


# ---------------------------------------------------------------------------------------------------- RV-1011


def wam_of_core(r, balance):
    """The weighted average maturity of the core part (everything beyond bucket 1) from the slotted flows."""
    mids = np.array(MIDS)
    core = r.flows[1:]
    return float(np.sum(core * mids[1:]) / np.sum(core)) if core.sum() > 0 else float("nan")


def test_nmd_split_golden_caps_and_reporting():
    r = sl.nmd_split(100.0, "retail-transactional", 0.9, 5.0)
    expect = np.zeros(19)
    expect[0], expect[10], expect[11] = 10.0, 45.0, 45.0
    assert np.allclose(r.flows, expect, atol=1e-12) and r.cap_applied is False
    assert (r.requested_core_share, r.core_share_cap, r.core_share_used) == (0.9, 0.9, 0.9)
    assert (r.requested_core_maturity_years, r.core_maturity_cap_years, r.core_maturity_used_years) == (5.0, 5.0, 5.0)
    # wholesale 0.6 against a 0.5 cap: cut and reported with the standardised treatment, kept without it
    cut = sl.nmd_split(100.0, "wholesale", 0.6, 4.0, enforce_caps=True)
    assert cut.cap_applied is True and cut.core_share_used == 0.5 and cut.requested_core_share == 0.6
    assert cut.core_share_cap == 0.5 and cut.core_maturity_used_years == 4.0
    assert cut.flows[0] == pytest.approx(50.0, abs=1e-12)
    kept = sl.nmd_split(100.0, "wholesale", 0.6, 4.0, enforce_caps=False)
    assert kept.cap_applied is False and kept.core_share_used == 0.6 and kept.core_share_cap == 0.5
    assert kept.flows[0] == pytest.approx(40.0, abs=1e-12)
    # a maturity above its cap is cut too, and either cut sets the flag
    mat_cut = sl.nmd_split(100.0, "retail-non-transactional", 0.5, 6.0)
    assert mat_cut.cap_applied is True and mat_cut.core_maturity_used_years == 4.5 and mat_cut.core_share_used == 0.5
    assert mat_cut.requested_core_maturity_years == 6.0 and mat_cut.core_maturity_cap_years == 4.5
    # exactly on a cap is within it
    assert sl.nmd_split(100.0, "wholesale", 0.5, 4.0).cap_applied is False
    # the caps of Table 4, by category and by Annex 1 alias
    caps = {"retail-transactional": (0.9, 5.0), "retail-non-transactional": (0.7, 4.5), "wholesale": (0.5, 4.0)}
    for cat, (cs, cm) in caps.items():
        r = sl.nmd_split(10.0, cat, 1.0, 100.0)
        assert (r.core_share_cap, r.core_maturity_cap_years) == (cs, cm)
        assert (r.core_share_used, r.core_maturity_used_years) == (cs, cm)
        assert r.cap_applied is True
    aliases = (("retail-operational", "retail-transactional"), ("retail-non-operational", "retail-non-transactional"))
    for alias, cat in aliases:
        a, b = sl.nmd_split(50.0, alias, 0.65, 3.3), sl.nmd_split(50.0, cat, 0.65, 3.3)
        assert np.array_equal(a.flows, b.flows) and a.core_share_cap == b.core_share_cap
    # the total is the balance, the non-core part is in bucket 1 and the core keeps its weighted average maturity
    rng = np.random.default_rng(7)
    for _ in range(200):
        bal, share, m = float(rng.uniform(0, 1000)), float(rng.uniform(0, 1)), float(rng.uniform(0.01, 5.0))
        r = sl.nmd_split(bal, "retail-transactional", share, m, enforce_caps=False)
        assert r.flows.sum() == pytest.approx(bal, abs=1e-9) and np.all(r.flows >= -1e-12)
        if m >= 0.0417 and share * bal > 1e-9:
            assert wam_of_core(r, bal) == pytest.approx(m, abs=1e-9)
    # a maturity beyond the last midpoint without caps goes to the last bucket
    far = sl.nmd_split(100.0, "wholesale", 1.0, 40.0, enforce_caps=False)
    assert far.flows[18] == 100.0 and far.cap_applied is False
    with pytest.raises(ValueError):
        sl.nmd_split(100.0, "corporate", 0.5, 2.0)
    for bad in ((-1.0, 0.5, 2.0), (100.0, 1.5, 2.0), (100.0, -0.1, 2.0), (100.0, 0.5, -1.0)):
        with pytest.raises(ValueError):
            sl.nmd_split(bad[0], "wholesale", bad[1], bad[2])


# ---------------------------------------------------------------------------------------------------- RV-1012


def test_prepayment_and_term_deposit_multipliers():
    assert sl.prepayment_cpr("parallel-up", 0.05) == pytest.approx(0.04, abs=1e-15)
    assert sl.prepayment_cpr("parallel-down", 0.05) == pytest.approx(0.06, abs=1e-15)
    assert sl.prepayment_cpr("parallel-down", 0.9) == 1.0
    gamma = dict(zip(SHOCKS, (0.8, 1.2, 0.8, 1.2, 0.8, 1.2), strict=True))
    u = dict(zip(SHOCKS, (1.2, 0.8, 0.8, 1.2, 1.2, 0.8), strict=True))
    for name in SHOCKS:
        for x in (0.0, 0.05, 0.5, 0.9, 1.0):
            assert sl.prepayment_cpr(name, x) == pytest.approx(min(1.0, gamma[name] * x), abs=1e-15)
            assert sl.term_deposit_redemption_rate(name, x) == pytest.approx(min(1.0, u[name] * x), abs=1e-15)
            assert 0.0 <= sl.prepayment_cpr(name, x) <= 1.0
    assert sl.term_deposit_redemption_rate("parallel-up", 0.05) == pytest.approx(0.06, abs=1e-15)
    assert sl.term_deposit_redemption_rate("parallel-down", 0.05) == pytest.approx(0.04, abs=1e-15)
    for fn in (sl.prepayment_cpr, sl.term_deposit_redemption_rate):
        with pytest.raises(ValueError):
            fn("twist", 0.05)
        with pytest.raises(ValueError):
            fn("parallel-up", 1.5)
        with pytest.raises(ValueError):
            fn("parallel-up", -0.1)


def test_smm_cpr_roundtrip():
    assert sl.smm_from_cpr(0.06) == pytest.approx(1 - (1 - 0.06) ** (1 / 12), abs=1e-15)
    assert sl.smm_from_cpr(0.06) == pytest.approx(0.005143, abs=5e-7)
    assert sl.cpr_from_smm(0.005143) == pytest.approx(0.06, abs=1e-6)
    assert sl.smm_from_cpr(0.0) == 0.0 and sl.smm_from_cpr(1.0) == 1.0
    assert sl.cpr_from_smm(0.0) == 0.0 and sl.cpr_from_smm(1.0) == 1.0
    for c in np.linspace(0, 1, 41):
        assert sl.cpr_from_smm(sl.smm_from_cpr(float(c))) == pytest.approx(c, abs=1e-12)
        # twelve months of the monthly rate compound to the annual one (independent route: survival product)
        surv = 1.0
        for _ in range(12):
            surv *= 1 - sl.smm_from_cpr(float(c))
        assert 1 - surv == pytest.approx(c, abs=1e-12)
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            sl.smm_from_cpr(bad)
        with pytest.raises(ValueError):
            sl.cpr_from_smm(bad)


def test_psa_cpr_goldens():
    assert sl.psa_cpr(1) == pytest.approx(0.002, abs=1e-15)
    assert sl.psa_cpr(30) == pytest.approx(0.06, abs=1e-15)
    assert sl.psa_cpr(60) == pytest.approx(0.06, abs=1e-15)
    assert sl.psa_cpr(20, 150.0) == pytest.approx(0.06, abs=1e-15)
    assert sl.psa_cpr(40, 150.0) == pytest.approx(0.09, abs=1e-15)
    months = np.arange(1, 121)
    curve = sl.psa_cpr(months)
    assert isinstance(curve, np.ndarray) and np.all(np.diff(curve) >= 0) and np.all(curve[29:] == 0.06)
    assert np.allclose(curve[:30], 0.002 * months[:30], atol=1e-15)
    # linear in the PSA percentage before the cap of the benchmark
    assert np.allclose(sl.psa_cpr(months[:20], 200.0), 2 * sl.psa_cpr(months[:20]), atol=1e-15)
    with pytest.raises(ValueError):
        sl.psa_cpr(0)
    with pytest.raises(ValueError):
        sl.psa_cpr(5, -10.0)
