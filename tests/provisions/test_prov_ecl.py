"""RV-509 to RV-512: discounting, the scenario ECL, probability weighting, Stage 3, POCI, the simplified matrix,
commitments and the portfolio allowance (dossier 02 B.3, B.4, B.5, B.7; the contract's golden values)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.provisions import ecl
from riskvalidation.provisions import term_structure as ts
from riskvalidation.provisions.overlays import Overlay


def overlay(oid="OVL-1", amount=5.0, stage="stage1", **kw) -> Overlay:
    base = {
        "id": oid, "rationale_en": "r", "rationale_es": "r", "method": "expert", "target": "pd", "segment_id": "s1",
        "stage": stage, "amount": amount, "staging_effect": "none", "review_date": "2020-12-31", "currency": "USD",
    }
    base.update(kw)
    return Overlay(**base)


def test_discount_factors_golden_and_zero_eir():
    df = ecl.discount_factors(np.array([0.06]), np.array([3, 6]))
    np.testing.assert_allclose(df[0], [0.992742848, 0.978386160], atol=1e-9)
    np.testing.assert_allclose(df[0], [1.06**-0.125, 1.06**-0.375], atol=1e-15)
    assert df.shape == (1, 2)
    end = ecl.discount_factors(np.array([0.06]), np.array([3, 6]), "end-of-period")
    np.testing.assert_allclose(end[0], [1.06**-0.25, 1.06**-0.5], atol=1e-15)
    assert np.all(end < df)
    zero = ecl.discount_factors(np.array([0.0, 0.0]), np.array([3, 6, 12]))
    assert np.all(zero == 1.0)
    many = ecl.discount_factors(np.array([0.01, 0.05, 0.2]), np.arange(1, 25) * 3)
    assert np.all((many > 0) & (many <= 1)) and np.all(np.diff(many, axis=1) < 0)
    # unequal periods: mid-period is the midpoint of the period
    un = ecl.discount_factors(np.array([0.1]), np.array([3, 12]))
    np.testing.assert_allclose(un[0], [1.1 ** (-1.5 / 12), 1.1 ** (-7.5 / 12)], atol=1e-15)
    assert ecl.discount_factors(np.array([]), np.array([3, 6])).shape == (0, 2)
    with pytest.raises(ValueError, match="eir"):
        ecl.discount_factors(np.array([-0.01]), np.array([3]))
    with pytest.raises(ValueError, match="convention"):
        ecl.discount_factors(np.array([0.01]), np.array([3]), "start")


def test_ecl_scenario_golden_and_properties():
    q = np.array([[0.01, 0.01]])
    months = np.array([6, 12])
    kw = {"lgd": 0.4, "ead": 100000.0, "eir": np.array([0.0]), "months": months}
    assert ecl.ecl_scenario(q, horizon_months=12, **kw)[0] == pytest.approx(800.0, abs=1e-9)
    assert ecl.ecl_scenario(q, horizon_months=6, **kw)[0] == pytest.approx(400.0, abs=1e-9)
    # a horizon beyond the last period is the lifetime ECL
    assert ecl.ecl_scenario(q, horizon_months=60, **kw)[0] == pytest.approx(800.0, abs=1e-9)
    with pytest.raises(ValueError, match="horizon_months"):
        ecl.ecl_scenario(q, horizon_months=9, **kw)
    # discounting, linearity and monotonicity
    rng = np.random.default_rng(2)
    ts_ = ts.from_hazard(rng.uniform(0.005, 0.03, size=(4, 8)), np.arange(1, 9) * 3)
    eir = np.array([0.02, 0.05, 0.0, 0.1])
    base = {"marginal_pd": ts_.marginal, "lgd": np.array([0.3, 0.4, 0.5, 0.6]), "ead": np.array([10.0, 20.0, 30.0, 40.0]),
            "eir": eir, "months": ts_.months}
    prev = np.zeros(4)
    for h in (3, 6, 12, 24):
        cur = ecl.ecl_scenario(horizon_months=h, **base)
        assert np.all(cur >= prev) and np.all(cur >= 0)
        prev = cur
    full = ecl.ecl_scenario(horizon_months=24, **base)
    doubled = ecl.ecl_scenario(horizon_months=24, **{**base, "ead": 2 * base["ead"]})
    np.testing.assert_allclose(doubled, 2 * full, rtol=1e-12)
    half_lgd = ecl.ecl_scenario(horizon_months=24, **{**base, "lgd": 0.5 * base["lgd"]})
    np.testing.assert_allclose(half_lgd, 0.5 * full, rtol=1e-12)
    # by hand for account 1: sum q lgd ead DF
    df = ecl.discount_factors(eir, base["months"])
    by_hand = np.sum(base["marginal_pd"] * base["lgd"][:, None] * base["ead"][:, None] * df, axis=1)
    np.testing.assert_allclose(full, by_hand, rtol=1e-12)
    # per-period LGD and EAD arrays (A, T)
    lgd_t = np.tile(base["lgd"][:, None], (1, 8))
    ead_t = np.tile(base["ead"][:, None], (1, 8))
    np.testing.assert_allclose(ecl.ecl_scenario(horizon_months=24, **{**base, "lgd": lgd_t, "ead": ead_t}), full, rtol=1e-12)
    # at a zero EIR the discounting is a no-op; a positive EIR lowers the ECL
    zero = ecl.ecl_scenario(horizon_months=24, **{**base, "eir": np.zeros(4)})
    assert np.all(zero >= full - 1e-15)
    # empty portfolio
    empty = ecl.ecl_scenario(np.zeros((0, 2)), 0.4, 1.0, np.zeros(0), months, 12)
    assert empty.shape == (0,)
    # inputs out of range are refused
    with pytest.raises(ValueError, match="lgd"):
        ecl.ecl_scenario(q, horizon_months=12, **{**kw, "lgd": 1.2})
    with pytest.raises(ValueError, match="ead"):
        ecl.ecl_scenario(q, horizon_months=12, **{**kw, "ead": -1.0})
    with pytest.raises(ValueError, match="marginal_pd"):
        ecl.ecl_scenario(np.array([[0.0, 0.01]]), horizon_months=12, **kw)


def test_revolving_facility_lifetime_with_commitment_ead():
    """A revolving facility: behavioural life of 24 months, EAD from drawn plus CCF times undrawn, lifetime ECL."""
    drawn, undrawn, ccf = np.array([60.0]), np.array([40.0]), 0.5
    ead = ecl.commitment_ead(drawn, undrawn, ccf)
    assert ead[0] == pytest.approx(80.0)
    curve = ts.from_hazard(np.full((1, 8), 0.01), np.arange(1, 9) * 3)
    life = ecl.ecl_scenario(curve.marginal, 0.4, ead, np.array([0.1]), curve.months, 24)
    twelve = ecl.ecl_scenario(curve.marginal, 0.4, ead, np.array([0.1]), curve.months, 12)
    assert life[0] > twelve[0] > 0
    # EAD that varies by period
    ead_t = np.linspace(80, 60, 8)[None, :]
    assert ecl.ecl_scenario(curve.marginal, 0.4, ead_t, np.array([0.1]), curve.months, 24)[0] > 0


def test_probability_weighted_ecl_golden_and_properties():
    pits = np.array([0.005255059, 0.014287387, 0.073424082])
    w = np.array([0.25, 0.5, 0.25])
    assert ecl.probability_weighted_ecl(0.4 * pits, w) == pytest.approx(0.010725392, abs=1e-8)
    assert 0.4 * 0.017997448 == pytest.approx(0.007198979, abs=1e-8)
    # (S, A) input
    m = np.array([[1.0, 10.0], [2.0, 20.0], [6.0, 60.0]])
    np.testing.assert_allclose(ecl.probability_weighted_ecl(m, w), [2.75, 27.5], atol=1e-12)
    # permuting scenarios with their weights leaves it unchanged
    perm = np.array([2, 0, 1])
    np.testing.assert_allclose(ecl.probability_weighted_ecl(m[perm], w[perm]), ecl.probability_weighted_ecl(m, w), atol=1e-12)
    # zero and maximal weights
    assert ecl.probability_weighted_ecl(np.array([10.0, 99.0]), np.array([1.0, 0.0])) == 10.0
    # the convex regime: weighted >= ECL at the mean path
    zs = np.array([1.0, 0.0, -2.0])
    loss = 0.4 * ts.ttc_to_pit(0.02, 0.12, zs)
    assert ecl.probability_weighted_ecl(loss, w) >= 0.4 * ts.ttc_to_pit(0.02, 0.12, float(w @ zs))
    for scen, bad in ((np.array([1.0]), np.array([1.0])), (np.array([1.0, 2.0]), np.array([0.6, 0.6])),
                      (np.array([1.0, 2.0]), np.array([-0.2, 1.2]))):
        with pytest.raises(ValueError):
            ecl.probability_weighted_ecl(scen, bad)
    with pytest.raises(ValueError, match="weights"):
        ecl.probability_weighted_ecl(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))


def test_non_linearity_index_and_weight_sensitivity():
    assert ecl.non_linearity_index(0.026813479, 0.017997448) == pytest.approx(0.4898489, abs=1e-6)
    assert 0.026813479 / 0.017997448 == pytest.approx(1.49, abs=1e-2)
    assert ecl.non_linearity_index(5.0, 5.0) == 0.0
    for den in (0.0, -1.0):
        with pytest.raises(ValueError, match="ecl_at_mean_path"):
            ecl.non_linearity_index(1.0, den)
    sens = ecl.weight_sensitivity(np.array([10.0, 20.0, 60.0]), 1)
    np.testing.assert_allclose(sens, [-10.0, 0.0, 40.0])
    # antisymmetry under a swapped reference, and the derivative of the weighted ECL
    a = ecl.weight_sensitivity(np.array([10.0, 20.0, 60.0]), 0)
    b = ecl.weight_sensitivity(np.array([10.0, 20.0, 60.0]), 2)
    assert a[2] == -b[0]
    e = np.array([10.0, 20.0, 60.0])
    w = np.array([0.3, 0.5, 0.2])
    dw = 0.05
    moved = w + dw * np.array([0.0, -1.0, 1.0])  # weight from reference 1 to scenario 2
    assert ecl.probability_weighted_ecl(e, moved) - ecl.probability_weighted_ecl(e, w) == pytest.approx(
        dw * ecl.weight_sensitivity(e, 1)[2], abs=1e-12)
    with pytest.raises(ValueError, match="reference_index"):
        ecl.weight_sensitivity(e, 3)


def test_stage3_ecl_golden_and_properties():
    v = ecl.stage3_ecl(np.array([50000.0]), np.array([[30000.0]]), np.array([[12]]), np.array([0.08]))
    assert v[0] == pytest.approx(50000 - 30000 / 1.08, abs=1e-6)
    assert v[0] == pytest.approx(22222.22, abs=5e-3)
    # zero recoveries give the gross carrying amount; recoveries above the carrying amount clip at zero
    assert ecl.stage3_ecl(np.array([100.0]), np.zeros((1, 0)), np.zeros((1, 0)), np.array([0.05]))[0] == 100.0
    assert ecl.stage3_ecl(np.array([100.0]), np.array([[500.0]]), np.array([[6]]), np.array([0.05]))[0] == 0.0
    # nonincreasing in each recovery and discounted at the EIR
    lo = ecl.stage3_ecl(np.array([100.0]), np.array([[10.0, 20.0]]), np.array([[6, 24]]), np.array([0.1]))
    hi = ecl.stage3_ecl(np.array([100.0]), np.array([[12.0, 20.0]]), np.array([[6, 24]]), np.array([0.1]))
    assert hi[0] < lo[0]
    assert lo[0] == pytest.approx(100 - 10 * 1.1**-0.5 - 20 * 1.1**-2, abs=1e-9)
    with pytest.raises(ValueError, match="expected_recoveries"):
        ecl.stage3_ecl(np.array([100.0]), np.array([[-1.0]]), np.array([[6]]), np.array([0.1]))
    with pytest.raises(ValueError, match="recovery_months"):
        ecl.stage3_ecl(np.array([100.0]), np.array([[1.0, 2.0]]), np.array([[6]]), np.array([0.1]))


def test_poci_matrix_and_commitment_goldens():
    np.testing.assert_allclose(ecl.poci_allowance(np.array([120.0, 80.0]), np.array([100.0, 100.0])), [20.0, -20.0])
    assert ecl.simplified_matrix_ecl(np.array([1000.0, 500.0, 200.0]), np.array([0.01, 0.05, 0.20])) == pytest.approx(75.0)
    # multiplier, cap and linearity below the cap
    g, r = np.array([1000.0, 500.0, 200.0]), np.array([0.01, 0.05, 0.20])
    assert ecl.simplified_matrix_ecl(g, r, 2.0) == pytest.approx(150.0)
    assert ecl.simplified_matrix_ecl(g, r, 1000.0) == pytest.approx(1700.0)  # capped at the exposure
    assert ecl.simplified_matrix_ecl(2 * g, r) == pytest.approx(150.0)
    with pytest.raises(ValueError, match="loss_rate_by_bucket"):
        ecl.simplified_matrix_ecl(g, np.array([0.1, 0.2, 1.5]))
    with pytest.raises(ValueError, match="forward_looking_multiplier"):
        ecl.simplified_matrix_ecl(g, r, -1.0)
    assert ecl.commitment_ead(np.array([60.0]), np.array([40.0]), 0.5)[0] == 80.0
    assert ecl.commitment_ead(np.array([60.0]), np.array([40.0]), 0.0)[0] == 60.0
    assert ecl.commitment_ead(np.array([60.0]), np.array([40.0]), 1.0)[0] == 100.0
    with pytest.raises(ValueError, match="ccf"):
        ecl.commitment_ead(np.array([60.0]), np.array([40.0]), 1.5)


def test_portfolio_allowance_golden_poci_and_empty():
    stage = np.array([1, 2, 3])
    e12 = np.array([10.0, 20.0, 30.0])
    life = np.array([15.0, 40.0, 45.0])
    s3 = np.array([0.0, 0.0, 25.0])
    out = ecl.portfolio_allowance(stage, e12, life, s3, overlays=[overlay(amount=5.0)])
    assert out.by_stage == {"stage1": 10.0, "stage2": 40.0, "stage3": 25.0, "poci": 0.0}
    assert out.model_total == 75.0 and out.overlay_total == 5.0 and out.total == 80.0
    assert out.total == out.model_total + out.overlay_total
    # overlays never enter the model ECL
    assert sum(out.by_stage.values()) == out.model_total
    # a POCI account counts once, in poci
    p = ecl.portfolio_allowance(stage, e12, life, s3, poci=np.array([False, False, True]),
                                poci_delta=np.array([np.nan, np.nan, 7.0]), overlays=[overlay(amount=5.0)])
    assert p.by_stage == {"stage1": 10.0, "stage2": 40.0, "stage3": 0.0, "poci": 7.0}
    assert p.model_total == 57.0 and p.total == 62.0
    # a negative POCI delta is kept
    neg = ecl.portfolio_allowance(stage, e12, life, s3, poci=np.array([False, False, True]),
                                  poci_delta=np.array([0.0, 0.0, -3.0]))
    assert neg.by_stage["poci"] == -3.0
    with pytest.raises(ValueError, match="poci_delta"):
        ecl.portfolio_allowance(stage, e12, life, s3, poci=np.array([False, False, True]),
                                poci_delta=np.array([0.0, 0.0, np.nan]))
    with pytest.raises(ValueError, match="poci_delta"):
        ecl.portfolio_allowance(stage, e12, life, s3, poci=np.array([False, False, True]))
    with pytest.raises(ValueError, match="stage"):
        ecl.portfolio_allowance(np.array([1, 4, 3]), e12, life, s3)
    # POCI rows may carry any stage value
    ok = ecl.portfolio_allowance(np.array([1, 0, 3]), e12, life, s3, poci=np.array([False, True, False]),
                                 poci_delta=np.array([np.nan, 2.0, np.nan]))
    assert ok.by_stage["poci"] == 2.0
    # duplicate overlay ids are refused
    with pytest.raises(ValueError, match="overlay"):
        ecl.portfolio_allowance(stage, e12, life, s3, overlays=[overlay("A"), overlay("A")])
    # empty portfolio
    z = np.zeros(0)
    e = ecl.portfolio_allowance(np.zeros(0, dtype=np.int64), z, z, z)
    assert e.by_stage == {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "poci": 0.0} and e.total == 0.0
    # a single account
    assert ecl.portfolio_allowance(np.array([2]), np.array([1.0]), np.array([4.0]), np.array([0.0])).total == 4.0
    # mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        ecl.portfolio_allowance(stage, e12[:2], life, s3)
