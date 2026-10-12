"""RV-506 to RV-508: the staging policy, the quantitative SICR test, the stage allocation with its backstops,
the cure probation and the collective transfer (dossier 02 B.5 and the contract's golden rows)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.provisions import staging as st

SCHEDULE = ((0.01, 4.0), (0.05, 2.5), (0.5, 1.5))


def policy(**kw) -> st.StagingPolicy:
    base = {"multiplier_schedule": SCHEDULE, "absolute_floor": 0.005}
    base.update(kw)
    return st.StagingPolicy(**base)


def arr(*v, dtype=float):
    return np.array(v, dtype=dtype)


def test_staging_policy_validation():
    p = policy()
    assert p.threefold_ratio == 3.0 and p.dpd_backstop == 30 and p.default_dpd == 90 and p.probation_months == 0
    assert p.low_credit_risk_pd is None
    bad = [
        {"multiplier_schedule": ((0.05, 2.0), (0.01, 3.0))},  # bounds not increasing
        {"multiplier_schedule": ((0.01, 2.0), (0.05, 3.0))},  # multiplier rises with the origination PD
        {"multiplier_schedule": ((0.01, 0.9),)},  # multiplier below one
        {"multiplier_schedule": ()},
        {"multiplier_schedule": ((1.5, 2.0),)},  # bound above one
        {"absolute_floor": -0.1},
        {"absolute_floor": 1.1},
        {"threefold_ratio": 1.0},
        {"probation_months": -1},
        {"dpd_backstop": -1},
        {"default_dpd": 20},  # not above the backstop
        {"low_credit_risk_pd": 0.0},
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            policy(**kw)
    # the threefold ratio is a policy parameter, not a constant
    assert policy(threefold_ratio=5.0).threefold_ratio == 5.0


def test_sicr_quantitative_goldens_and_properties():
    p = policy(threefold_ratio=3.0)
    got = st.sicr_quantitative(arr(0.02, 0.02, 0.001), arr(0.05, 0.049, 0.003), p)
    assert got.dtype == bool
    assert got.tolist() == [True, False, True]
    # equality never flags
    x = np.array([0.001, 0.02, 0.2, 0.5])
    assert not st.sicr_quantitative(x, x, p).any()
    # monotone in the current PD
    orig = np.full(60, 0.02)
    cur = np.linspace(0.021, 0.5, 60)
    flags = st.sicr_quantitative(orig, cur, p)
    assert np.all(np.diff(flags.astype(int)) >= 0) and flags[-1] and not flags[0]
    # elementwise: a row depends on its own row only
    both = st.sicr_quantitative(arr(0.02, 0.2), arr(0.05, 0.21), p)
    assert both[0] == st.sicr_quantitative(arr(0.02), arr(0.05), p)[0]
    assert both[1] == st.sicr_quantitative(arr(0.2), arr(0.21), p)[0]
    # a different threefold ratio changes the outcome of the backstop only
    assert not st.sicr_quantitative(arr(0.001), arr(0.003), policy(threefold_ratio=4.0))[0]
    # origination above the schedule's last bound is refused; PDs outside (0, 1) are refused
    with pytest.raises(ValueError, match="multiplier_schedule"):
        st.sicr_quantitative(arr(0.7), arr(0.8), p)
    with pytest.raises(ValueError, match="pd_current"):
        st.sicr_quantitative(arr(0.02), arr(1.0), p)
    with pytest.raises(ValueError, match="same length"):
        st.sicr_quantitative(arr(0.02, 0.03), arr(0.05), p)


def stage_args(n=1, **over):
    a = {
        "pd_origination_annualised": np.full(n, 0.02),
        "pd_current_annualised": np.full(n, 0.021),
        "dpd": np.zeros(n, dtype=np.int64),
        "forborne": np.zeros(n, dtype=bool),
        "watch_list": np.zeros(n, dtype=bool),
        "defaulted": np.zeros(n, dtype=bool),
    }
    a.update(over)
    return a


def test_allocate_stages_goldens_and_precedence():
    p = policy()
    r = st.allocate_stages(policy=p, **stage_args(3, dpd=arr(31, 30, 90, dtype=np.int64)))
    assert r.stage.tolist() == [2, 1, 3]
    assert r.stage.dtype == np.int8
    assert r.trigger.tolist() == ["dpd-backstop", "none", "default"]
    assert r.lcr_exempt.tolist() == [False, False, False]
    # each qualitative trigger alone
    one = lambda **kw: st.allocate_stages(policy=p, **stage_args(1, **kw))  # noqa: E731
    assert one(forborne=arr(True, dtype=bool)).trigger[0] == "forbearance"
    assert one(watch_list=arr(True, dtype=bool)).trigger[0] == "watch-list"
    assert st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **stage_args(1)).trigger[0] == "collective"
    assert one(pd_current_annualised=arr(0.05)).trigger[0] == "relative-absolute"
    assert one(pd_origination_annualised=arr(0.001), pd_current_annualised=arr(0.003)).trigger[0] == "threefold"
    d = one(defaulted=arr(True, dtype=bool))
    assert (d.stage[0], d.trigger[0]) == (3, "default")
    # precedence: default over dpd over forbearance over watch-list over collective over quantitative
    everything = stage_args(
        1, dpd=arr(40, dtype=np.int64), forborne=arr(True, dtype=bool), watch_list=arr(True, dtype=bool),
        pd_current_annualised=arr(0.06),
    )
    assert st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **everything).trigger[0] == "dpd-backstop"
    everything["dpd"] = arr(0, dtype=np.int64)
    assert st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **everything).trigger[0] == "forbearance"
    everything["forborne"] = arr(False, dtype=bool)
    assert st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **everything).trigger[0] == "watch-list"
    everything["watch_list"] = arr(False, dtype=bool)
    assert st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **everything).trigger[0] == "collective"
    assert st.allocate_stages(policy=p, **everything).trigger[0] == "relative-absolute"
    everything["defaulted"] = arr(True, dtype=bool)
    assert st.allocate_stages(policy=p, **everything).trigger[0] == "default"
    # trigger is none iff stage 1
    big = st.allocate_stages(policy=p, **stage_args(5, dpd=np.array([0, 31, 95, 10, 29], dtype=np.int64)))
    assert ((big.trigger == "none") == (big.stage == 1)).all()
    # a different default and backstop day count is honoured (policy parameters)
    q = policy(dpd_backstop=15, default_dpd=60)
    r2 = st.allocate_stages(policy=q, **stage_args(3, dpd=np.array([16, 15, 60], dtype=np.int64)))
    assert r2.stage.tolist() == [2, 1, 3]
    with pytest.raises(ValueError, match="dpd"):
        st.allocate_stages(policy=p, **stage_args(1, dpd=arr(-1, dtype=np.int64)))


def test_low_credit_risk_exemption():
    p = policy(low_credit_risk_pd=0.03)
    a = stage_args(1, pd_origination_annualised=arr(0.001), pd_current_annualised=arr(0.003))
    r = st.allocate_stages(policy=p, **a)
    assert (r.stage[0], r.trigger[0], bool(r.lcr_exempt[0])) == (1, "none", True)
    r31 = st.allocate_stages(policy=p, **{**a, "dpd": arr(31, dtype=np.int64)})
    assert (r31.stage[0], r31.trigger[0], bool(r31.lcr_exempt[0])) == (2, "dpd-backstop", False)
    # forbearance, watch-list and collective still apply under the exemption (5.5.11, B5.5.37)
    yes = arr(True, dtype=bool)
    for kw, trig in (({"forborne": yes}, "forbearance"), ({"watch_list": yes}, "watch-list")):
        rr = st.allocate_stages(policy=p, **{**a, **kw})
        assert (rr.stage[0], rr.trigger[0], bool(rr.lcr_exempt[0])) == (2, trig, False)
    rc = st.allocate_stages(policy=p, collective=arr(True, dtype=bool), **a)
    assert (rc.stage[0], rc.trigger[0], bool(rc.lcr_exempt[0])) == (2, "collective", False)
    rd = st.allocate_stages(policy=p, **{**a, "defaulted": arr(True, dtype=bool)})
    assert (rd.stage[0], rd.trigger[0]) == (3, "default")
    # above the exemption threshold the quantitative trigger applies
    hi = st.allocate_stages(policy=p, **{**a, "pd_current_annualised": arr(0.05)})
    assert hi.stage[0] == 2 and not hi.lcr_exempt[0]
    # an account the quantitative test would not flag is not counted as exempt
    calm = st.allocate_stages(policy=p, **stage_args(1))
    assert not calm.lcr_exempt[0]
    # without the exemption the same account is stage 2
    assert st.allocate_stages(policy=policy(), **a).stage[0] == 2


def test_stage_isolation_and_empty_portfolio():
    p = policy()
    rng = np.random.default_rng(3)
    n = 40
    base = stage_args(
        n,
        pd_origination_annualised=rng.uniform(0.002, 0.2, n),
        dpd=rng.integers(0, 100, n).astype(np.int64),
        forborne=rng.random(n) < 0.1,
        watch_list=rng.random(n) < 0.1,
    )
    base["pd_current_annualised"] = base["pd_origination_annualised"] * rng.uniform(0.8, 4.0, n)
    ref = st.allocate_stages(policy=p, **base)
    changed = {k: v.copy() for k, v in base.items()}
    changed["pd_current_annualised"][7] = 0.9
    changed["dpd"][7] = 120
    out = st.allocate_stages(policy=p, **changed)
    keep = np.arange(n) != 7
    assert (out.stage[keep] == ref.stage[keep]).all() and (out.trigger[keep] == ref.trigger[keep]).all()
    assert out.stage[7] == 3
    # inputs are not mutated
    assert base["dpd"].dtype == np.int64
    # empty portfolio
    e = st.allocate_stages(policy=p, **stage_args(0))
    assert e.stage.shape == (0,) and e.trigger.shape == (0,) and e.lcr_exempt.shape == (0,)
    assert st.sicr_quantitative(np.array([]), np.array([]), p).shape == (0,)
    # single account
    assert st.allocate_stages(policy=p, **stage_args(1)).stage.tolist() == [1]


def test_cure_probation():
    p = policy(probation_months=6)
    now = np.array([1, 1, 1, 2, 3], dtype=np.int8)
    before = np.array([2, 2, 1, 2, 2], dtype=np.int8)
    months = np.array([3, 6, 0, 0, 0], dtype=np.int64)
    cur = st.StageResult(stage=now, trigger=np.array(["none", "none", "none", "watch-list", "default"], dtype="U20"),
                         lcr_exempt=np.zeros(5, dtype=bool))
    r = st.apply_cure_probation(now, before, months, p, cur)
    assert r.stage.tolist() == [2, 1, 1, 2, 3]
    assert r.trigger[0] == "cure-probation" and r.trigger[1] == "none" and r.trigger[2] == "none"
    assert r.stage.dtype == np.int8
    # the triggers of the current allocation are carried for the accounts left unchanged
    assert r.trigger.tolist() == ["cure-probation", "none", "none", "watch-list", "default"]
    # months_since_cure = 5 is still probation, 6 is not
    one = st.StageResult(stage=np.array([1], dtype=np.int8), trigger=np.array(["none"], dtype="U20"),
                         lcr_exempt=np.zeros(1, dtype=bool))
    assert st.apply_cure_probation(np.array([1], dtype=np.int8), np.array([2], dtype=np.int8),
                                   np.array([5], dtype=np.int64), p, one).stage[0] == 2
    # zero probation: the output equals the current stage
    z = st.apply_cure_probation(now, before, months, policy(probation_months=0), cur)
    assert z.stage.tolist() == now.tolist()
    with pytest.raises(ValueError, match="now"):
        st.apply_cure_probation(now, before, months, p, st.StageResult(stage=before, trigger=cur.trigger,
                                                                       lcr_exempt=cur.lcr_exempt))


def test_collective_transfer():
    out = st.collective_transfer(np.array([100000.0]), np.array(["s1"]), {"s1": 0.1})
    assert out == {"s1": pytest.approx(10000.0)}
    gca = np.array([100.0, 300.0, 50.0, 50.0])
    seg = np.array(["a", "b", "a", "b"])
    res = st.collective_transfer(gca, seg, {"a": 0.2, "b": 0.0, "unused": 0.5})
    assert res == {"a": pytest.approx(30.0), "b": 0.0}
    assert sum(res.values()) <= gca.sum()
    with pytest.raises(KeyError):
        st.collective_transfer(gca, seg, {"a": 0.2})
    with pytest.raises(ValueError, match="share"):
        st.collective_transfer(gca, seg, {"a": 1.2, "b": 0.0})
    with pytest.raises(ValueError, match="gross_carrying_amount"):
        st.collective_transfer(np.array([-1.0]), np.array(["a"]), {"a": 0.1})
    assert st.collective_transfer(np.array([]), np.array([], dtype=str), {}) == {}
