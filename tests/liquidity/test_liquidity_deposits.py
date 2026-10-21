"""RV-1117 and RV-1118: the depositor generator with known truth (contract `engine-liquidity-api`, module deposits;
dossier 08 G.2 line 698 and the generator contract of `generators._base`)."""

from __future__ import annotations

import dataclasses
import itertools
import json

import numpy as np
import pytest

from riskvalidation.generators._base import Check, GeneratorBase
from riskvalidation.liquidity import deposits as d
from riskvalidation.liquidity import runoff


def seg(segment="retail-uninsured", share=1.0, hc=0.001, hs=0.01):
    return d.DepositorSegment(segment, share, hc, hs)


def cfg(*segments, n=200, horizon=3, enter=1.0, exit_=0.0):
    return d.DepositorConfig(tuple(segments), n, horizon, enter, exit_)


def test_depositor_config_validation():
    c = cfg(seg())
    assert hash(c) == hash(cfg(seg())) and c == cfg(seg())  # frozen and hashable
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.n_depositors = 5  # type: ignore[misc]
    # the three-segment shares of the contract sum to 1.0
    three = cfg(seg("retail-insured", 0.057), seg("retail-uninsured", 0.864), seg("operational", 0.079))
    assert sum(s.share_of_deposits for s in three.segments) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="share_of_deposits"):
        cfg(seg(share=0.9))
    with pytest.raises(ValueError, match="share_of_deposits"):
        cfg(seg("retail-insured", 0.5), seg("retail-uninsured", 0.4))
    for field, kwargs in (("calm_daily_hazard", {"hc": 1.1}), ("stress_daily_hazard", {"hs": -0.1})):
        with pytest.raises(ValueError, match=field):
            cfg(seg(**kwargs))
    # BL-082: the chain probabilities belong to the configuration, each in [0, 1]
    for field, kwargs in (("p_enter_stress", {"enter": 2.0}), ("p_exit_stress", {"exit_": -0.5})):
        with pytest.raises(ValueError, match=field):
            cfg(seg(), **kwargs)
    with pytest.raises(ValueError, match="p_enter_stress"):
        cfg(seg(), enter=float("nan"))
    with pytest.raises(ValueError, match="n_depositors"):
        cfg(seg(), n=0)
    with pytest.raises(ValueError, match="horizon_days"):
        cfg(seg(), horizon=0)
    with pytest.raises(ValueError, match="segment"):
        cfg(seg("household"))
    with pytest.raises(ValueError, match="segments"):
        d.DepositorConfig((), 10, 3, 1.0, 0.0)
    with pytest.raises(ValueError, match="segment"):  # a segment id twice
        cfg(seg("retail-insured", 0.5), seg("retail-insured", 0.5))
    # the state chain is shared by all depositors, so it is one pair of numbers of the configuration: a segment no
    # longer takes it (passing it is a TypeError, the BL-082 break of engine-liquidity-api 2.0.0)
    with pytest.raises(TypeError):
        d.DepositorSegment("retail-uninsured", 1.0, 0.001, 0.01, 0.2, 0.3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        d.DepositorSegment("retail-uninsured", 1.0, 0.001, 0.01, p_enter_stress=0.2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        d.DepositorConfig((seg(),), 10, 3)  # type: ignore[call-arg]  # the two probabilities are required
    assert not {"p_enter_stress", "p_exit_stress"} & {f.name for f in dataclasses.fields(d.DepositorSegment)}
    assert [f.name for f in dataclasses.fields(d.DepositorConfig)] == [
        "segments", "n_depositors", "horizon_days", "p_enter_stress", "p_exit_stress"]
    # a rate of exactly 0 or 1 is legitimate for a segment and for the chain
    cfg(seg("financial", 1.0, hc=1.0, hs=1.0), enter=0.0, exit_=1.0)
    cfg(seg(hc=0.0, hs=0.0))


def test_depositor_config_serialises():
    g = d.DepositorGenerator(cfg(seg("retail-insured", 0.4), seg("retail-uninsured", 0.6), n=50, horizon=4))
    assert isinstance(g, GeneratorBase) and g.name == "depositors"
    out = g.config()
    json.dumps(out)  # JSON-safe
    assert out["generator"] == "depositors" and out["riskvalidation"]
    params = out["parameters"]
    assert params["n_depositors"] == 50 and params["horizon_days"] == 4
    assert [s["segment"] for s in params["segments"]] == ["retail-insured", "retail-uninsured"]
    assert params["segments"][1]["share_of_deposits"] == 0.6


def brute_truth(hc, hs, enter, exit_, horizon):
    """The expected cumulative outflow by enumerating every state path (calm on day 1): the second route."""
    out = np.zeros(horizon)
    for path in itertools.product((0, 1), repeat=horizon - 1):
        states = (0,) + path
        p = 1.0
        for a, b in zip(states, states[1:], strict=False):
            to_stress = enter if a == 0 else 1.0 - exit_
            p *= to_stress if b == 1 else 1.0 - to_stress
        surv = 1.0
        for day, st in enumerate(states):
            surv *= 1.0 - (hc if st == 0 else hs)
            out[day] += p * (1.0 - surv)
    return out


def test_depositor_truth_goldens_and_invariants():
    g = d.DepositorGenerator(cfg(seg(hc=0.001, hs=0.01), enter=1.0, exit_=0.0, horizon=3))
    t = g.truth()["outflow_rate"]["retail-uninsured"]
    assert t.tolist() == pytest.approx([0.001, 0.01099, 0.0208801], abs=5e-7)
    assert t[2] == pytest.approx(1 - 0.999 * 0.99 * 0.99, abs=1e-12)
    # with p_enter = 0 the truth is the constant compounding of runoff
    g0 = d.DepositorGenerator(cfg(seg(hc=0.001, hs=0.01), enter=0.0, exit_=0.5, horizon=30))
    t0 = g0.truth()["outflow_rate"]["retail-uninsured"]
    assert t0 == pytest.approx(runoff.cumulative_outflow_constant(0.001, np.arange(1, 31)), abs=1e-12)
    # a genuine chain, against the enumeration of all state paths
    for enter, exit_ in ((0.3, 0.2), (0.05, 0.6), (1.0, 1.0), (0.5, 0.0)):
        gg = d.DepositorGenerator(cfg(seg(hc=0.002, hs=0.03), enter=enter, exit_=exit_, horizon=8))
        got = gg.truth()["outflow_rate"]["retail-uninsured"]
        assert got == pytest.approx(brute_truth(0.002, 0.03, enter, exit_, 8), abs=1e-12), (enter, exit_)
        assert np.all(np.diff(got) >= 0.0)  # non-decreasing in the horizon
    # several segments: one truth per segment on the one chain
    two = d.DepositorGenerator(cfg(seg("retail-insured", 0.5, hc=0.0005, hs=0.002),
                                   seg("financial", 0.5, hc=0.01, hs=1.0), enter=0.3, exit_=0.2, horizon=6))
    truth = two.truth()["outflow_rate"]
    assert set(truth) == {"retail-insured", "financial"}
    assert truth["financial"][-1] > truth["retail-insured"][-1]
    assert truth["financial"] == pytest.approx(brute_truth(0.01, 1.0, 0.3, 0.2, 6), abs=1e-12)
    # the truth does not depend on the number of depositors or on any seed
    assert np.array_equal(g.truth()["outflow_rate"]["retail-uninsured"], t)


def test_depositor_draw_seeded_and_consistent():
    c = cfg(seg("retail-insured", 0.4, hc=0.01, hs=0.05),
            seg("retail-uninsured", 0.6, hc=0.02, hs=0.2), n=500, horizon=20, enter=0.2, exit_=0.3)
    g = d.DepositorGenerator(c)
    a = g.draw(np.random.default_rng(11))
    b = g.draw(np.random.default_rng(11))
    other = g.draw(np.random.default_rng(12))
    assert np.array_equal(a["state"], b["state"])
    for s in ("retail-insured", "retail-uninsured"):
        assert np.array_equal(a["withdrawn_day"][s], b["withdrawn_day"][s])
        assert np.array_equal(a["outflow_rate"][s], b["outflow_rate"][s])
    assert not np.array_equal(a["withdrawn_day"]["retail-uninsured"], other["withdrawn_day"]["retail-uninsured"])
    assert a["state"].shape == (20,) and a["state"][0] == 0  # calm on day 1
    assert set(np.unique(a["state"])) <= {0, 1}
    for s in ("retail-insured", "retail-uninsured"):
        wd = a["withdrawn_day"][s]
        assert wd.shape == (500,) and wd.min() >= 0 and wd.max() <= 20  # 0 = not within the horizon
        rate = a["outflow_rate"][s]
        assert rate.shape == (20,) and np.all(np.diff(rate) >= 0.0) and rate[-1] <= 1.0
        for day in (1, 5, 20):  # the cumulative share is the share withdrawn by that day
            assert rate[day - 1] == pytest.approx(np.mean((wd >= 1) & (wd <= day)), abs=1e-12)
    # a rate of 1 in the calm state empties the segment on day 1; a rate of 0 never withdraws anyone
    full = d.DepositorGenerator(cfg(seg("financial", 1.0, hc=1.0, hs=1.0), n=30, horizon=4, enter=0.0, exit_=1.0))
    f = full.draw(np.random.default_rng(1))
    assert f["withdrawn_day"]["financial"].tolist() == [1] * 30 and f["outflow_rate"]["financial"].tolist() == [1.0] * 4
    none = d.DepositorGenerator(cfg(seg(hc=0.0, hs=0.0), n=30, horizon=4, enter=0.5, exit_=0.5))
    z = none.draw(np.random.default_rng(1))
    assert z["withdrawn_day"]["retail-uninsured"].tolist() == [0] * 30
    assert z["outflow_rate"]["retail-uninsured"].sum() == 0


def test_depositor_verify_reproduces_truth():
    g = d.DepositorGenerator(cfg(seg("retail-insured", 0.4, hc=0.002, hs=0.02),
                                 seg("financial", 0.6, hc=0.01, hs=0.3), n=300, horizon=10, enter=0.2, exit_=0.3))
    checks = g.verify(n_draws=4000)
    assert checks and all(isinstance(c, Check) for c in checks)
    assert all(c.ok(4.0) for c in checks), [(c.name, c.z) for c in checks]
    assert {c.name.split("@")[0] for c in checks} == {"retail-insured", "financial"}
    # it checks against the exact truth and is seeded: the same call gives the same observations
    again = g.verify(n_draws=4000)
    assert [c.observed for c in checks] == [c.observed for c in again]
    truth = g.truth()["outflow_rate"]
    for c in checks:
        seg_name, horizon = c.name.split("@")
        assert c.expected == pytest.approx(truth[seg_name][int(horizon) - 1], abs=1e-15)
    # the verifier is a real test: a generator whose truth is wrong is caught
    wrong = d.DepositorGenerator(g.spec)
    wrong.truth = lambda: {"outflow_rate": {k: v * 0.5 for k, v in truth.items()}}  # type: ignore[method-assign]
    assert any(not c.ok(4.0) for c in wrong.verify(n_draws=4000))
