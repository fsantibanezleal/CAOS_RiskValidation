"""RV-1109 to RV-1111: the contractual and behavioural ladders, the net liquidity position and the survival period
(contract `engine-liquidity-api`, module ladder; dossier 05 E.1 lines 234 to 236, E.5)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.liquidity import ladder
from riskvalidation.liquidity import runoff as r


def a(*x):
    return np.array(x, dtype=float)


def test_contractual_gaps_golden():
    gap, cum = ladder.contractual_gaps(a(10, 20, 30), a(40, 10, 5))
    assert gap.tolist() == [-30.0, 10.0, 25.0] and cum.tolist() == [-30.0, -20.0, 5.0]
    # the last cumulative gap is the sum of all gaps
    rng = np.random.default_rng(1)
    i, o = rng.uniform(0, 100, 12), rng.uniform(0, 100, 12)
    g, c = ladder.contractual_gaps(i, o)
    assert c[-1] == pytest.approx(g.sum(), abs=1e-9)
    assert g == pytest.approx(i - o, abs=0.0)
    with pytest.raises(ValueError, match="same length"):
        ladder.contractual_gaps(a(1, 2), a(1))
    with pytest.raises(ValueError, match="^inflows "):
        ladder.contractual_gaps(a(1, np.nan), a(1, 2))


def test_behavioural_outflows_golden_and_invariants():
    out = ladder.behavioural_outflows(a(100.0), np.array([[0.1, 0.1]]))
    assert out.shape == (1, 2)
    assert out[0] == pytest.approx([10.0, 9.0], abs=1e-12)  # the second day is 90 x 0.1
    # each row is opening x diff(cumulative path), computed here through the contract's own identity
    rng = np.random.default_rng(3)
    opening = a(100.0, 250.0, 40.0)
    rates = rng.uniform(0.0, 0.2, (3, 20))
    out = ladder.behavioural_outflows(opening, rates)
    for s in range(3):
        cum = r.cumulative_outflow_path(rates[s])
        assert out[s] == pytest.approx(opening[s] * np.diff(cum, prepend=0.0), abs=1e-10)
        assert out[s].sum() == pytest.approx(opening[s] * cum[-1], abs=1e-10)
    # the second route: day by day on the stock
    stock = 100.0
    day = []
    for q in (0.3, 0.0, 0.5):
        day.append(stock * q)
        stock *= 1 - q
    assert ladder.behavioural_outflows(a(100.0), np.array([[0.3, 0.0, 0.5]]))[0] == pytest.approx(day, abs=1e-12)
    # the stock never goes negative: a rate of 1 empties it, later days are zero
    out = ladder.behavioural_outflows(a(100.0), np.array([[0.5, 1.0, 0.3, 0.3]]))
    assert out[0].tolist() == pytest.approx([50.0, 50.0, 0.0, 0.0], abs=1e-12)
    assert np.all(out >= 0.0)
    with pytest.raises(ValueError, match="^daily_rates: "):
        ladder.behavioural_outflows(a(1.0), np.array([[0.1, 1.5]]))
    with pytest.raises(ValueError, match="opening"):
        ladder.behavioural_outflows(a(1.0, 2.0), np.array([[0.1, 0.1]]))
    with pytest.raises(ValueError, match="^opening: "):
        ladder.behavioural_outflows(a(-1.0), np.array([[0.1]]))


def test_net_liquidity_position_golden():
    cbc, out, inf = a(100, 100, 100, 100), a(30, 40, 50, 60), a(0, 10, 0, 0)
    nlp = ladder.net_liquidity_position(cbc, out, inf)
    assert nlp.tolist() == [70.0, 40.0, -10.0, -70.0]
    # the identity through a loop
    run, expect = 0.0, []
    for t in range(4):
        run += out[t] - inf[t]
        expect.append(cbc[t] - run)
    assert nlp.tolist() == expect
    assert cbc.tolist() == [100.0] * 4 and out.tolist() == [30.0, 40.0, 50.0, 60.0]  # not mutated
    with pytest.raises(ValueError, match="same length"):
        ladder.net_liquidity_position(cbc, out[:3], inf)


def test_survival_period_never_a_number_when_it_survives():
    s = ladder.survival_period(a(70.0, 40.0, -10.0, -70.0))
    assert isinstance(s, ladder.SurvivalResult)
    assert (s.survives, s.day, s.horizon) == (False, 3, 4)
    s = ladder.survival_period(a(5.0, 4.0, 3.0))
    assert (s.survives, s.day, s.horizon) == (True, None, 3)
    s = ladder.survival_period(a(1.0, 0.0, 0.0))  # a zero NLP is not negative
    assert s.survives is True and s.day is None and s.horizon == 3
    assert ladder.survival_period(a(-1.0)).day == 1
    # the survival day is never a number when the position stays non-negative: checked on random ladders
    rng = np.random.default_rng(5)
    for _ in range(50):
        nlp = rng.normal(5.0, 4.0, 15)
        s = ladder.survival_period(nlp)
        negatives = np.flatnonzero(nlp < 0)
        assert s.survives == (negatives.size == 0)
        assert s.day == (None if negatives.size == 0 else int(negatives[0]) + 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.day = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="^nlp "):
        ladder.survival_period(a())


def test_minimum_nlp_golden():
    assert ladder.minimum_nlp(a(70.0, 40.0, -10.0, -70.0)) == (-70.0, 4)
    assert ladder.minimum_nlp(a(5.0, 3.0, 3.0)) == (3.0, 2)  # the first day on which the minimum is reached
    v, d = ladder.minimum_nlp(a(1.0, 2.0))
    assert isinstance(v, float) and isinstance(d, int)


def test_cliff_effect_golden_and_refusal():
    nlp = np.full(40, 0.0)
    nlp[29], nlp[34] = 100.0, 40.0
    assert ladder.cliff_effect(nlp, 1000.0) == pytest.approx(-0.06, abs=1e-12)
    nlp[34] = 95.0
    assert ladder.cliff_effect(nlp, 1000.0) == pytest.approx(-0.005, abs=1e-12)
    # near and far are the caller's (the named parameter cliff_days)
    assert ladder.cliff_effect(nlp, 1000.0, near=30, far=40) == pytest.approx((nlp[39] - nlp[29]) / 1000.0)
    with pytest.raises(ValueError, match="far"):
        ladder.cliff_effect(np.zeros(30), 1000.0)  # a 30-day ladder
    with pytest.raises(ValueError, match="^total_assets: "):
        ladder.cliff_effect(nlp, 0.0)
    with pytest.raises(ValueError, match="near"):
        ladder.cliff_effect(nlp, 1000.0, near=35, far=30)
