"""RV-520 to RV-522: the fixes of the engine review (F2 negative ECL, C2 required `now` of the cure probation)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from riskvalidation.provisions import ecl
from riskvalidation.provisions import staging as st


def test_portfolio_allowance_refuses_negative_model_ecl_for_stages_1_to_3():
    stage = np.array([1, 2, 3])
    ok = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 3.0]))
    assert ecl.portfolio_allowance(stage, *ok).model_total == 6.0
    for which in range(3):
        bad = [x.copy() for x in ok]
        bad[which][which] = -0.01
        with pytest.raises(ValueError, match="negative"):
            ecl.portfolio_allowance(stage, *bad)
    # a negative value outside the stage of the account is not an input of the allowance
    other = [x.copy() for x in ok]
    other[2][0] = -5.0
    assert ecl.portfolio_allowance(stage, *other).model_total == 6.0
    # POCI may be negative (a favourable change in lifetime ECL)
    out = ecl.portfolio_allowance(np.array([1, 1]), np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                                  poci=np.array([False, True]), poci_delta=np.array([0.0, -20.0]))
    assert out.by_stage["poci"] == -20.0


def test_probability_weighted_ecl_refuses_negative_ecl():
    w = np.array([0.5, 0.5])
    assert ecl.probability_weighted_ecl(np.array([0.0, 4.0]), w) == 2.0
    for bad in (np.array([-1.0, 4.0]), np.array([[1.0, 2.0], [-0.5, 2.0]])):
        with pytest.raises(ValueError, match="negative"):
            ecl.probability_weighted_ecl(bad, w)


def test_apply_cure_probation_requires_the_current_stage_result():
    assert inspect.signature(st.apply_cure_probation).parameters["now"].default is inspect.Parameter.empty
    pol = st.StagingPolicy(multiplier_schedule=((0.5, 2.0),), absolute_floor=0.0, probation_months=6)
    stage = np.array([1, 2, 3], dtype=np.int8)
    before = np.array([2, 2, 3], dtype=np.int8)
    months = np.array([3, 0, 0], dtype=np.int64)
    with pytest.raises(TypeError):
        st.apply_cure_probation(stage, before, months, pol)  # type: ignore[call-arg]
    cur = st.StageResult(stage=stage, trigger=np.array(["none", "watch-list", "default"], dtype="U20"),
                         lcr_exempt=np.zeros(3, dtype=bool))
    r = st.apply_cure_probation(stage, before, months, pol, cur)
    assert set(r.trigger.tolist()) <= set(st.TRIGGERS)
    assert r.trigger.tolist() == ["cure-probation", "watch-list", "default"]
    assert "unspecified" not in r.trigger.tolist()
