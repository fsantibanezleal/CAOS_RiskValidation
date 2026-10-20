"""RV-1462: replication of an estimate on the next year, by sign and size (decision Q4)."""

from __future__ import annotations

import dataclasses

import pytest

from riskvalidation.fairness.conditional import Effect
from riskvalidation.fairness.replication import Replication, replicate_effect


def _effect(coef: float, se: float = 0.1) -> Effect:
    import math

    return Effect(coefficient=coef, odds_ratio=math.exp(coef), lo=coef - 1.96 * se, hi=coef + 1.96 * se, level=0.95)


def test_replicate_effect_planted_equal_estimates():
    r = replicate_effect(_effect(-0.4), _effect(-0.4))
    assert isinstance(r, Replication)
    assert r.same_sign is True and r.size_ratio == 1.0
    assert (r.fit_coefficient, r.confirm_coefficient) == (-0.4, -0.4)
    half = replicate_effect(_effect(0.5), _effect(0.25))
    assert half.same_sign is True and half.size_ratio == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.same_sign = False  # type: ignore[misc]


def test_replicate_effect_sign_flip_and_zero_fit():
    flip = replicate_effect(_effect(0.3), _effect(-0.15))
    assert flip.same_sign is False
    assert flip.size_ratio == -0.5
    zero = replicate_effect(_effect(0.0), _effect(0.2))
    assert zero.size_ratio is None
    assert zero.same_sign is False
    both_zero = replicate_effect(_effect(0.0), _effect(0.0))
    assert both_zero.size_ratio is None
    with pytest.raises(ValueError, match="fit"):
        replicate_effect(_effect(float("nan")), _effect(0.1))
    with pytest.raises(ValueError, match="confirm"):
        replicate_effect(_effect(0.1), _effect(float("inf")))
