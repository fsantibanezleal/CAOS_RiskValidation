"""Shared fixtures: a small policy and two self-test registrations used only by the core tests."""

from __future__ import annotations

import pytest

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy, Threshold
from riskvalidation.core.registry import register


@register(
    "selftest.p",
    family="selftest",
    name="p-value self-test",
    h0="the self-test null holds",
    reference="riskvalidation core test fixture",
    decision="p_value",
)
def selftest_p(p: float, *, policy=None, model_id=None, segment=None):
    return make_result("selftest.p", p_value=p, policy=policy, model_id=model_id, segment=segment)


@register(
    "selftest.raises",
    family="selftest",
    name="always raises",
    h0="never evaluated",
    reference="riskvalidation core test fixture",
    decision="metric",
)
def selftest_raises(*, policy=None, model_id=None, segment=None):
    raise ZeroDivisionError("planted failure")


@pytest.fixture
def policy() -> Policy:
    return Policy(
        policy_id="test-policy",
        version="1.2.3",
        description="fixture",
        thresholds={"selftest.p": Threshold(kind="p_value", amber=0.05, red=0.01, source="fixture")},
    )
