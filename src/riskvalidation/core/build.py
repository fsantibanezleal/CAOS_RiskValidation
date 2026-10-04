"""Assemble a :class:`TestResult` from a registered test's statistics and a policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.policy import Policy, default_policy
from riskvalidation.core.registry import spec as get_spec
from riskvalidation.core.result import Light, TestResult


def make_result(
    test_id: str,
    *,
    statistic: float | None = None,
    p_value: float | None = None,
    metric: float | None = None,
    policy: Policy | None = None,
    alternative: str = "",
    n: int | None = None,
    n_events: int | None = None,
    model_id: str | None = None,
    segment: str | None = None,
    inputs: Sequence[Any] = (),
    notes: str = "",
    extras: Mapping[str, Any] | None = None,
) -> TestResult:
    """Build the result of a registered test, classifying it under ``policy`` (the default when ``None``)."""
    s = get_spec(test_id)
    pol = policy if policy is not None else default_policy()
    light, th = pol.classify(test_id, p_value=p_value, metric=metric)
    if th is not None and light is Light.NOT_EVALUATED and not notes:
        notes = "statistic undefined for this sample; no light assigned"
    elif th is None and not notes:
        notes = f"policy {pol.label} holds no thresholds for this test; no light assigned"
    return TestResult(
        test_id=test_id,
        h0=s.h0,
        reference=s.reference,
        policy_version=pol.label,
        light=light,
        statistic=statistic,
        p_value=p_value,
        metric=metric,
        alternative=alternative,
        alpha_amber=None if th is None else th.amber,
        alpha_red=None if th is None else th.red,
        n=n,
        n_events=n_events,
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(*inputs) if inputs else "",
        notes=notes,
        extras=dict(extras or {}),
    )
