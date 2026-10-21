"""Run a battery of validation tests under one policy, never stopping at the first failure.

A validator needs the whole picture: a test that raises (bad input, an empty segment) is recorded as an
``ERROR`` result with the exception text, and the remaining tests still run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from riskvalidation.core.policy import Policy, default_policy
from riskvalidation.core.result import Light, TestResult

Task = tuple[Callable[..., TestResult], Mapping[str, Any]]


@dataclass(frozen=True)
class SuiteRun:
    """Results of one suite run, in task order."""

    policy_version: str
    results: tuple[TestResult, ...]

    def counts(self) -> dict[str, int]:
        c = Counter(r.light.value for r in self.results)
        return {light.value: c.get(light.value, 0) for light in Light}

    def table(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def worst(self) -> Light:
        """Most severe evaluated light (ERROR outranks RED); NOT_EVALUATED only if nothing was evaluated."""
        order = [Light.ERROR, Light.RED, Light.AMBER, Light.GREEN]
        present = {r.light for r in self.results}
        for light in order:
            if light in present:
                return light
        return Light.NOT_EVALUATED


def run_suite(
    tasks: Iterable[Task],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> SuiteRun:
    """Run each ``(test_function, kwargs)`` task with the shared policy, model id and segment."""
    pol = policy if policy is not None else default_policy()
    results: list[TestResult] = []
    for func, kwargs in tasks:
        call = dict(kwargs)
        call.setdefault("policy", pol)
        if model_id is not None:
            call.setdefault("model_id", model_id)
        if segment is not None:
            call.setdefault("segment", segment)
        try:
            result = func(**call)
            if not isinstance(result, TestResult):
                raise TypeError(f"{func.__qualname__} returned {type(result).__name__}, not TestResult")
        except Exception as exc:  # recorded, never raised: the suite must finish
            s = getattr(func, "__rv_spec__", None)
            result = TestResult(
                test_id=s.test_id if s else func.__qualname__,
                h0=s.h0 if s else "not computed",
                reference=s.reference if s else "not computed",
                policy_version=pol.label,
                light=Light.ERROR,
                model_id=call.get("model_id"),
                segment=call.get("segment"),
                notes=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
    return SuiteRun(policy_version=pol.label, results=tuple(results))

__all__ = [
    "SuiteRun",
    "run_suite",
]
