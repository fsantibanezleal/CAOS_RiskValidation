"""The single record every validation test returns.

A validation result is evidence, not a verdict on compliance: it carries the statistic, the null hypothesis it
was tested against, the reference that defines it, and the traffic light assigned by a versioned *policy*.
Thresholds never live in the test code (see :mod:`riskvalidation.core.policy`).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class Light(str, enum.Enum):
    """Traffic light of one test result under a policy.

    ``NOT_EVALUATED`` means the policy holds no thresholds for the test, or the statistic is undefined (for
    example a calibration test on a grade with no obligors); no colour is shown for it. ``ERROR`` means the
    test raised; the exception is recorded in ``notes``.
    """

    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NOT_EVALUATED = "not_evaluated"
    ERROR = "error"


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class TestResult:
    """Outcome of one validation test on one model, segment and sample.

    Required: ``test_id``, ``h0`` (the null hypothesis or, for descriptive metrics, what is being measured),
    ``reference`` (the primary source that defines the test), ``policy_version`` and ``light``. A result that
    is not an error must carry at least one of ``statistic``, ``p_value`` or ``metric``.
    """

    __test__ = False  # not a pytest test class

    test_id: str
    h0: str
    reference: str
    policy_version: str
    light: Light
    statistic: float | None = None
    p_value: float | None = None
    metric: float | None = None
    alternative: str = ""
    alpha_amber: float | None = None
    alpha_red: float | None = None
    n: int | None = None
    n_events: int | None = None
    model_id: str | None = None
    segment: str | None = None
    inputs_hash: str = ""
    notes: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("test_id", "h0", "reference", "policy_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TestResult.{name} must be a non-empty string")
        override = getattr(self.policy_version, "override", None)
        if override is not None:  # a "+custom" label carries the cut points the row used: keep them in the row
            object.__setattr__(self, "policy_version", str(self.policy_version))
            object.__setattr__(self, "extras", {**dict(self.extras), "policy_override": dict(override)})
        if not isinstance(self.light, Light):
            raise TypeError("TestResult.light must be a Light")
        if self.p_value is not None and not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"TestResult.p_value must lie in [0, 1], got {self.p_value}")
        if self.light is not Light.ERROR and all(
            v is None for v in (self.statistic, self.p_value, self.metric)
        ):
            raise ValueError("a non-error TestResult needs a statistic, a p-value or a metric")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary; non-finite numbers become ``None``."""
        out: dict[str, Any] = {
            "test_id": self.test_id,
            "model_id": self.model_id,
            "segment": self.segment,
            "statistic": _finite_or_none(self.statistic),
            "p_value": _finite_or_none(self.p_value),
            "metric": _finite_or_none(self.metric),
            "h0": self.h0,
            "alternative": self.alternative,
            "policy_version": self.policy_version,
            "alpha_amber": _finite_or_none(self.alpha_amber),
            "alpha_red": _finite_or_none(self.alpha_red),
            "light": self.light.value,
            "n": self.n,
            "n_events": self.n_events,
            "inputs_hash": self.inputs_hash,
            "reference": self.reference,
            "notes": self.notes,
            "extras": {k: _jsonable(v) for k, v in self.extras.items()},
        }
        return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return _finite_or_none(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "tolist"):  # numpy scalars and arrays
        return _jsonable(v.tolist())
    return v

__all__ = [
    "Light",
    "TestResult",
]
