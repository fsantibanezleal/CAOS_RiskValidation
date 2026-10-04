"""Versioned validation policy: the thresholds that turn a statistic into a traffic light.

Thresholds are an institution's policy choice, not regulation. The ECB's 2019 instructions for reporting IRB
validation results define the tests and statistics but set no pass or fail thresholds; the PSI bands of 0.10
and 0.25 are an industry convention with no control of the type I error (Yurdakul and Naranjo 2020); BCBS
Working Paper 14 (2005) notes that no really powerful PD calibration test exists once defaults are correlated.
A policy therefore lives in a versioned data file, is shown to users as "policy", and travels with every
result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from riskvalidation.core.result import Light

KINDS = ("p_value", "upper_bad", "lower_bad")


@dataclass(frozen=True)
class Threshold:
    """Two cut points for one test.

    - ``p_value``: red if ``p < red``, amber if ``red <= p < amber``, green otherwise (``red <= amber``).
    - ``upper_bad`` (larger is worse, for example PSI): red if ``m >= red``, amber if ``amber <= m < red``.
    - ``lower_bad`` (smaller is worse, for example AUC): red if ``m <= red``, amber if ``red < m <= amber``.
    """

    kind: str
    amber: float
    red: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"threshold kind must be one of {KINDS}, got {self.kind!r}")
        if not (math.isfinite(self.amber) and math.isfinite(self.red)):
            raise ValueError("thresholds must be finite")
        if self.kind in ("p_value", "lower_bad") and self.red > self.amber:
            raise ValueError(f"{self.kind}: red ({self.red}) must not exceed amber ({self.amber})")
        if self.kind == "upper_bad" and self.red < self.amber:
            raise ValueError(f"upper_bad: red ({self.red}) must not be below amber ({self.amber})")

    def classify(self, value: float | None) -> Light:
        if value is None or not math.isfinite(float(value)):
            return Light.NOT_EVALUATED
        v = float(value)
        if self.kind == "p_value":
            if v < self.red:
                return Light.RED
            return Light.AMBER if v < self.amber else Light.GREEN
        if self.kind == "upper_bad":
            if v >= self.red:
                return Light.RED
            return Light.AMBER if v >= self.amber else Light.GREEN
        if v <= self.red:  # lower_bad
            return Light.RED
        return Light.AMBER if v <= self.amber else Light.GREEN


@dataclass(frozen=True)
class Policy:
    """A named, versioned set of thresholds keyed by test id."""

    policy_id: str
    version: str
    description: str
    thresholds: Mapping[str, Threshold] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.policy_id} {self.version}"

    def threshold(self, test_id: str) -> Threshold | None:
        return self.thresholds.get(test_id)

    def classify(
        self, test_id: str, *, p_value: float | None = None, metric: float | None = None
    ) -> tuple[Light, Threshold | None]:
        """Light for a result. ``p_value`` thresholds read the p-value; the others read the metric."""
        th = self.threshold(test_id)
        if th is None:
            return Light.NOT_EVALUATED, None
        value = p_value if th.kind == "p_value" else metric
        return th.classify(value), th


def policy_from_mapping(data: Mapping[str, Any]) -> Policy:
    try:
        thresholds = {
            str(tid): Threshold(
                kind=str(spec["kind"]),
                amber=float(spec["amber"]),
                red=float(spec["red"]),
                source=str(spec.get("source", "")),
            )
            for tid, spec in dict(data.get("thresholds", {})).items()
        }
        return Policy(
            policy_id=str(data["policy_id"]),
            version=str(data["version"]),
            description=str(data.get("description", "")),
            thresholds=thresholds,
        )
    except KeyError as exc:
        raise ValueError(f"policy file is missing the field {exc.args[0]!r}") from exc


def load_policy(path: str | Path) -> Policy:
    """Load a policy from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return policy_from_mapping(json.load(fh))


_DEFAULT: Policy | None = None


def default_policy() -> Policy:
    """The packaged default policy (conventional thresholds, documented per test)."""
    global _DEFAULT
    if _DEFAULT is None:
        text = resources.files("riskvalidation.policies").joinpath("default.json").read_text("utf-8")
        _DEFAULT = policy_from_mapping(json.loads(text))
    return _DEFAULT
