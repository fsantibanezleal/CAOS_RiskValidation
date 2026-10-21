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
from types import MappingProxyType
from typing import Any

from riskvalidation.core.result import Light

KINDS = ("p_value", "upper_bad", "lower_bad")


@dataclass(frozen=True)
class Threshold:
    """Two cut points for one test.

    - ``p_value``: red if ``p < red``, amber if ``red <= p < amber``, green otherwise (``red <= amber``).
    - ``upper_bad`` (larger is worse, for example PSI): red if ``m >= red``, amber if ``amber <= m < red``.
    - ``lower_bad`` (smaller is worse, for example AUC): red if ``m <= red``, amber if ``red < m <= amber``.

    A packaged entry may also carry ``parameters`` (cut points beyond the two, by name), ``rule`` (the name of a
    classifier of ``riskvalidation.policies.rules`` for a light that is not two cut points on one value), and
    ``by`` with ``cases`` (the entry is keyed by a field of the row: ``segment`` or ``extras.<key>``; ``cases`` maps
    each field value to the cut points that override ``amber``, ``red`` and ``parameters`` for that value).
    ``parameters`` name the cut points an entry already holds (or add the ones beyond the pair), so two thresholds
    with the same pair, rule and cases are equal whatever names they give their cut points.
    """

    kind: str
    amber: float
    red: float
    source: str = ""
    parameters: Mapping[str, float] = field(default_factory=dict, compare=False)
    rule: str = ""
    by: str = ""
    cases: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(
            self, "cases", MappingProxyType({str(k): MappingProxyType(dict(v)) for k, v in dict(self.cases).items()})
        )
        if bool(self.by) != bool(self.cases):
            raise ValueError("by and cases go together: a keyed entry names its field and holds one case per value")
        for name, value in self.parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"parameter {name!r} must be finite")
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

    def for_row(self, row: Any) -> Threshold | None:
        """The effective entry for a row: itself when not keyed, else the case the row's ``by`` field selects.

        ``None`` when the row does not state the field or the entry holds no case for its value.
        """
        if not self.by:
            return self
        value = _row_field(row, self.by)
        case = self.cases.get(str(value)) if value is not None else None
        if case is None:
            return None
        merged = {**self.parameters, **{k: v for k, v in case.items() if k not in ("amber", "red")}}
        return Threshold(
            kind=self.kind,
            amber=float(case.get("amber", self.amber)),
            red=float(case.get("red", self.red)),
            source=self.source,
            parameters=merged,
            rule=self.rule,
        )

    def classify_row(self, row: Any) -> Light:
        """The light of a result row under this entry: its ``rule`` when it names one, else the two cut points."""
        if self.rule:
            from riskvalidation.policies.rules import classify_rule

            return classify_rule(self.rule, self, row)
        value = _row_field(row, "p_value" if self.kind == "p_value" else "metric")
        return self.classify(value)


def _row_field(row: Any, name: str) -> Any:
    """A field of a result row (a mapping or an object): ``statistic``, ``segment``, ``extras.<key>``, and so on."""
    head, *rest = name.split(".")
    value = row.get(head) if isinstance(row, Mapping) else getattr(row, head, None)
    for key in rest:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


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

    def classify_row(self, test_id: str, row: Any) -> tuple[Light, Threshold | None]:
        """Light for a result row (a mapping or a ``TestResult``), through the entry's rule and its keyed case.

        ``NOT_EVALUATED`` with the entry (or ``None``) when the test has no entry or a keyed entry has no case for
        the row's field value. A test whose policy states no threshold by contract (``rules.RULE_ONLY``: the fairness
        flags and the ml estimate rows) is classified by its rule whatever the policy holds.
        """
        th = self.threshold(test_id)
        if th is None:
            from riskvalidation.policies.rules import RULE_ONLY, classify_rule

            if test_id not in RULE_ONLY:
                return Light.NOT_EVALUATED, None
            rule, constants = RULE_ONLY[test_id]  # no policy row by contract: the rule is the whole classification
            if rule == "worst_of_screens":
                return self._worst_of_screens(row), None
            return classify_rule(rule, Threshold("upper_bad", 0.0, 0.0, parameters=constants), row), None
        eff = th.for_row(row)
        if eff is None:
            return Light.NOT_EVALUATED, th
        if eff.rule == "worst_of_screens":
            return self._worst_of_screens(row), eff
        return eff.classify_row(row), eff

    def _worst_of_screens(self, row: Any) -> Light:
        """The worst light over the screens of a row (``extras.screens``: ``key`` and ``value`` each), each read by the
        entry of its own key (``ml.leakage_screen.temporal_order`` and so on)."""
        screens = _row_field(row, "extras.screens")
        if not isinstance(screens, Mapping) or not screens:
            return Light.NOT_EVALUATED
        order = {Light.GREEN: 0, Light.AMBER: 1, Light.RED: 2}
        worst = Light.GREEN
        for screen in screens.values():
            th = self.threshold(str(screen.get("key")))
            if th is None:
                return Light.NOT_EVALUATED
            light = th.classify(screen.get("value"))
            if light is Light.NOT_EVALUATED:
                return light
            if order[light] > order[worst]:
                worst = light
        return worst


def policy_from_mapping(data: Mapping[str, Any]) -> Policy:
    try:
        thresholds = {
            str(tid): Threshold(
                kind=str(spec["kind"]),
                amber=float(spec["amber"]),
                red=float(spec["red"]),
                source=str(spec.get("source", "")),
                parameters={str(k): float(v) for k, v in dict(spec.get("parameters", {})).items()},
                rule=str(spec.get("rule", "")),
                by=str(spec.get("by", "")),
                cases={
                    str(k): {str(n): float(x) for n, x in dict(v).items()}
                    for k, v in dict(spec.get("cases", {})).items()
                },
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


def packaged_labels() -> tuple[str, ...]:
    """The labels of the packaged policies (``default`` and one per Contraste battery), sorted."""
    labels = []
    for entry in resources.files("riskvalidation.policies").iterdir():
        if entry.name.endswith(".json"):
            data = json.loads(entry.read_text("utf-8"))
            labels.append(f"{data['policy_id']} {data['version']}")
    return tuple(sorted(labels))


_PACKAGED_CACHE: dict[str, Policy] = {}


def packaged_policy(label: str) -> Policy:
    """Load a packaged policy by its label (``<policy_id> <version>``, for example ``contraste-market-battery 1.0.0``).

    The file is ``policies/<policy_id>-<version>.json`` (``default.json`` for the default).
    ``KeyError`` naming the label and the packaged labels when it is absent.
    """
    if label in _PACKAGED_CACHE:
        return _PACKAGED_CACHE[label]
    if label == default_policy().label:
        _PACKAGED_CACHE[label] = default_policy()
        return _PACKAGED_CACHE[label]
    policy_id, _, version = str(label).partition(" ")
    name = f"{policy_id}-{version}.json"
    root = resources.files("riskvalidation.policies")
    if not policy_id or not version or "/" in name or "\\" in name or not root.joinpath(name).is_file():
        raise KeyError(f"no packaged policy {label!r}; the packaged labels are {list(packaged_labels())}")
    policy = policy_from_mapping(json.loads(root.joinpath(name).read_text("utf-8")))
    if policy.label != label:
        raise KeyError(
            f"the file {name} holds {policy.label!r}, not {label!r}; packaged labels: {list(packaged_labels())}"
        )
    _PACKAGED_CACHE[label] = policy
    return policy
