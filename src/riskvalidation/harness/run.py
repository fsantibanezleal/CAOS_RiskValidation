"""Run a registered test over repetitions of a generator and measure its rejection rates.

One seed per study, one independent stream per repetition (Morris, White and Crowther 2019, section 4.1:
"set the seed at the beginning, once and only once", and separate streams so repetitions never share random numbers):
``SeedSequence(seed).spawn(n_rep)`` gives repetition ``i`` its own generator, so it re-runs alone to the same data. A
test with its own randomness receives a seed drawn from the repetition's stream after the data, never a constant
(section 4.1: check "that such programs do not manipulate the seed").
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from riskvalidation.core.registry import spec as get_spec
from riskvalidation.core.result import TestResult
from riskvalidation.generators._base import jsonable
from riskvalidation.harness.rates import Rate

CACHE_LIMIT = 256  # inputs with at most this many values are cached (a deterministic test repeats them often)


@dataclass(frozen=True)
class Rule:
    """A decision read from a result: ``True`` rejects, ``False`` does not, ``None`` is undefined."""

    label: str
    decide: Callable[[TestResult], bool | None]


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def p_below(alpha: float) -> Rule:
    """Reject when the p-value is below ``alpha`` (the engine's light rule: amber below 5%, red below 1%)."""
    return Rule(f"p<{alpha:g}", lambda r: (r.p_value < alpha) if _finite(r.p_value) else None)


def metric_above(threshold: float, label: str | None = None) -> Rule:
    """Reject when the metric exceeds ``threshold`` (for example the PSI's 0.10 and 0.25 rules of thumb)."""
    return Rule(label or f"metric>{threshold:g}", lambda r: (r.metric > threshold) if _finite(r.metric) else None)


def light_not_green(label: str = "light not green") -> Rule:
    """Reject when the policy light is amber or red."""
    def decide(r: TestResult) -> bool | None:
        if r.light.value in ("not_evaluated", "error"):
            return None
        return r.light.value in ("amber", "red")

    return Rule(label, decide)


@dataclass
class Simulation:
    """The outcome of :func:`simulate`: rates per rule, the p-values, metrics and statistics of every repetition."""

    test_id: str
    generator: dict[str, Any]
    truth: dict[str, Any]
    n_rep: int
    seed: int
    test_kwargs: dict[str, Any]
    rates: dict[str, Rate]
    p_values: np.ndarray
    metrics: np.ndarray
    statistics: np.ndarray
    seed_injected: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def rate(self, rule: str | float) -> Rate:
        """The rate of a rule by its label, or of ``p < alpha`` by the level."""
        key = rule if isinstance(rule, str) else f"p<{rule:g}"
        return self.rates[key]

    def p_histogram(self, bins: int = 20) -> list[int]:
        """Counts of the defined p-values in ``bins`` equal bins of [0, 1] (flat under an exact null)."""
        p = self.p_values[np.isfinite(self.p_values)]
        return np.histogram(p, bins=bins, range=(0.0, 1.0))[0].astype(int).tolist()

    def to_dict(self, histogram_bins: int = 20) -> dict[str, Any]:
        return jsonable({
            "test_id": self.test_id,
            "generator": self.generator,
            "truth": self.truth,
            "n_rep": self.n_rep,
            "seed": self.seed,
            "test_kwargs": self.test_kwargs,
            "seed_injected": self.seed_injected,
            "rates": {k: v.to_dict() for k, v in self.rates.items()},
            "undefined": int(np.sum(~np.isfinite(self.p_values))) if self.p_values.size else 0,
            "p_histogram": self.p_histogram(histogram_bins),
            "extras": self.extras,
        })


def _key(kwargs: Mapping[str, Any]) -> tuple | None:
    items = []
    for k in sorted(kwargs):
        v = kwargs[k]
        if isinstance(v, np.ndarray):
            if v.size > CACHE_LIMIT:
                return None
            items.append((k, v.shape, v.dtype.str, v.tobytes()))
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            if arr.dtype == object or arr.size > CACHE_LIMIT:
                return None
            items.append((k, arr.shape, arr.dtype.str, arr.tobytes()))
        elif isinstance(v, (int, float, str, bool, np.integer, np.floating)) or v is None:
            items.append((k, repr(v)))
        else:
            return None
    return tuple(items)


def _resolve(test: str | Callable[..., TestResult]) -> tuple[str | None, Callable[..., TestResult]]:
    if isinstance(test, str):
        s = get_spec(test)
        return s.test_id, s.func
    s = getattr(test, "__rv_spec__", None)
    return (s.test_id if s is not None else None), test


def simulate(
    test: str | Callable[..., TestResult],
    generator: Any,
    *,
    n_rep: int,
    seed: int,
    feed: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    alphas: Sequence[float] = (0.05, 0.01),
    rules: Sequence[Rule] = (),
    test_kwargs: Mapping[str, Any] | None = None,
    cache: bool = True,
) -> Simulation:
    """Draw ``n_rep`` datasets from ``generator``, run ``test`` on each and count rejections per level and rule.

    ``feed`` maps a draw to the test's arguments (default: the draw's keys the test accepts); keys the test does not
    accept are dropped, so one feed serves several tests. ``test_kwargs`` are added to every call. A result without a
    p-value (or metric, for a metric rule) is undefined: not a rejection, and counted.
    """
    if n_rep < 1:
        raise ValueError("n_rep must be positive")
    test_id, func = _resolve(test)
    params = inspect.signature(func).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    fixed = dict(test_kwargs or {})
    inject = "seed" in params and "seed" not in fixed
    all_rules = [p_below(a) for a in alphas] + list(rules)
    counts = {r.label: 0 for r in all_rules}
    undefined = {r.label: 0 for r in all_rules}
    pv = np.full(n_rep, np.nan)
    mv = np.full(n_rep, np.nan)
    sv = np.full(n_rep, np.nan)
    memo: dict[tuple, TestResult] = {}
    children = np.random.SeedSequence(seed).spawn(n_rep)
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        draw = generator.draw(rng)
        raw = feed(draw) if feed is not None else draw
        kwargs = {k: v for k, v in raw.items() if accepts_any or k in params}
        kwargs.update(fixed)
        if inject:
            kwargs["seed"] = int(rng.integers(0, 2**63 - 1))
        key = _key(kwargs) if cache and not inject else None
        result = memo.get(key) if key is not None else None
        if result is None:
            result = func(**kwargs)
            if key is not None:
                memo[key] = result
        if test_id is None:
            test_id = result.test_id
        pv[i] = result.p_value if _finite(result.p_value) else np.nan
        mv[i] = result.metric if _finite(result.metric) else np.nan
        sv[i] = result.statistic if _finite(result.statistic) else np.nan
        for r in all_rules:
            v = r.decide(result)
            if v is None:
                undefined[r.label] += 1
            elif v:
                counts[r.label] += 1
    rates = {lab: Rate(counts[lab], n_rep, undefined[lab]) for lab in counts}
    truth = generator.truth() if hasattr(generator, "truth") else {}
    config = generator.config() if hasattr(generator, "config") else {"generator": type(generator).__name__}
    return Simulation(test_id=test_id or "", generator=config, truth=truth, n_rep=n_rep, seed=seed,
                      test_kwargs=jsonable(fixed), rates=rates, p_values=pv, metrics=mv, statistics=sv,
                      seed_injected=inject, extras={"cached_inputs": len(memo)})


__all__ = ["CACHE_LIMIT", "Rule", "Simulation", "light_not_green", "metric_above", "p_below", "simulate"]
