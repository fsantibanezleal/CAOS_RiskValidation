"""Planted-bias generators for the size and power of the fairness tests (SDD l.335; dossier 07 D pattern 6): a
group-dependent decision threshold, group-dependent label noise and a proxy feature, each built on
`generators.LogisticPortfolio`, each with `truth()` (exact) and `verify()` (a large draw reproduces the truth within
Monte Carlo error) as the generator contract asks.

Conventions of a draw: `group` is 1 for the protected group and 0 for the reference group; `decision` is 1 for the
favourable decision (the model approves the obligors of low linear predictor); `defaults` is the default indicator (1
is a default) and `y` the outcome aligned with the decision (1 repaid, `y = 1 - defaults`), both as observed (the
label-noise plant flips them for the protected group; `y_true` and `defaults_true` are the unflipped ones); `pd` is
the model's PD (calibrated) and `risk` its linear predictor. The seed of a plant is the stream of `draw_default()` and
of the default `verify()`; the harness draws from its own streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from scipy import special, stats

from riskvalidation.generators._base import Check, GeneratorBase, mean_check
from riskvalidation.generators.scores import LogisticPortfolio

DEFAULT_COEFFICIENTS = (0.8, 0.5, 0.3)


def _check_common(n: int, base_rate: float, share: float) -> None:
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be an integer of at least 2, got {n!r}")
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must lie strictly between 0 and 1, got {base_rate!r}")
    if not 0.0 < share < 1.0:
        raise ValueError(f"protected_share must lie strictly between 0 and 1, got {share!r}")


class _PortfolioPlant(GeneratorBase):
    """What the three plants share: the portfolio, the stream of a draw and the decision cutoff."""

    n: int
    seed: int
    base_rate: float
    protected_share: float
    intercept: float
    coefficients: Any
    correlation: float

    @cached_property
    def _portfolio(self) -> LogisticPortfolio:
        return LogisticPortfolio(n=self.n, intercept=self.intercept, coefficients=self.coefficients,
                                 correlation=self.correlation)

    @cached_property
    def _moments(self) -> tuple[float, float]:
        """Mean and standard deviation of the linear predictor, which is normal."""
        return self._portfolio._eta_moments()

    def _cutoff(self) -> float:
        """The linear-predictor cutoff that approves `base_rate` of the obligors (eta is normal, exactly)."""
        m, s = self._moments
        return m + s * float(stats.norm.ppf(self.base_rate))

    def _base_draw(self, rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray]:
        d = self._portfolio.draw(rng)
        group = (rng.random(self.n) < self.protected_share).astype(np.int64)
        return d, group

    def draw_default(self) -> dict[str, Any]:
        """One dataset drawn from the plant's own seed."""
        return self.draw(np.random.default_rng(self.seed))  # type: ignore[attr-defined]

    def _stream(self, seed: int | None, salt: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed if seed is None else seed, salt]))


@dataclass(frozen=True)
class ThresholdBiasPlant(_PortfolioPlant):
    """The model's linear predictor is the same for both groups but the protected group is approved below a cutoff
    lower by `gap` (in log-odds): a group-dependent decision threshold. `base_rate` is the reference group's selection
    rate; `gap` 0 is the null. `truth()` gives the exact selection rates, since the linear predictor is normal."""

    name: ClassVar[str] = "threshold_bias_plant"
    n: int
    gap: float
    seed: int
    base_rate: float = 0.7
    protected_share: float = 0.5
    intercept: float = -1.0
    coefficients: Any = DEFAULT_COEFFICIENTS
    correlation: float = 0.3

    def __post_init__(self) -> None:
        _check_common(self.n, self.base_rate, self.protected_share)
        if not (math.isfinite(self.gap) and self.gap >= 0.0):
            raise ValueError(f"gap must be a non-negative number, got {self.gap!r}")
        object.__setattr__(self, "n", int(self.n))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        d, group = self._base_draw(rng)
        eta = d["risk"]
        decision = (eta <= self._cutoff() - self.gap * group).astype(np.int64)
        defaults = d["defaults"].astype(np.int64)
        return {"group": group, "decision": decision, "y": 1 - defaults, "defaults": defaults, "pd": d["pd"],
                "risk": eta, "x": d["x"]}

    def truth(self) -> dict[str, Any]:
        s = self._moments[1]
        ref = float(self.base_rate)
        prot = ref if self.gap == 0.0 else float(stats.norm.cdf(stats.norm.ppf(self.base_rate) - self.gap / s))
        return {"selection_rate_reference": ref, "selection_rate_protected": prot,
                "selection_rate_gap": ref - prot if self.gap > 0.0 else 0.0, "gap": float(self.gap),
                "protected_share": float(self.protected_share), "null": self.gap == 0.0}

    def verify(self, n_draws: int = 200, seed: int | None = None) -> list[Check]:
        rng = self._stream(seed, 1469)
        tr = self.truth()
        ref, prot, share = [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            ref.append(d["decision"][d["group"] == 0].mean())
            prot.append(d["decision"][d["group"] == 1].mean())
            share.append(d["group"].mean())
        return [mean_check("selection rate, reference group", tr["selection_rate_reference"], np.array(ref)),
                mean_check("selection rate, protected group", tr["selection_rate_protected"], np.array(prot)),
                mean_check("protected share", tr["protected_share"], np.array(share))]


@dataclass(frozen=True)
class LabelNoiseBiasPlant(_PortfolioPlant):
    """The model decides identically for both groups, but the observed outcome of the protected group is wrong at the
    rate `noise`: each of its labels is flipped with that probability, so the error rates measured against the observed
    outcome differ between groups although the model treats them alike. `noise` is in [0, 0.5]; 0 is the null."""

    name: ClassVar[str] = "label_noise_bias_plant"
    n: int
    noise: float
    seed: int
    base_rate: float = 0.7
    protected_share: float = 0.5
    intercept: float = -1.0
    coefficients: Any = DEFAULT_COEFFICIENTS
    correlation: float = 0.3

    def __post_init__(self) -> None:
        _check_common(self.n, self.base_rate, self.protected_share)
        if not (math.isfinite(self.noise) and 0.0 <= self.noise <= 0.5):
            raise ValueError(f"noise must lie in [0, 0.5], got {self.noise!r}")
        object.__setattr__(self, "n", int(self.n))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        d, group = self._base_draw(rng)
        eta = d["risk"]
        decision = (eta <= self._cutoff()).astype(np.int64)
        true_defaults = d["defaults"].astype(np.int64)
        flips = ((rng.random(self.n) < self.noise) & (group == 1)).astype(np.int64)
        defaults = true_defaults ^ flips
        return {"group": group, "decision": decision, "y": 1 - defaults, "y_true": 1 - true_defaults,
                "defaults": defaults, "defaults_true": true_defaults, "pd": d["pd"], "risk": eta, "x": d["x"]}

    def truth(self) -> dict[str, Any]:
        return {"flip_rate": float(self.noise), "flipped_group": 1, "selection_rate": float(self.base_rate),
                "protected_share": float(self.protected_share)}

    def verify(self, n_draws: int = 200, seed: int | None = None) -> list[Check]:
        rng = self._stream(seed, 1470)
        tr = self.truth()
        prot, ref, rate = [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            flipped = d["y"] != d["y_true"]
            prot.append(flipped[d["group"] == 1].mean())
            ref.append(flipped[d["group"] == 0].mean())
            rate.append(d["decision"].mean())
        return [mean_check("flip rate, protected group", tr["flip_rate"], np.array(prot)),
                mean_check("flip rate, reference group", 0.0, np.array(ref)),
                mean_check("selection rate", tr["selection_rate"], np.array(rate))]


@dataclass(frozen=True)
class ProxyFeaturePlant(_PortfolioPlant):
    """A feature, the proxy, has correlation `proxy_strength` with the protected indicator, and the decision depends on
    the covariates and the proxy only (never on the protected indicator): the disparity is transmitted by the proxy
    alone. A decision model that controls for the covariates but not the proxy finds a protected effect; once the
    proxy is a control that effect is 0 exactly (`truth()['protected_effect_given_proxy']`). `proxy_strength` 0 is
    the null."""

    name: ClassVar[str] = "proxy_feature_plant"
    n: int
    proxy_strength: float
    seed: int
    proxy_effect: float = 1.0
    base_rate: float = 0.7
    protected_share: float = 0.5
    intercept: float = 0.2
    coefficients: Any = DEFAULT_COEFFICIENTS
    correlation: float = 0.3

    def __post_init__(self) -> None:
        _check_common(self.n, self.base_rate, self.protected_share)
        if not (math.isfinite(self.proxy_strength) and 0.0 <= self.proxy_strength < 1.0):
            raise ValueError(f"proxy_strength must lie in [0, 1), got {self.proxy_strength!r}")
        if not math.isfinite(self.proxy_effect):
            raise ValueError("proxy_effect must be finite")
        object.__setattr__(self, "n", int(self.n))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        d, group = self._base_draw(rng)
        share = self.protected_share
        standardised = (group - share) / math.sqrt(share * (1.0 - share))  # mean 0, variance 1
        proxy = self.proxy_strength * standardised + math.sqrt(1.0 - self.proxy_strength**2) * rng.standard_normal(
            self.n)
        x = d["x"]
        logit = self.intercept + x @ np.asarray(self.coefficients, dtype=float) + self.proxy_effect * proxy
        decision = (rng.random(self.n) < special.expit(logit)).astype(np.int64)
        return {"group": group, "decision": decision, "protected": group.astype(float)[:, None], "controls": x,
                "proxy": proxy, "controls_with_proxy": np.column_stack([x, proxy])}

    def truth(self) -> dict[str, Any]:
        return {"proxy": "proxy", "proxy_index": len(self.coefficients), "proxy_strength": float(self.proxy_strength),
                "proxy_effect": float(self.proxy_effect), "protected_effect_given_proxy": 0.0,
                "protected_share": float(self.protected_share)}

    def verify(self, n_draws: int = 200, seed: int | None = None) -> list[Check]:
        rng = self._stream(seed, 1471)
        tr = self.truth()
        corr, share, mean_proxy = [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            corr.append(np.corrcoef(d["proxy"], d["group"])[0, 1])
            share.append(d["group"].mean())
            mean_proxy.append(d["proxy"].mean())
        return [mean_check("correlation of the proxy with the protected indicator", tr["proxy_strength"],
                           np.array(corr)),
                mean_check("protected share", tr["protected_share"], np.array(share)),
                mean_check("mean of the proxy", 0.0, np.array(mean_proxy))]


__all__ = ["LabelNoiseBiasPlant", "ProxyFeaturePlant", "ThresholdBiasPlant"]
