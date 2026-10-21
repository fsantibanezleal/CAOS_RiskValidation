"""Back-test series of the market tests: VaR exceedances, ES losses and probability integral transforms, each with a
null that holds exactly and one stated departure (U16-E2 follow-up, BL-088).

- ``VarHits``: ``t`` days of VaR exceedances at level ``alpha`` (with a VaR series for the dynamic quantile test). The
  null is independent Bernoulli(alpha) hits; ``rate_factor`` multiplies the hit probability and ``cluster`` is the
  probability of a hit the day after a hit (a first-order Markov chain with the same unconditional hit rate).
- ``EsLosses``: ``t`` standard normal losses against the true VaR and ES of level ``alpha``; ``scale`` above 1 is a
  forecast that understates the risk.
- ``PitSeries``: ``t`` probability integral transforms of the realised losses, uniform under the null; ``tail_shape``
  above 1 puts too much mass in the lower tail (``u = r ** tail_shape``) and ``cluster`` makes the tail hits
  (``u < alpha``) cluster in time, leaving the marginal law uniform.

Where ``rate_factor``, ``cluster``, ``scale`` and ``tail_shape`` are at their defaults each ``draw`` consumes the stream
exactly as the inline loops of ``tests/validation_market/test_market_size.py`` did (hits, then the VaR series), so a
suite that loops ``draw`` on one stream reproduces its earlier rejection counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import stats

from riskvalidation.generators._base import Check, GeneratorBase, mean_check, proportion_check


def _level(name: str, value: float) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return float(value)


def _chain(u: np.ndarray, p: float, cluster: float) -> np.ndarray:
    """A two-state chain with P(hit) = p and P(hit | hit yesterday) = cluster, driven by the uniforms ``u``."""
    if not p <= cluster < 1.0:
        raise ValueError("cluster is the probability of a hit after a hit: it must lie in [alpha, 1)")
    p01 = p * (1.0 - cluster) / (1.0 - p)
    hits = np.empty(u.size, dtype=bool)
    prev = u[0] < p
    hits[0] = prev
    for i in range(1, u.size):
        prev = u[i] < (cluster if prev else p01)
        hits[i] = prev
    return hits


@dataclass(frozen=True)
class VarHits(GeneratorBase):
    """``t`` days of VaR exceedances at ``alpha`` (see the module docstring); ``with_var`` also draws the VaR series
    ``-0.02 + 0.004 Z`` that the dynamic quantile test regresses on."""

    name: ClassVar[str] = "var_hits"
    t: int
    alpha: float
    rate_factor: float = 1.0
    cluster: float | None = None
    with_var: bool = True

    def __post_init__(self) -> None:
        if int(self.t) != self.t or int(self.t) < 20:
            raise ValueError("t must be an integer of at least 20")
        object.__setattr__(self, "t", int(self.t))
        _level("alpha", self.alpha)
        if not float(self.rate_factor) > 0.0 or self.hit_rate >= 1.0:
            raise ValueError("rate_factor must be positive and keep the hit rate below 1")
        if self.cluster is not None:
            _chain(np.array([0.5, 0.5]), self.hit_rate, float(self.cluster))  # the range check

    @property
    def hit_rate(self) -> float:
        return float(self.alpha * self.rate_factor)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        u = rng.random(self.t)
        hits = u < self.hit_rate if self.cluster is None else _chain(u, self.hit_rate, float(self.cluster))
        out: dict[str, Any] = {"exceedances": hits, "hits": hits, "alpha": float(self.alpha)}
        if self.with_var:
            out["var"] = -0.02 + 0.004 * rng.standard_normal(self.t)
        return out

    def truth(self) -> dict[str, Any]:
        return {"hit_rate": self.hit_rate, "nominal_hit_rate": float(self.alpha),
                "p_hit_after_hit": self.hit_rate if self.cluster is None else float(self.cluster),
                "null_holds": self.rate_factor == 1.0 and self.cluster is None}

    def verify(self, n_draws: int = 300, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 441]))
        runs = [self.draw(rng)["hits"] for _ in range(n_draws)]
        hits = np.concatenate(runs)
        tr = self.truth()
        prev = np.concatenate([h[:-1] for h in runs])
        nxt = np.concatenate([h[1:] for h in runs])
        return [proportion_check("hit rate", tr["hit_rate"], hits),
                proportion_check("hit after a hit", tr["p_hit_after_hit"], nxt[prev])]


@dataclass(frozen=True)
class EsLosses(GeneratorBase):
    """``t`` standard normal losses with the true VaR and ES of level ``alpha`` as forecasts; the realised losses are
    multiplied by ``scale`` (above 1: the forecasts understate the risk)."""

    name: ClassVar[str] = "es_losses"
    t: int = 250
    alpha: float = 0.025
    scale: float = 1.0

    def __post_init__(self) -> None:
        if int(self.t) != self.t or int(self.t) < 20:
            raise ValueError("t must be an integer of at least 20")
        object.__setattr__(self, "t", int(self.t))
        _level("alpha", self.alpha)
        if not float(self.scale) > 0.0:
            raise ValueError("scale must be positive")

    @property
    def var_z(self) -> float:
        return float(stats.norm.ppf(1.0 - self.alpha))

    @property
    def es_z(self) -> float:
        return float(stats.norm.pdf(self.var_z) / self.alpha)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        t = self.t
        z = rng.standard_normal(t)
        return {"loss": z if self.scale == 1.0 else self.scale * z,
                "var": np.full(t, self.var_z), "es": np.full(t, self.es_z), "alpha": float(self.alpha),
                "simulate_null": lambda g: g.standard_normal(t)}

    def truth(self) -> dict[str, Any]:
        # P(scale Z > VaR) and E[scale Z | scale Z > VaR] against the forecast ES
        exceed = float(stats.norm.sf(self.var_z / self.scale))
        return {"exceedance_probability": exceed, "nominal_exceedance_probability": float(self.alpha),
                "null_holds": self.scale == 1.0}

    def verify(self, n_draws: int = 500, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 442]))
        d = [self.draw(rng) for _ in range(n_draws)]
        loss = np.concatenate([x["loss"] for x in d])
        return [proportion_check("exceedance frequency", self.truth()["exceedance_probability"],
                                 loss > self.var_z),
                mean_check("mean loss", 0.0, loss)]


@dataclass(frozen=True)
class PitSeries(GeneratorBase):
    """``t`` probability integral transforms; ``alpha`` is the level of the tail hits ``u < alpha`` of the ES tests."""

    name: ClassVar[str] = "pit_series"
    t: int
    alpha: float
    tail_shape: float = 1.0
    cluster: float | None = None

    def __post_init__(self) -> None:
        if int(self.t) != self.t or int(self.t) < 20:
            raise ValueError("t must be an integer of at least 20")
        object.__setattr__(self, "t", int(self.t))
        _level("alpha", self.alpha)
        if not float(self.tail_shape) >= 1.0:
            raise ValueError("tail_shape must be at least 1 (1 is uniform)")
        if self.cluster is not None:
            _chain(np.array([0.5, 0.5]), float(self.alpha), float(self.cluster))
            if self.tail_shape != 1.0:
                raise ValueError("a PIT series departs from the null by tail_shape or by cluster, not both")

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        if self.cluster is None:
            r = rng.random(self.t)
            return {"pit": r if self.tail_shape == 1.0 else r**self.tail_shape, "alpha": float(self.alpha)}
        hit = _chain(rng.random(self.t), float(self.alpha), float(self.cluster))
        r = rng.random(self.t)
        return {"pit": np.where(hit, self.alpha * r, self.alpha + (1.0 - self.alpha) * r), "alpha": float(self.alpha)}

    def truth(self) -> dict[str, Any]:
        a = float(self.alpha)
        mass = a ** (1.0 / self.tail_shape)  # P(u < alpha) for u = r ** shape
        # E[H] with H = (alpha - u) 1(u < alpha): the null value is alpha / 2 (times alpha), as in Du and Escanciano
        return {"p_tail_hit": mass, "nominal_p_tail_hit": a, "p_hit_after_hit": mass if self.cluster is None else
                float(self.cluster), "null_holds": self.tail_shape == 1.0 and self.cluster is None}

    def verify(self, n_draws: int = 300, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 443]))
        runs = [self.draw(rng)["pit"] < self.alpha for _ in range(n_draws)]
        hits = np.concatenate(runs)
        tr = self.truth()
        prev = np.concatenate([h[:-1] for h in runs])
        nxt = np.concatenate([h[1:] for h in runs])
        return [proportion_check("tail hit rate", tr["p_tail_hit"], hits),
                proportion_check("tail hit after a tail hit", tr["p_hit_after_hit"], nxt[prev])]


__all__ = ["EsLosses", "PitSeries", "VarHits"]
