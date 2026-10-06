r"""Rejection rates with their Monte Carlo error.

A simulated rejection rate is an estimate: with $n$ repetitions and $k$ rejections, $\hat r = k/n$ has the Monte Carlo
standard error $\sqrt{\hat r(1-\hat r)/n}$ (Morris, White and Crowther 2019, doi:10.1002/sim.8086, Table 6, "Rejection %
(power or type I error)"). The interval is Wilson's, recommended for binomial proportions by Brown, Cai and DasGupta
(2001, doi:10.1214/ss/1009213286) and well behaved near 0, where small sizes sit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Z95 = 1.959963984540054
Z_SIZE = 3.090232306167813  # one-sided 0.1%: a size above alpha by more than this many SEs is not Monte Carlo noise
Z_AGREE = 3.290526731491926  # two-sided 0.1%

MORRIS = "Morris, White and Crowther (2019), Using simulation studies to evaluate statistical methods, Statistics in " \
         "Medicine 38(11), doi:10.1002/sim.8086, Table 6"


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. At k = 0 and k = n the closed form gives exactly 0 and 1;
    those bounds are returned exactly, since ``centre - half`` leaves a residue of about 1e-20 for some n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


@dataclass(frozen=True)
class Rate:
    """``rejections`` out of ``n`` repetitions; ``undefined`` of them gave the rule no answer (counted as not
    rejected)."""

    rejections: int
    n: int
    undefined: int = 0

    def __post_init__(self) -> None:
        if self.n <= 0 or not 0 <= self.rejections <= self.n or not 0 <= self.undefined <= self.n:
            raise ValueError("invalid counts")

    @property
    def rate(self) -> float:
        return self.rejections / self.n

    @property
    def se(self) -> float:
        r = self.rate
        return math.sqrt(r * (1.0 - r) / self.n)

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.rejections, self.n)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.interval
        return {"rejections": self.rejections, "n": self.n, "undefined": self.undefined, "rate": self.rate,
                "se": self.se, "wilson_low": lo, "wilson_high": hi}


def size_bound(alpha: float, n: int, z: float = Z_SIZE) -> float:
    """The largest rate consistent with a true size of at most ``alpha``: alpha + z sqrt(alpha (1 - alpha) / n)."""
    return alpha + z * math.sqrt(alpha * (1.0 - alpha) / n)


def within_size(rate: Rate, alpha: float, z: float = Z_SIZE) -> bool:
    """Whether a rate measured under the null is consistent with a size at most ``alpha`` (one-sided)."""
    return rate.rate <= size_bound(alpha, rate.n, z)


def agrees(rate: Rate, exact: float, z: float = Z_AGREE) -> bool:
    """Whether a simulated rate agrees with an exact probability: |r - p| <= z sqrt(p (1 - p) / n), and equals it
    when p is 0 or 1."""
    if exact <= 0.0 or exact >= 1.0:
        return rate.rate == exact
    return abs(rate.rate - exact) <= z * math.sqrt(exact * (1.0 - exact) / rate.n)


def agrees_with_published(rate: Rate, published: float, n_published: int, z: float = Z_AGREE) -> tuple[bool, float]:
    """Whether a rate agrees with a published simulated rate of ``n_published`` runs: both estimate one true rate, so
    the difference has variance r (1 - r) (1/n + 1/n_published) at the pooled r. Returns (agrees, z of the
    difference)."""
    pooled = (rate.rejections + published * n_published) / (rate.n + n_published)
    se = math.sqrt(max(pooled * (1.0 - pooled), 0.0) * (1.0 / rate.n + 1.0 / n_published))
    diff = rate.rate - published
    if se == 0.0:
        return diff == 0.0, 0.0 if diff == 0.0 else math.inf
    zz = diff / se
    return abs(zz) <= z, zz


__all__ = ["MORRIS", "Rate", "Z95", "Z_AGREE", "Z_SIZE", "agrees", "agrees_with_published", "size_bound",
           "wilson_interval", "within_size"]
