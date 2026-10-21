"""Monte Carlo generators with an exact anchor: the null of a convergence test is a closed form, never a second run of
the simulation it checks (U16-E2, BL-088).

- ``McQuantile``: the ``level``-quantile of the loss rate of a large homogeneous portfolio, estimated from
  ``scenarios`` draws of the systematic factor with the standard error of the sample quantile (Woodruff's
  order-statistic interval). The anchor is the closed form ``Phi((Phi^-1(pd) + sqrt(rho) Phi^-1(q)) / sqrt(1 - rho))``
  (Vasicek 2002), computed here from ``scipy.special`` and not from the engine; ``mc_rho`` different from ``rho``
  simulates a Monte Carlo run of another model than the anchor's.
- ``CreditRiskPlusSample``: a Monte Carlo sample of the Poisson-Gamma losses of a one-sector CreditRisk+ portfolio next
  to the pmf of the recursion (``pmf_stable``). The sample is drawn from the model's definition (the Gamma factor, then
  Poisson counts by exposure band), which shares no step with the recursion; ``mc_sigma2`` different from ``sigma2``
  simulates a sample of another sector variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
from scipy import special

from riskvalidation.generators._base import Check, GeneratorBase, mean_check, variance_check
from riskvalidation.portfolio.creditrisk_plus import CreditRiskPlusInput, pmf_stable

WOODRUFF_Z = 1.959963984540054


def lhp_quantile_closed_form(level: float, pd: float, rho: float) -> float:
    """The ``level``-quantile of the large-portfolio loss rate (Vasicek 2002), straight from the normal functions."""
    return float(special.ndtr((special.ndtri(pd) + math.sqrt(rho) * special.ndtri(level)) / math.sqrt(1.0 - rho)))


def sample_quantile_with_se(x: np.ndarray, level: float, z: float = WOODRUFF_Z) -> tuple[float, float]:
    """The sample ``level``-quantile (linear interpolation) and its standard error by Woodruff's method: the
    distribution-free interval for the quantile has ranks ``S q +- z sqrt(S q (1 - q))``; its half width over ``z`` is
    the standard error (Woodruff 1952, JASA 47(260):635-646)."""
    s = np.sort(np.asarray(x, dtype=float))
    n = s.size
    est = float(np.quantile(s, level))
    half = z * math.sqrt(n * level * (1.0 - level))
    lo = max(int(math.floor(n * level - half)), 0)
    hi = min(int(math.ceil(n * level + half)), n - 1)
    return est, float((s[hi] - s[lo]) / (2.0 * z))


@dataclass(frozen=True)
class McQuantile(GeneratorBase):
    """``scenarios`` draws of the LHP loss rate at correlation ``mc_rho`` (default ``rho``) against the anchor at
    ``rho``."""

    name: ClassVar[str] = "mc_quantile"
    scenarios: int
    level: float = 0.99
    pd: float = 0.02
    rho: float = 0.12
    mc_rho: float | None = None
    se_method: str = "woodruff"

    def __post_init__(self) -> None:
        if int(self.scenarios) != self.scenarios or int(self.scenarios) < 100:
            raise ValueError("scenarios must be an integer of at least 100")
        object.__setattr__(self, "scenarios", int(self.scenarios))
        for name in ("level", "pd", "rho"):
            if not 0.0 < float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.mc_rho is not None and not 0.0 < float(self.mc_rho) < 1.0:
            raise ValueError("mc_rho must lie strictly between 0 and 1")
        if self.se_method not in ("woodruff", "asymptotic"):
            raise ValueError("se_method is 'woodruff' (estimated from the sample) or 'asymptotic' (exact)")

    @property
    def simulated_rho(self) -> float:
        return float(self.rho if self.mc_rho is None else self.mc_rho)

    @property
    def anchor(self) -> float:
        return lhp_quantile_closed_form(self.level, self.pd, self.rho)

    @property
    def asymptotic_se(self) -> float:
        """The standard error of the sample quantile, ``sqrt(q (1 - q) / S) / f(x_q)``, from the model's own density
        (the diagnostic: no estimation noise in the standard error)."""
        r = self.simulated_rho
        x = lhp_quantile_closed_form(self.level, self.pd, r)
        g = float(special.ndtri(x))
        pdf = math.sqrt((1.0 - r) / r) * math.exp(0.5 * g * g
                                                  - (math.sqrt(1.0 - r) * g - float(special.ndtri(self.pd))) ** 2
                                                  / (2.0 * r))
        return math.sqrt(self.level * (1.0 - self.level) / self.scenarios) / pdf

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        r = self.simulated_rho
        z = rng.standard_normal(self.scenarios)
        rates = special.ndtr((special.ndtri(self.pd) + math.sqrt(r) * z) / math.sqrt(1.0 - r))
        est, se = sample_quantile_with_se(rates, self.level)
        if self.se_method == "asymptotic":
            se = self.asymptotic_se
        return {"estimate": est, "se": se, "anchor": self.anchor, "quantity": "var"}

    def truth(self) -> dict[str, Any]:
        return {"anchor": self.anchor, "simulated_quantile": lhp_quantile_closed_form(self.level, self.pd,
                                                                                      self.simulated_rho),
                "null_holds": self.simulated_rho == self.rho}

    def verify(self, n_draws: int = 2_000, seed: int = 20261021) -> list[Check]:
        """The estimates centre on the simulated model's quantile, and the Woodruff standard error is the standard
        deviation of the estimates (within the Monte Carlo error of both)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 431]))
        d = [self.draw(rng) for _ in range(n_draws)]
        est = np.array([x["estimate"] for x in d])
        se = np.array([x["se"] for x in d])
        sd = float(est.std(ddof=1))
        mean_se = float(se.mean())
        return [mean_check("mean estimate", self.truth()["simulated_quantile"], est),
                Check("mean standard error", sd, mean_se, sd / math.sqrt(2.0 * n_draws)
                      + float(se.std(ddof=1)) / math.sqrt(n_draws))]


@lru_cache(maxsize=16)
def _recursion_pmf(nu: tuple, pd: tuple, sigma2: float, grid: int) -> np.ndarray:
    n = len(nu)
    inp = CreditRiskPlusInput(nu=np.array(nu), pd=np.array(pd), w0=np.zeros(n), weights=np.ones((n, 1)),
                              sigma2=np.array([sigma2]), loss_unit=1.0, currency="EUR")
    out = pmf_stable(inp, grid)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class CreditRiskPlusSample(GeneratorBase):
    """``scenarios`` Monte Carlo losses (in units) of ``obligors[j]`` obligors of exposure ``j + 1`` units and default
    probability ``pd``, all in one sector of variance ``mc_sigma2`` (default ``sigma2``), next to the recursion's pmf
    at ``sigma2`` on ``grid + 1`` points."""

    name: ClassVar[str] = "creditrisk_plus_sample"
    scenarios: int
    obligors: tuple[int, ...] = (200, 100, 60, 30, 10)
    pd: float = 0.02
    sigma2: float = 0.5
    mc_sigma2: float | None = None
    grid: int = 400

    def __post_init__(self) -> None:
        if int(self.scenarios) != self.scenarios or int(self.scenarios) < 10:
            raise ValueError("scenarios must be an integer of at least 10")
        object.__setattr__(self, "scenarios", int(self.scenarios))
        object.__setattr__(self, "obligors", tuple(int(n) for n in self.obligors))
        if not self.obligors or any(n < 0 for n in self.obligors) or sum(self.obligors) < 1:
            raise ValueError("obligors needs non-negative counts, at least one obligor")
        if not 0.0 < float(self.pd) < 1.0:
            raise ValueError("pd must lie strictly between 0 and 1")
        if not float(self.sigma2) > 0.0 or (self.mc_sigma2 is not None and not float(self.mc_sigma2) > 0.0):
            raise ValueError("sigma2 and mc_sigma2 must be positive")
        if int(self.grid) != self.grid or int(self.grid) < 10:
            raise ValueError("grid must be an integer of at least 10")

    @property
    def simulated_sigma2(self) -> float:
        return float(self.sigma2 if self.mc_sigma2 is None else self.mc_sigma2)

    @property
    def band_rates(self) -> np.ndarray:
        """The expected defaults of each exposure band: obligors times pd."""
        return np.asarray(self.obligors, dtype=float) * self.pd

    @property
    def pmf(self) -> np.ndarray:
        nu = tuple(j + 1 for j, n in enumerate(self.obligors) for _ in range(n))
        return _recursion_pmf(nu, (float(self.pd),) * len(nu), float(self.sigma2), int(self.grid))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """The Gamma sector factor (mean 1, variance ``mc_sigma2``), then a Poisson count per exposure band."""
        s2 = self.simulated_sigma2
        gamma = rng.gamma(1.0 / s2, s2, size=self.scenarios)
        counts = rng.poisson(self.band_rates[None, :] * gamma[:, None])
        units = counts @ np.arange(1, len(self.obligors) + 1)
        return {"pmf": self.pmf, "mc_units": units.astype(np.int64)}

    def truth(self) -> dict[str, Any]:
        mean = float(self.band_rates @ np.arange(1, len(self.obligors) + 1))
        second = float(self.band_rates @ (np.arange(1, len(self.obligors) + 1) ** 2))
        s2 = self.simulated_sigma2
        return {"mean_units": mean, "variance_units": second + s2 * mean * mean,
                "recursion_variance_units": second + self.sigma2 * mean * mean,
                "null_holds": s2 == self.sigma2}

    def verify(self, n_draws: int = 200, seed: int = 20261021) -> list[Check]:
        """The sample moments against the closed form, and the recursion's moments against the same closed form."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 432]))
        units = np.concatenate([self.draw(rng)["mc_units"] for _ in range(n_draws)])
        tr = self.truth()
        k = np.arange(self.grid + 1)
        f = self.pmf
        pm = float(f @ k)
        pv = float(f @ k**2 - pm * pm)
        return [mean_check("mean loss units of the sample", tr["mean_units"], units),
                variance_check("variance of the loss units of the sample", tr["variance_units"], units),
                Check("mean of the recursion's pmf", tr["mean_units"], pm, 1e-9 + 1e-6 * tr["mean_units"]),
                Check("variance of the recursion's pmf", tr["recursion_variance_units"], pv,
                      1e-9 + 1e-6 * tr["recursion_variance_units"])]


__all__ = ["CreditRiskPlusSample", "McQuantile", "lhp_quantile_closed_form", "sample_quantile_with_se"]
