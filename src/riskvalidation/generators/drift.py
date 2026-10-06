"""Two samples of one variable, the second possibly shifted: the design of Yurdakul and Naranjo's PSI study.

"We generated n observations from the N(mu_1 = 0, sigma^2 = 64) and let (p_1, ..., p_10) be the observed proportions
that fall into the B = 10 population deciles. A second sample of m observations was generated from N(mu_2, sigma^2 = 64)
and we let (q_1, ..., q_10) be the observed proportions falling into the deciles of N(0, sigma^2 = 64)" (Yurdakul and
Naranjo 2020, doi:10.21314/JRMV.2020.227, section 4). With ``binning="sample"`` the bins are instead the base
sample's own quantiles, as `riskvalidation.validation.stability.quantile_edges` builds them in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import special

from riskvalidation.generators._base import Check, GeneratorBase, mean_check
from riskvalidation.validation.stability import bin_counts, quantile_edges

YURDAKUL_NARANJO = (
    "Yurdakul and Naranjo (2020), Statistical properties of the population stability index, Journal of Risk Model "
    "Validation 14(4), doi:10.21314/JRMV.2020.227, section 4 and Table 4"
)


@dataclass(frozen=True)
class TwoSample(GeneratorBase):
    """``n_base`` observations of N(0, sd^2) and ``n_target`` of N(shift sd, (scale sd)^2), binned in ``bins`` bins."""

    name: ClassVar[str] = "two_sample"
    n_base: int
    n_target: int
    shift: float = 0.0
    scale: float = 1.0
    sd: float = 8.0
    bins: int = 10
    binning: str = "population"

    def __post_init__(self) -> None:
        if int(self.n_base) < 2 or int(self.n_target) < 2:
            raise ValueError("both samples need at least two observations")
        if float(self.scale) <= 0.0 or float(self.sd) <= 0.0:
            raise ValueError("scale and sd must be positive")
        if int(self.bins) < 2:
            raise ValueError("at least two bins are required")
        if self.binning not in ("population", "sample"):
            raise ValueError("binning is 'population' or 'sample'")
        object.__setattr__(self, "n_base", int(self.n_base))
        object.__setattr__(self, "n_target", int(self.n_target))
        object.__setattr__(self, "bins", int(self.bins))

    @property
    def population_edges(self) -> np.ndarray:
        inner = self.sd * special.ndtri(np.arange(1, self.bins) / self.bins)
        return np.concatenate([[-np.inf], inner, [np.inf]])

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """The base sample, then the target sample, then the counts on the chosen bins."""
        base = self.sd * rng.standard_normal(self.n_base)
        target = self.shift * self.sd + self.scale * self.sd * rng.standard_normal(self.n_target)
        edges = self.population_edges if self.binning == "population" else quantile_edges(base, self.bins)
        return {"development": base, "current": target, "expected_counts": bin_counts(base, edges),
                "actual_counts": bin_counts(target, edges), "edges": edges}

    def truth(self) -> dict[str, Any]:
        edges = self.population_edges
        p = np.diff(special.ndtr(edges / self.sd))
        q = np.diff(special.ndtr((edges - self.shift * self.sd) / (self.scale * self.sd)))
        return {
            "bins_base": p.tolist(),
            "bins_target": q.tolist(),
            "population_psi": float(np.sum((q - p) * np.log(q / p))),
            "wasserstein_1": None if self.scale != 1.0 else abs(self.shift) * self.sd,
            "null_holds": self.shift == 0.0 and self.scale == 1.0,
            "binning": self.binning,
            "reference": YURDAKUL_NARANJO,
        }

    def verify(self, n_draws: int = 4_000, seed: int = 20261005) -> list[Check]:
        """The share of each sample in every population bin against the exact bin probabilities."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 305]))
        tr = self.truth()
        edges = self.population_edges
        shares_b, shares_t = [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            shares_b.append(bin_counts(d["development"], edges) / self.n_base)
            shares_t.append(bin_counts(d["current"], edges) / self.n_target)
        sb, st = np.array(shares_b), np.array(shares_t)
        out = []
        for i in range(self.bins):
            out.append(mean_check(f"base share, bin {i + 1}", tr["bins_base"][i], sb[:, i]))
            out.append(mean_check(f"target share, bin {i + 1}", tr["bins_target"][i], st[:, i]))
        return out


def population_psi(p: np.ndarray, q: np.ndarray) -> float:
    """PSI between two populations' bin probabilities (no sampling): sum (q - p) ln(q / p)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if np.any(p <= 0) or np.any(q <= 0):
        return math.inf
    return float(np.sum((q - p) * np.log(q / p)))


__all__ = ["TwoSample", "YURDAKUL_NARANJO", "population_psi"]
