"""Exceedance generators of the back-tests: a standardised series whose exceedance law is exact.

Three mechanisms, each a null at the boundary of its test and one stated departure from it (U16-E2, BL-088):

- ``PfeSeries``: a realised mark-to-market against a PFE forecast at quantile ``q``; the forecast is the true quantile
  times ``forecast_factor`` (1.0 is the null, below 1 a PFE understated by ``1 - forecast_factor``).
- ``EsResiduals``: normal losses with their true VaR and ES forecasts and a scale ``sigma_t``; the ES forecast is the
  true one times ``es_factor``. Under the null the exceedance residuals ``(L - ES) / sigma`` have mean zero.
- ``LhpLossRates``: annual loss rates of a large homogeneous portfolio under the one-factor Gaussian model with the
  true correlation ``true_rho``, handed to the tail back-test next to the fitted ``pd`` and ``rho`` (the null is
  ``true_rho == rho``).

The scale of the dates varies (``sigma_t = exp(dispersion Z)``) so that the series are not identically distributed in
the raw units, as a book's exposure is not: the standardised series the tests read is identically distributed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import special, stats

from riskvalidation.generators._base import Check, GeneratorBase, mean_check, proportion_check


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, got {value}")
    return int(value)


def _open_unit(name: str, value: float) -> float:
    if not (isinstance(value, (int, float)) and 0.0 < float(value) < 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return float(value)


@dataclass(frozen=True)
class PfeSeries(GeneratorBase):
    """``n_dates`` back-test dates: ``realised = sigma_t Z`` against ``pfe_forecast = forecast_factor sigma_t z_q``."""

    name: ClassVar[str] = "pfe_series"
    n_dates: int
    quantile: float = 0.95
    forecast_factor: float = 1.0
    dispersion: float = 0.3

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_dates", _positive_int("n_dates", self.n_dates, 2))
        _open_unit("quantile", self.quantile)
        if not float(self.forecast_factor) > 0.0:
            raise ValueError("forecast_factor must be positive")
        if float(self.dispersion) < 0.0:
            raise ValueError("dispersion must not be negative")

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """The scales, then the standardised realised values."""
        sigma = np.exp(self.dispersion * rng.standard_normal(self.n_dates))
        realised = sigma * rng.standard_normal(self.n_dates)
        forecast = self.forecast_factor * sigma * float(special.ndtri(self.quantile))
        return {"realised": realised, "pfe_forecast": forecast, "quantile": float(self.quantile)}

    def truth(self) -> dict[str, Any]:
        p = float(special.ndtr(-self.forecast_factor * special.ndtri(self.quantile)))
        return {"exceedance_probability": p, "nominal_exceedance_probability": 1.0 - self.quantile,
                "null_holds": math.isclose(p, 1.0 - self.quantile, rel_tol=1e-12)}

    def verify(self, n_draws: int = 4_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 411]))
        hits = np.concatenate([(d["realised"] > d["pfe_forecast"]) for d in (self.draw(rng) for _ in range(n_draws))])
        return [proportion_check("exceedance frequency", self.truth()["exceedance_probability"], hits)]


@dataclass(frozen=True)
class EsResiduals(GeneratorBase):
    """``n_days`` of normal losses ``sigma_t Z`` with the true VaR at ``alpha`` and the ES forecast ``es_factor`` times
    the true ES (``es_factor`` below 1 understates the ES)."""

    name: ClassVar[str] = "es_residuals"
    n_days: int
    alpha: float = 0.05
    es_factor: float = 1.0
    dispersion: float = 0.3

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_days", _positive_int("n_days", self.n_days, 2))
        _open_unit("alpha", self.alpha)
        if not float(self.es_factor) > 0.0:
            raise ValueError("es_factor must be positive")
        if float(self.dispersion) < 0.0:
            raise ValueError("dispersion must not be negative")

    @property
    def var_z(self) -> float:
        return float(special.ndtri(1.0 - self.alpha))

    @property
    def es_z(self) -> float:
        return float(stats.norm.pdf(self.var_z) / self.alpha)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        sigma = np.exp(self.dispersion * rng.standard_normal(self.n_days))
        loss = sigma * rng.standard_normal(self.n_days)
        return {"loss": loss, "var": sigma * self.var_z, "es": self.es_factor * sigma * self.es_z, "sigma": sigma,
                "alpha": float(self.alpha)}

    def truth(self) -> dict[str, Any]:
        # E[Z - es_factor ES | Z > VaR] = (1 - es_factor) ES: zero at the null, positive when the ES is understated
        return {"mean_residual": (1.0 - self.es_factor) * self.es_z, "exceedance_probability": float(self.alpha),
                "null_holds": self.es_factor == 1.0, "var_z": self.var_z, "es_z": self.es_z}

    def verify(self, n_draws: int = 2_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 412]))
        res, hits = [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            exc = d["loss"] > d["var"]
            hits.append(exc)
            res.append((d["loss"][exc] - d["es"][exc]) / d["sigma"][exc])
        tr = self.truth()
        return [proportion_check("exceedance frequency", tr["exceedance_probability"], np.concatenate(hits)),
                mean_check("mean exceedance residual", tr["mean_residual"], np.concatenate(res))]


@dataclass(frozen=True)
class LhpLossRates(GeneratorBase):
    """``n_years`` annual loss rates of an infinitely granular portfolio of PD ``pd`` and true correlation
    ``true_rho``; the fitted ``(pd, rho)`` go to the tail back-test (``true_rho = None`` is the null, ``rho``)."""

    name: ClassVar[str] = "lhp_loss_rates"
    n_years: int
    pd: float = 0.02
    rho: float = 0.12
    true_rho: float | None = None
    levels: tuple[float, ...] = (0.95, 0.99, 0.999)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_years", _positive_int("n_years", self.n_years))
        _open_unit("pd", self.pd)
        _open_unit("rho", self.rho)
        if self.true_rho is not None:
            _open_unit("true_rho", self.true_rho)
        object.__setattr__(self, "levels", tuple(_open_unit("level", q) for q in self.levels))

    @property
    def true_correlation(self) -> float:
        return float(self.rho if self.true_rho is None else self.true_rho)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        r = self.true_correlation
        z = rng.standard_normal(self.n_years)
        rates = special.ndtr((special.ndtri(self.pd) + math.sqrt(r) * z) / math.sqrt(1.0 - r))
        return {"realised": rates, "pd": float(self.pd), "rho": float(self.rho)}

    def fitted_quantile(self, level: float) -> float:
        return float(special.ndtr((special.ndtri(self.pd) + math.sqrt(self.rho) * special.ndtri(level))
                                  / math.sqrt(1.0 - self.rho)))

    def exceedance_probability(self, level: float) -> float:
        """P(a year's rate exceeds the fitted ``level``-quantile) under the true correlation (exact)."""
        r = self.true_correlation
        x = self.fitted_quantile(level)
        threshold = (math.sqrt(1.0 - r) * special.ndtri(x) - special.ndtri(self.pd)) / math.sqrt(r)
        return float(special.ndtr(-threshold))

    def truth(self) -> dict[str, Any]:
        return {"exceedance_probability": {str(q): self.exceedance_probability(q) for q in self.levels},
                "nominal_exceedance_probability": {str(q): 1.0 - q for q in self.levels},
                "null_holds": self.true_correlation == self.rho}

    def verify(self, n_draws: int = 4_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 413]))
        rates = np.concatenate([self.draw(rng)["realised"] for _ in range(n_draws)])
        return [proportion_check(f"exceedance frequency at {q}", self.exceedance_probability(q),
                                 rates > self.fitted_quantile(q))
                for q in self.levels if self.exceedance_probability(q) * rates.size > 200]


__all__ = ["EsResiduals", "LhpLossRates", "PfeSeries"]
