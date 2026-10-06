r"""Default counts of rating grades over years in the one-factor model of BCBS Working Paper 14's simulation study.

The model (WP14, pages 61 to 62): year $t$ has a systematic factor $S_t$; the factors are jointly standard normal with
$\mathrm{corr}(S_s, S_t) = \vartheta^{|s-t|}$; given the factors the default counts are independent binomials with
the conditional PD $\Phi((\Phi^{-1}(p_t) - \sqrt{\rho_t}\,S_t)/\sqrt{1-\rho_t})$, so a small $S_t$ is a bad year. The
PDs a test is given are the forecasts; when they equal the true PDs every calibration null holds at its boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from scipy import special, stats

from riskvalidation.generators._base import (
    Check,
    GeneratorBase,
    bvn_equal_excess,
    frozen_array,
    mean_check,
    variance_check,
)

WP14 = (
    "BCBS (2005), Studies on the Validation of Internal Rating Systems (revised), Working Paper 14, simulation study "
    "on the performance of the normal and traffic lights tests (pages 61 to 67)"
)


def joint_default_probability(pd: float, rho: float) -> float:
    r"""P(two obligors with PD ``pd`` and asset correlation ``rho`` both default), the bivariate normal
    $\Phi_2(c, c; \rho)$ at $c = \Phi^{-1}(pd)$, computed as
    $\Phi(c)^2 + \int_0^\rho e^{-c^2/(1+r)}/(2\pi\sqrt{1-r^2})\,dr$ (the derivative of the bivariate normal
    distribution function in its correlation is its density)."""
    if not 0.0 < pd < 1.0:
        raise ValueError("pd must lie in (0, 1)")
    if not 0.0 <= rho < 1.0:
        raise ValueError("asset correlation must lie in [0, 1)")
    return pd * pd + bvn_equal_excess(float(stats.norm.ppf(pd)), rho)


def default_correlation(pd: float, rho: float) -> float:
    r"""Correlation of two default indicators in the one-factor model: $(\Phi_2(c, c; \rho) - pd^2)/(pd(1-pd))$.

    WP14 tabulates it next to the asset correlation in its binomial-test examples (pages 58 to 59), for example 0.41%
    at PD 1% and asset correlation 5%."""
    return (joint_default_probability(pd, rho) - pd * pd) / (pd * (1.0 - pd))


@dataclass(frozen=True)
class DefaultCounts(GeneratorBase):
    """``obligors[t][g]`` obligors of grade ``g`` in year ``t``, true PDs ``pd_true[t][g]``, the PDs given to the test
    ``pd_forecast[t][g]`` (the true PDs when omitted), asset correlation ``rho[t]`` and factor correlation ``theta``.

    Array arguments broadcast to (years, grades); a one-dimensional argument is one year of grades. Use
    :meth:`over_years` for one grade observed over years.
    """

    name: ClassVar[str] = "default_counts"
    obligors: Any
    pd_true: Any
    pd_forecast: Any = None
    rho: Any = 0.0
    theta: float = 0.0

    def __post_init__(self) -> None:
        n = np.atleast_2d(np.asarray(self.obligors))
        pt = np.atleast_2d(np.asarray(self.pd_true, dtype=float))
        pf = pt if self.pd_forecast is None else np.atleast_2d(np.asarray(self.pd_forecast, dtype=float))
        shape = np.broadcast_shapes(n.shape, pt.shape, pf.shape)
        if len(shape) != 2:
            raise ValueError("obligors and PDs must broadcast to (years, grades)")
        if not np.all(np.equal(np.mod(n, 1), 0)) or np.any(n < 1):
            raise ValueError("obligors must be positive integers")
        for arr, label in ((pt, "pd_true"), (pf, "pd_forecast")):
            if np.any(arr <= 0.0) or np.any(arr >= 1.0):
                raise ValueError(f"{label} must lie in (0, 1)")
        rho = np.asarray(self.rho, dtype=float).reshape(-1)
        rho = np.broadcast_to(rho, (shape[0],)) if rho.size in (1, shape[0]) else None
        if rho is None:
            raise ValueError("rho needs one value, or one per year")
        if np.any(rho < 0.0) or np.any(rho >= 1.0):
            raise ValueError("asset correlation must lie in [0, 1)")
        if not -1.0 < float(self.theta) < 1.0:
            raise ValueError("factor correlation theta must lie in (-1, 1)")
        object.__setattr__(self, "obligors", frozen_array(np.broadcast_to(n, shape), dtype=np.int64))
        object.__setattr__(self, "pd_true", frozen_array(np.broadcast_to(pt, shape)))
        object.__setattr__(self, "pd_forecast", frozen_array(np.broadcast_to(pf, shape)))
        object.__setattr__(self, "rho", frozen_array(rho))
        object.__setattr__(self, "theta", float(self.theta))

    @classmethod
    def over_years(cls, obligors: Any, pd_true: Any, pd_forecast: Any = None, rho: Any = 0.0,
                   theta: float = 0.0) -> DefaultCounts:
        """One grade observed over years; each argument holds one value per year, or one for every year."""
        args = [np.asarray(a, dtype=float).reshape(-1) for a in (obligors, pd_true) +
                (() if pd_forecast is None else (pd_forecast,))]
        years = max(a.size for a in args)
        col = lambda a: np.broadcast_to(np.asarray(a).reshape(-1), (years,)).reshape(years, 1)  # noqa: E731
        return cls(obligors=col(np.asarray(obligors)), pd_true=col(pd_true),
                   pd_forecast=None if pd_forecast is None else col(pd_forecast), rho=rho, theta=theta)

    # arrays, built once per instance
    @cached_property
    def _n(self) -> np.ndarray:
        return np.asarray(self.obligors, dtype=np.int64)

    @cached_property
    def _pt(self) -> np.ndarray:
        return np.asarray(self.pd_true, dtype=float)

    @cached_property
    def _pf(self) -> np.ndarray:
        return np.asarray(self.pd_forecast, dtype=float)

    @cached_property
    def _rho(self) -> np.ndarray:
        return np.asarray(self.rho, dtype=float)

    @cached_property
    def factor_correlation(self) -> np.ndarray:
        """corr(S_s, S_t) = theta^|s - t|."""
        t = np.arange(self._n.shape[0])
        return float(self.theta) ** np.abs(t[:, None] - t[None, :])

    @cached_property
    def _chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.factor_correlation)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._n.shape)  # type: ignore[return-value]

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """One panel of counts: the factors first (one standard normal per year), then the binomial counts."""
        years = self._n.shape[0]
        z = rng.standard_normal(years)
        s = z if self.theta == 0.0 else self._chol @ z
        rho = self._rho
        if np.all(rho == 0.0):
            cond = self._pt
        else:
            r = rho[:, None]
            shifted = (special.ndtri(self._pt) - np.sqrt(r) * s[:, None]) / np.sqrt(1.0 - r)
            cond = np.where(r > 0.0, special.ndtr(shifted), self._pt)
        d = rng.binomial(self._n, cond)
        return {"n": self._n, "d": d, "pd": self._pf, "pd_true": self._pt, "factor": s}

    def truth(self) -> dict[str, Any]:
        n, pt, rho = self._n, self._pt, self._rho
        joint = np.array([[joint_default_probability(float(pt[t, g]), float(rho[t])) for g in range(n.shape[1])]
                          for t in range(n.shape[0])])
        var = n * pt * (1.0 - pt) + n * (n - 1.0) * (joint - pt * pt)
        return {
            "obligors": n.tolist(),
            "pd_true": pt.tolist(),
            "pd_forecast": self._pf.tolist(),
            "rho": rho.tolist(),
            "theta": float(self.theta),
            "expected_defaults": (n * pt).tolist(),
            "variance_defaults": var.tolist(),
            "default_correlation": ((joint - pt * pt) / (pt * (1.0 - pt))).tolist(),
            "factor_correlation": self.factor_correlation.tolist(),
            "calibration_null_holds": bool(np.all(self._pf >= pt)),
            "reference": WP14,
        }

    def verify(self, n_draws: int = 20_000, seed: int = 20261005) -> list[Check]:
        """Means and variances of every count, and the factor correlation at lag one, on the generator's own stream."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 302]))
        draws = [self.draw(rng) for _ in range(n_draws)]
        d = np.stack([x["d"] for x in draws])
        s = np.stack([x["factor"] for x in draws])
        tr = self.truth()
        out: list[Check] = []
        years, grades = self.shape
        for t in range(years):
            for g in range(grades):
                out.append(mean_check(f"mean defaults, year {t + 1}, grade {g + 1}", tr["expected_defaults"][t][g],
                                      d[:, t, g]))
                out.append(variance_check(f"variance of defaults, year {t + 1}, grade {g + 1}",
                                          tr["variance_defaults"][t][g], d[:, t, g]))
        for t in range(years - 1):
            out.append(mean_check(f"factor correlation, years {t + 1} and {t + 2}", self.theta, s[:, t] * s[:, t + 1]))
        return out


def feed_cell(t: int = 0, g: int = 0):
    """Feed one grade of one year to a count test: ``n``, ``d``, ``pd`` as scalars."""

    def feed(draw: dict[str, Any]) -> dict[str, Any]:
        return {"n": int(draw["n"][t, g]), "d": int(draw["d"][t, g]), "pd": float(draw["pd"][t, g])}

    return feed


def feed_series(g: int = 0):
    """Feed one grade over the years: ``n``, ``d``, ``pd`` per year and the yearly ``default_rates``."""

    def feed(draw: dict[str, Any]) -> dict[str, Any]:
        n, d = draw["n"][:, g], draw["d"][:, g]
        return {"n": n, "d": d, "pd": draw["pd"][:, g], "default_rates": d / n}

    return feed


def feed_grades(t: int = 0):
    """Feed the grades of one year: ``n``, ``d``, ``pd`` per grade."""

    def feed(draw: dict[str, Any]) -> dict[str, Any]:
        return {"n": draw["n"][t], "d": draw["d"][t], "pd": draw["pd"][t]}

    return feed


def feed_portfolio(t: int = 0):
    """Feed the portfolio of one year: all obligors, all defaults and the obligor-weighted PD (the portfolio row of
    the ECB's Jeffreys test, section 2.5.3.1)."""

    def feed(draw: dict[str, Any]) -> dict[str, Any]:
        n, d, pd = draw["n"][t], draw["d"][t], draw["pd"][t]
        total = int(n.sum())
        return {"n": total, "d": int(d.sum()), "pd": float(np.sum(n * pd) / total)}

    return feed


__all__ = ["WP14", "DefaultCounts", "default_correlation", "feed_cell", "feed_grades", "feed_portfolio",
           "feed_series", "joint_default_probability"]
