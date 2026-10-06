"""The generator contract: a frozen configuration, one dataset per draw, an exact truth, a verifier.

A generator is the data-generating mechanism of a simulation study (Morris, White and Crowther 2019,
doi:10.1002/sim.8086): its parameters are known, so what a test or an estimator should find is known too. Each
generator here serialises its configuration (the configuration written next to the data), draws one dataset from a
NumPy ``Generator`` (so one seed gives one dataset), returns its truth computed exactly, and verifies itself: a large
sample drawn on its own stream reproduces the truth within Monte Carlo error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np

# Gauss-Hermite rule for expectations over a standard normal factor: E[f(Z)] = sum w_i f(x_i)
GH_NODES, GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(256)
GH_WEIGHTS = GH_WEIGHTS / math.sqrt(2.0 * math.pi)


def jsonable(value: Any) -> Any:
    """NumPy arrays, tuples and scalars as plain JSON values (non-finite floats become ``None``)."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def frozen_array(value: Any, *, dtype: type = float, ndim: int | None = None) -> tuple:
    """A nested tuple (hashable, so the dataclass stays frozen) from an array-like."""
    arr = np.asarray(value, dtype=dtype)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape {arr.shape}")
    return tuple(_to_tuple(arr.tolist()))


def _to_tuple(x: Any) -> Any:
    return tuple(_to_tuple(v) for v in x) if isinstance(x, list) else x


@dataclass(frozen=True)
class Check:
    """One verification: the truth, what a large draw observed, and the Monte Carlo standard error."""

    name: str
    expected: float
    observed: float
    se: float

    @property
    def z(self) -> float:
        if self.se > 0:
            return (self.observed - self.expected) / self.se
        return 0.0 if math.isclose(self.observed, self.expected, rel_tol=1e-12, abs_tol=1e-15) else math.inf

    def ok(self, z_max: float = 4.0) -> bool:
        return abs(self.z) <= z_max

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"name": self.name, "expected": self.expected, "observed": self.observed, "se": self.se,
                         "z": self.z, "ok": self.ok()})


class GeneratorBase:
    """Shared behaviour of the generators: the serialised configuration and the verification report."""

    name: ClassVar[str] = "generator"

    def config(self) -> dict[str, Any]:
        """The generator's name, the package version and every parameter, JSON-safe."""
        from riskvalidation import __version__

        params = {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]
        return {"generator": self.name, "riskvalidation": __version__, "parameters": params}

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def truth(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def verify(self, n_draws: int = 20_000, seed: int = 20261005) -> list[Check]:  # pragma: no cover - interface
        raise NotImplementedError


def bvn_equal_excess(c: float, rho: float) -> float:
    r"""$\Phi_2(c, c; \rho) - \Phi(c)^2 = \int_0^\rho e^{-c^2/(1+r)}/(2\pi\sqrt{1-r^2})\,dr$, the bivariate standard
    normal distribution function at (c, c) above its value under independence (Plackett's identity: the derivative of
    the distribution function in the correlation is the density)."""
    from scipy import integrate

    if not -1.0 < rho < 1.0:
        raise ValueError("correlation must lie in (-1, 1)")
    if rho == 0.0:
        return 0.0
    val, _ = integrate.quad(lambda r: math.exp(-c * c / (1.0 + r)) / (2.0 * math.pi * math.sqrt(1.0 - r * r)),
                            0.0, rho, epsabs=1e-16, epsrel=1e-12, limit=200)
    return float(val)


def bvn_equal(c: float, rho: float) -> float:
    r"""The bivariate standard normal distribution function $\Phi_2(c, c; \rho)$."""
    from scipy import stats

    return float(stats.norm.cdf(c)) ** 2 + bvn_equal_excess(c, rho)


def mean_check(name: str, expected: float, samples: np.ndarray) -> Check:
    """The sample mean of ``samples`` against ``expected``, with the standard error of a mean."""
    s = np.asarray(samples, dtype=float)
    return Check(name, float(expected), float(s.mean()), float(s.std(ddof=1) / math.sqrt(s.size)))


def variance_check(name: str, expected: float, samples: np.ndarray) -> Check:
    """The sample variance against ``expected``; its standard error from the fourth central moment."""
    s = np.asarray(samples, dtype=float)
    r = s.size
    c = s - s.mean()
    var = float(np.sum(c**2) / (r - 1))
    m4 = float(np.mean(c**4))
    se = math.sqrt(max(m4 - var**2 * (r - 3) / (r - 1), 0.0) / r)
    return Check(name, float(expected), var, se)


def proportion_check(name: str, expected: float, hits: np.ndarray) -> Check:
    """A share of hits against ``expected``, with the binomial standard error at the expected value."""
    h = np.asarray(hits, dtype=bool)
    p = float(expected)
    return Check(name, p, float(h.mean()), math.sqrt(max(p * (1.0 - p), 1e-300) / h.size))


__all__ = ["Check", "GeneratorBase", "GH_NODES", "GH_WEIGHTS", "bvn_equal", "bvn_equal_excess", "frozen_array",
           "jsonable", "mean_check", "proportion_check", "variance_check"]
