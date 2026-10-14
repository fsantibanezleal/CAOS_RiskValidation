"""The compound annual loss ``Z = sum_{i=1..N} X_i`` (dossier 05 F.2) by Monte Carlo, Panjer recursion and FFT, and the
quantities read from it (dossier 05 F.3; Shevchenko 2010, sections 5 and 6, arXiv 1008.1108).

Panjer recursion (Panjer 1981) with a discretised severity ``f_j = P(X = j Delta)`` and a frequency of the ``(a, b, 0)``
class, ``p_n = (a + b / n) p_(n-1)``::

    h_n = 1 / (1 - a f_0) sum_(j=1..n) (a + b j / n) f_j h_(n-j),    h_0 = sum_k f_0^k p_k

(Shevchenko Theorem 5.2, equation 36; Table 1 for ``a``, ``b`` and ``h_0``). Benchmark (Shevchenko Example 9.4 and
Table 2): Poisson(100) with LN(0, 2) has ``E[Z] = 738.9056``, ``Var[Z] = 298095.7987`` and skewness 40.3428; Panjer with
the central discretisation of step 1 gives ``H_5847 = 0.998999329`` and ``H_5849 = 0.999000217`` (Table 2 prints each
value against its own index), so the 99.9 percent quantile is 5849. Mass beyond a discretised grid is reported, never
renormalised or extrapolated. Contract: ``engine-oprisk-api`` 1.0.0, module aggregate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from riskvalidation.oprisk._checks import level_in_open_unit, positive, scalar, vec
from riskvalidation.oprisk.distributions import _freq_params, _ppf, _sev_params, severity_cdf, severity_moment

_ALIAS_TOL = 1e-12
_MAX_LOSSES_PER_STEP = 2_000_000  # bounds the memory of one Monte Carlo step, whatever chunk_years says
_METHODS = ("central", "forward", "backward")


# ----------------------------------------------------------------------------------------------------------------
# Panjer coefficients and h0


def panjer_coefficients(law: str, params: Sequence[float]) -> tuple[float, float]:
    """The ``(a, b)`` of Shevchenko Table 1: Poisson ``(0, lam)``; negative binomial ``(1 - q, (1 - q)(r - 1))``;
    binomial
    ``(-q / (1 - q), q (m + 1) / (1 - q))``."""
    p = _freq_params(law, params)
    if law == "poisson":
        return 0.0, p[0]
    if law == "negative-binomial":
        r, q = p
        return 1.0 - q, (1.0 - q) * (r - 1.0)
    m, q = p
    return -q / (1.0 - q), q * (m + 1.0) / (1.0 - q)


def panjer_h0(law: str, params: Sequence[float], f0: float) -> float:
    """``h_0 = P(Z = 0) = sum_k f_0^k p_k`` in closed form (Shevchenko Table 1); ``f0`` in [0, 1)."""
    p = _freq_params(law, params)
    f0 = scalar(f0, "f0")
    if not 0.0 <= f0 < 1.0:
        raise ValueError(f"f0 must lie in [0, 1), got {f0}")
    if law == "poisson":
        return math.exp(p[0] * (f0 - 1.0))
    if law == "negative-binomial":
        r, q = p
        return (1.0 + (1.0 - f0) * (1.0 - q) / q) ** (-r)
    m, q = p
    return (1.0 + q * (f0 - 1.0)) ** m


# ----------------------------------------------------------------------------------------------------------------
# discretisation


def discretise_severity(law: str, params: Sequence[float], step: float, n_points: int,
                        method: str = "central") -> np.ndarray:
    """The masses ``f_j`` for the loss ``j * step``, ``j = 0 .. n_points - 1``.

    ``central``: ``f_0 = F(step / 2)``, ``f_j = F((j + 1/2) step) - F((j - 1/2) step)`` (the benchmark's); ``forward``:
    ``f_j = F((j + 1) step) - F(j step)`` (an upper bound); ``backward``: ``f_0 = 0``, ``f_j = F(j step) - F((j - 1)
    step)`` (a lower bound). The sum is at most 1; the missing mass is the tail beyond the grid and is never
    renormalised.
    """
    step = positive(step, "step")
    if int(n_points) != n_points or n_points < 2:
        raise ValueError(f"n_points must be an integer of at least 2, got {n_points}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    n = int(n_points)
    j = np.arange(n + 1, dtype=float)
    if method == "central":
        edges = (j - 0.5) * step
        edges[0] = -1.0  # the cdf is zero below the origin
        cdf = severity_cdf(law, params, edges)
        f = np.diff(cdf)
    elif method == "forward":
        cdf = severity_cdf(law, params, j * step)
        f = np.diff(cdf)
    else:
        cdf = severity_cdf(law, params, j[:n] * step)
        f = np.concatenate([[0.0], np.diff(cdf)])
    return np.maximum(f[:n], 0.0)


# ----------------------------------------------------------------------------------------------------------------
# Panjer recursion


def panjer_recursion(law: str, params: Sequence[float], f: np.ndarray) -> np.ndarray:
    """The probabilities ``h_n = P(Z = n * step)``, ``n = 0 .. G - 1``, for a frequency law of the ``(a, b, 0)`` class
    and a discretised severity ``f`` (the step is the caller's). Raises when ``a f_0 = 1``, when ``f_0`` is outside [0,
    1)
    or when ``h_0`` underflows to zero (every ``h_n`` would be a silent zero)."""
    a, b = panjer_coefficients(law, params)
    fv = vec(f, "f")
    if np.any(fv < 0.0):
        raise ValueError("f must be non-negative")
    f0 = float(fv[0])
    h0 = panjer_h0(law, params, f0)
    denom = 1.0 - a * f0
    if denom == 0.0:
        raise ValueError("a * f_0 = 1: the recursion is undefined")
    if h0 == 0.0:
        raise ValueError("h_0 underflows to zero (the expected count is too large for the recursion); "
                         "use fft_aggregate")
    g = fv.size
    h = np.zeros(g)
    h[0] = h0
    jf = np.arange(g, dtype=float) * fv
    for n in range(1, g):
        rev = h[n - 1::-1]
        total = (b / n) * float(np.dot(jf[1:n + 1], rev))
        if a != 0.0:
            total += a * float(np.dot(fv[1:n + 1], rev))
        h[n] = total / denom
    return h


# ----------------------------------------------------------------------------------------------------------------
# FFT


def _pgf(law: str, p: tuple[float, ...], z: np.ndarray) -> np.ndarray:
    """The probability generating function of the frequency at ``z`` (complex)."""
    if law == "poisson":
        return np.exp(p[0] * (z - 1.0))
    if law == "negative-binomial":
        r, q = p
        return np.exp(r * (math.log(q) - np.log(1.0 - (1.0 - q) * z)))
    m, q = p
    return (1.0 + q * (z - 1.0)) ** int(m)


def fft_aggregate(law: str, params: Sequence[float], f: np.ndarray, padding_factor: int = 2,
                  tilt: float = 0.0) -> np.ndarray:
    """``h_hat = psi(f_hat)``: the probability generating function of the frequency applied to the transform of the
    zero-padded severity (Shevchenko section 6.2), inverted; the first ``G`` entries, real.

    The transform length is ``padding_factor * G`` rounded up to a power of two; zero padding guards against aliasing.
    ``tilt = theta >= 0`` multiplies the severity by ``exp(-theta j)`` before the transform and the result by
    ``exp(theta n)`` after. A value below ``-1e-12`` (before untilting) is aliasing or an unstable transform and raises;
    smaller negative roundoff is clipped to 0. Agrees with ``panjer_recursion`` to 1e-9 per entry on the benchmark.
    """
    p = _freq_params(law, params)
    fv = vec(f, "f")
    if np.any(fv < 0.0):
        raise ValueError("f must be non-negative")
    if int(padding_factor) != padding_factor or padding_factor < 1:
        raise ValueError(f"padding_factor must be an integer of at least 1, got {padding_factor}")
    tilt = scalar(tilt, "tilt")
    if tilt < 0.0:
        raise ValueError(f"tilt must not be negative, got {tilt}")
    g = fv.size
    m = 1 << max(1, math.ceil(math.log2(max(2, int(padding_factor) * g))))
    j = np.arange(g, dtype=float)
    padded = np.zeros(m)
    padded[:g] = fv * np.exp(-tilt * j)
    with np.errstate(over="ignore", invalid="ignore"):
        h = np.fft.ifft(_pgf(law, p, np.fft.fft(padded)))[:g].real
    if not np.all(np.isfinite(h)) or float(h.min()) < -_ALIAS_TOL:
        raise ValueError("negative probabilities beyond roundoff: aliasing or an unstable transform; "
                         "increase padding_factor or tilt")
    h = np.maximum(h, 0.0)
    return h * np.exp(tilt * j) if tilt else h


# ----------------------------------------------------------------------------------------------------------------
# Monte Carlo


@dataclass(frozen=True)
class AnnualLossSample:
    """``losses`` ``(n_years,)``: the simulated annual loss of each year (nonnegative); ``seed`` it was drawn from."""

    losses: np.ndarray
    n_years: int
    seed: int


def _expected_count(law: str, p: tuple[float, ...]) -> float:
    if law == "poisson":
        return p[0]
    if law == "negative-binomial":
        return p[0] * (1.0 - p[1]) / p[1]
    return p[0] * p[1]


def simulate_annual_losses(freq_law: str, freq_params: Sequence[float], sev_law: str, sev_params: Sequence[float],
                           n_years: int, seed: int, chunk_years: int = 1_000_000) -> AnnualLossSample:
    """The annual loss of ``n_years`` independent years: the annual count is drawn, then that many severities, summed.

    Deterministic for ``(n_years, seed)`` and independent of ``chunk_years``: the counts of all years come from one
    ``PCG64`` stream and the severities (by the inverse cdf of uniforms) from a second stream consumed in year order,
    both spawned from ``SeedSequence(seed)``. ``chunk_years`` bounds the memory of one block and is not part of the
    result (a block is also capped at two million losses).
    """
    fp = _freq_params(freq_law, freq_params)
    sp = _sev_params(sev_law, sev_params)
    if int(n_years) != n_years or n_years < 1:
        raise ValueError(f"n_years must be a positive integer, got {n_years}")
    if int(chunk_years) != chunk_years or chunk_years < 1:
        raise ValueError(f"chunk_years must be a positive integer, got {chunk_years}")
    if int(seed) != seed or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    n_years, chunk_years, seed = int(n_years), int(chunk_years), int(seed)
    count_ss, sev_ss = np.random.SeedSequence(seed).spawn(2)
    count_rng, sev_rng = np.random.Generator(np.random.PCG64(count_ss)), np.random.Generator(np.random.PCG64(sev_ss))
    if freq_law == "poisson":
        counts = count_rng.poisson(fp[0], size=n_years)
    elif freq_law == "negative-binomial":
        counts = count_rng.negative_binomial(fp[0], fp[1], size=n_years)
    else:
        counts = count_rng.binomial(int(fp[0]), fp[1], size=n_years)
    step = min(chunk_years, max(1, _MAX_LOSSES_PER_STEP // max(1, math.ceil(_expected_count(freq_law, fp)))))
    losses = np.zeros(n_years)
    for start in range(0, n_years, step):
        stop = min(n_years, start + step)
        c = counts[start:stop]
        total = int(c.sum())
        if total == 0:
            continue
        sev = _ppf(sev_law, sp, sev_rng.random(total))
        year = np.repeat(np.arange(stop - start), c)
        losses[start:stop] = np.bincount(year, weights=sev, minlength=stop - start)
    return AnnualLossSample(losses=losses, n_years=n_years, seed=seed)


def mc_quantile(losses: np.ndarray, level: float) -> tuple[float, float]:
    """``(quantile, standard_error)`` of a Monte Carlo sample.

    The quantile is the ``ceil(n * level)``-th order statistic (1-based; ``n * level`` is taken exactly on the decimal
    ``level`` so float fuzz never moves it). The standard error is half the distance between the order statistics at
    ``ceil(n level + s)`` and ``floor(n level - s)`` with ``s = sqrt(n level (1 - level))`` (indices clipped to [1, n]):
    a one-standard-deviation binomial interval (dossier 05 F.3: the error decays as ``O(K^(-1/2))``).
    """
    x = vec(losses, "losses")
    level = level_in_open_unit(level)
    n = int(x.size)
    k = math.ceil(Fraction(n) * Fraction(repr(float(level))))
    centre = n * level
    s = math.sqrt(n * level * (1.0 - level))
    hi = min(max(math.ceil(centre + s), 1), n)
    lo = min(max(math.floor(centre - s), 1), n)
    k = min(max(k, 1), n)
    idx = sorted({k - 1, hi - 1, lo - 1})
    part = np.partition(x, idx)
    return float(part[k - 1]), 0.5 * float(part[hi - 1] - part[lo - 1])


# ----------------------------------------------------------------------------------------------------------------
# what is read from the pmf


def _pmf(h: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    hv = vec(h, "h")
    if np.any(hv < 0.0):
        raise ValueError("h must be non-negative")
    return hv, positive(step, "step")


def quantile_from_pmf(h: np.ndarray, step: float, level: float) -> float:
    """``n * step`` for the smallest zero-based ``n`` whose cumulative sum is at least ``level`` (the VaR at that
    level).
    Raises when the cumulative sum never reaches ``level``: mass beyond the grid is never extrapolated."""
    hv, step = _pmf(h, step)
    level = level_in_open_unit(level)
    cs = np.cumsum(hv)
    idx = int(np.searchsorted(cs, level, side="left"))
    if idx >= cs.size:
        raise ValueError(f"the cumulative probability on the grid is {cs[-1]:.12g}, below the level {level}: "
                         "mass lies beyond the grid")
    return float(idx * step)


def es_from_pmf(h: np.ndarray, step: float, level: float, max_missing_share: float = 1e-3,
                infinite_mean: bool = False) -> float:
    """The expected shortfall ``ES = (E[Z; Z > q] + q (P(Z <= q) - level)) / (1 - level)`` with ``q =
    quantile_from_pmf(h, step, level)``; equivalently ``q + E[(Z - q)^+] / (1 - level)``, so ``ES >= VaR``, with
    equality for a point mass.

    The grid must carry the tail: ``ValueError`` when the mass beyond the grid, ``1 - sum(h)``, exceeds
    ``max_missing_share * (1 - level)`` (default ``1e-3 * (1 - level)``), because an expected shortfall read off a grid
    that ends before the tail is too small (on the benchmark's grid of 6000 points it would be 377.8 against a VaR of
    5849). **The pmf alone cannot reveal the tail index**: the expected shortfall does not exist for a severity with a
    tail index of 1 or more (the compound loss has an infinite mean), yet on a coarse enough grid (a large step) the
    missing mass of such a severity falls below the tolerance and a finite, meaningless number comes out. The caller
    therefore decides from the severity, before calling, whether the expected shortfall exists: pass
    ``infinite_mean=True`` (or use ``es_for_severity``, which asks ``severity_moment``) and the function raises
    ``ValueError("infinite mean")`` instead of computing. ``ValueError`` as ``quantile_from_pmf`` otherwise.
    """
    if infinite_mean:
        raise ValueError("infinite mean: the severity has a tail index of 1 or more, so the expected shortfall "
                         "does not exist and is never computed")
    hv, step = _pmf(h, step)
    level = level_in_open_unit(level)
    share = scalar(max_missing_share, "max_missing_share")
    if share <= 0.0:
        raise ValueError(f"max_missing_share must be positive, got {share}")
    cs = np.cumsum(hv)
    missing = 1.0 - float(cs[-1])
    if missing > share * (1.0 - level):
        raise ValueError(f"the mass beyond the grid, {missing:.6g}, exceeds {share} of the tail {1.0 - level:.6g}: "
                         "the grid "
                         "does not carry the tail, so the expected shortfall is not computed (it does not exist for a "
                         "tail index of 1 or more)")
    q = quantile_from_pmf(hv, step, level)
    idx = int(round(q / step))
    x = np.arange(hv.size, dtype=float) * step
    above = float(np.dot(x[idx + 1:], hv[idx + 1:]))
    return (above + q * (float(cs[idx]) - level)) / (1.0 - level)


def es_for_severity(h: np.ndarray, step: float, level: float, sev_law: str, sev_params: Sequence[float],
                    max_missing_share: float = 1e-3) -> float:
    """``es_from_pmf`` gated on the severity: ``ValueError("infinite mean")`` before anything is computed when
    ``severity_moment(sev_law, sev_params, 1)`` is infinite (a GPD or splice tail index of 1 or more), whatever the
    grid.
    The C17 bake writes the expected shortfall as null with the reason ``infinite mean`` on that error."""
    if math.isinf(severity_moment(sev_law, sev_params, 1)):
        raise ValueError("infinite mean: the severity has a tail index of 1 or more, so the expected shortfall "
                         "does not exist and is never computed")
    return es_from_pmf(h, step, level, max_missing_share)


def exceedance_from_pmf(h: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """``(x, sf)``: ``x_n = n * step`` and ``sf_n = P(Z > x_n) = 1 - cumsum(h)_n`` clipped at 0, nonincreasing in
    [0, 1]."""
    hv, step = _pmf(h, step)
    sf = np.minimum.accumulate(np.clip(1.0 - np.cumsum(hv), 0.0, 1.0))
    return np.arange(hv.size, dtype=float) * step, sf


# ----------------------------------------------------------------------------------------------------------------
# compound moments


@dataclass(frozen=True)
class CompoundMoments:
    """``mean`` EUR per year, ``variance`` EUR squared, ``skewness`` dimensionless; ``mean`` is ``inf`` when the
    severity
    mean is, ``variance`` when its second moment is, and ``skewness`` is ``nan`` when the variance is infinite."""

    mean: float
    variance: float
    skewness: float


def compound_poisson_moments(lam: float, sev_law: str, sev_params: Sequence[float]) -> CompoundMoments:
    """``E[Z] = lam E[X]``, ``Var[Z] = lam E[X^2]``, skewness ``lam E[X^3] / Var[Z]^(3/2)`` (Shevchenko Example 9.4)."""
    lam = positive(lam, "lam")
    m1, m2, m3 = (severity_moment(sev_law, sev_params, k) for k in (1, 2, 3))
    mean = lam * m1
    variance = lam * m2
    if math.isinf(variance):
        skew = math.nan
    elif math.isinf(m3):
        skew = math.inf
    else:
        skew = lam * m3 / variance ** 1.5
    return CompoundMoments(mean=mean, variance=variance, skewness=skew)
