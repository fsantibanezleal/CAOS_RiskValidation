"""The FRTB statistics of dossier 01 D.4 (MAR32.4 to MAR32.9, MAR32.18 to MAR32.19, MAR32.36 to MAR32.42).

The engine computes the statistics and the zones of the text; it states no compliance claim. The US multiplication
factors of 12 CFR 217.204 Table 1 (dossier 05 B.4) are reference values for a document, not a function of the engine.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import stats

PLA_N = 250
# the zone design (BCBS 1996, MAR99.17 notes): amber where the cumulative probability reaches 95 percent and red
# where it reaches 99.99 percent
AMBER_LEVEL = 0.95
RED_LEVEL = 0.9999
# MAR32.9 Table 1 (dossier 01 D.4): the backtesting-dependent multiplier m_c per number of exceptions in 250 days
_MULTIPLIER = (1.50, 1.50, 1.50, 1.50, 1.50, 1.70, 1.76, 1.83, 1.88, 1.92)
_RED_MULTIPLIER = 2.00
_DESK_LIMITS = {0.01: 12, 0.025: 30}   # MAR32.18 to MAR32.19: exceptions in the last 12 months at 99 and 97.5 percent


def binomial_cumulative_probability(k: int, n: int, p: float) -> float:
    """`P(X <= k)`, `X ~ Bin(n, p)`: `sum_{i=0}^{k} C(n, i) p^i (1-p)^(n-i)` (dossier 01 D.4)."""
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1], got {p}")
    return float(stats.binom.cdf(int(k), int(n), float(p)))


@lru_cache(maxsize=64)
def _zone_starts(n: int, p: float) -> tuple[int, int]:
    cdf = stats.binom.cdf(np.arange(n + 1), n, p)
    amber = int(np.argmax(cdf >= AMBER_LEVEL))
    red = int(np.argmax(cdf >= RED_LEVEL))
    return amber, red


def traffic_light_zone(exceptions: int, n: int = 250, coverage: float = 0.99) -> str:
    """Green below the count where the cumulative probability at true coverage first reaches 95 percent, amber from
    there, red from where it reaches 99.99 percent (n = 250 and 99 percent: green 0 to 4, amber 5 to 9, red 10 or
    more)."""
    if int(exceptions) != exceptions or exceptions < 0:
        raise ValueError(f"exceptions must be a non-negative integer, got {exceptions}")
    if int(n) != n or n < 1 or not (0.0 < coverage < 1.0):
        raise ValueError("n must be a positive integer and coverage must lie in (0, 1)")
    amber, red = _zone_starts(int(n), round(1.0 - float(coverage), 12))
    return "green" if exceptions < amber else "amber" if exceptions < red else "red"


def mar32_multiplier(exceptions: int) -> float:
    """The backtesting-dependent multiplier `m_c` of MAR32.9 Table 1: 1.50 for 0 to 4, 1.70, 1.76, 1.83, 1.88, 1.92
    for 5 to 9 and 2.00 for 10 or more."""
    if int(exceptions) != exceptions or exceptions < 0:
        raise ValueError(f"exceptions must be a non-negative integer, got {exceptions}")
    return _MULTIPLIER[int(exceptions)] if exceptions < len(_MULTIPLIER) else _RED_MULTIPLIER


def desk_exception_limit(alpha: float) -> int:
    """12 at 0.01 (99 percent VaR) and 30 at 0.025 (97.5 percent VaR): the most exceptions tolerated in 12 months."""
    for key, limit in _DESK_LIMITS.items():
        if abs(float(alpha) - key) < 1e-12:
            return limit
    raise ValueError(f"alpha must be 0.01 or 0.025, got {alpha}")


def desk_exceptions(loss: np.ndarray, var: np.ndarray, missing_as_exception: bool = True) -> int:
    """The count of days with `loss > var` (positive loss numbers); with `missing_as_exception` a NaN in either input
    counts as an exception (MAR32.4 to MAR32.9 for the bank-wide test; for a desk the dossier does not say, hence the
    flag)."""
    lo, va = np.asarray(loss, dtype=float), np.asarray(var, dtype=float)
    if lo.ndim != 1 or lo.shape != va.shape:
        raise ValueError(f"loss and var must be one-dimensional of the same length, got {lo.shape} and {va.shape}")
    missing = np.isnan(lo) | np.isnan(va)
    count = int(np.sum((lo > va) & ~missing))
    return count + (int(missing.sum()) if missing_as_exception else 0)


def _pla_series(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size != PLA_N:
        raise ValueError(f"{name} needs exactly {PLA_N} observations (MAR32.20), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def pla_spearman(hpl: np.ndarray, rtpl: np.ndarray) -> float:
    """The Pearson correlation of the average ranks, `cov(R_HPL, R_RTPL) / (sigma_R_HPL sigma_R_RTPL)` (MAR32.36 to
    MAR32.38).
    ValueError unless both series have exactly 250 finite observations and a nonzero spread."""
    h, r = _pla_series(hpl, "hpl"), _pla_series(rtpl, "rtpl")
    if np.ptp(h) == 0.0 or np.ptp(r) == 0.0:
        raise ValueError("a PLA series has zero spread: the rank correlation is undefined")
    rh, rr = stats.rankdata(h), stats.rankdata(r)
    rh, rr = rh - rh.mean(), rr - rr.mean()
    return float(np.clip((rh @ rr) / np.sqrt((rh @ rh) * (rr @ rr)), -1.0, 1.0))


def _pla_gap(h: np.ndarray, r: np.ndarray) -> int:
    """`max_x |#{rtpl <= x} - #{hpl <= x}|`, the exact integer count of ECDF steps."""
    sh, sr = np.sort(h), np.sort(r)
    pts = np.union1d(sh, sr)
    return int(np.max(np.abs(np.searchsorted(sr, pts, side="right") - np.searchsorted(sh, pts, side="right"))))


def pla_ks(hpl: np.ndarray, rtpl: np.ndarray) -> float:
    """`KS = sup_x |F_RTPL(x) - F_HPL(x)|` with `F(x) = 0.004 #{obs <= x}` (MAR32.39 to MAR32.41), computed as an exact
    integer count of steps `gap = max_x |#{rtpl <= x} - #{hpl <= x}|` divided by 250 (review F5): 30 steps give
    exactly the float 0.12, where `0.004 * a - 0.004 * b` gives 0.12000000000000002."""
    return _pla_gap(_pla_series(hpl, "hpl"), _pla_series(rtpl, "rtpl")) / PLA_N


def pla_zone(spearman: float, ks: float) -> str:
    """MAR32.42: green when Spearman is above 0.80 and KS below 0.09; red when Spearman is below 0.70 or KS above 0.12;
    amber otherwise (the boundaries belong to amber)."""
    if spearman < 0.70 or ks > 0.12:
        return "red"
    if spearman > 0.80 and ks < 0.09:
        return "green"
    return "amber"


def compare_spearman(x: np.ndarray, y: np.ndarray, threshold: float) -> int:
    """The sign of `rho - threshold` for the Spearman correlation of `x` and `y`, decided in exact arithmetic (review
    m4).

    With the doubled average ranks (integers) centred by `n`, `rho = Sxy / sqrt(Sxx Syy)` with integer sums, and the
    comparison with the threshold is made on `Sxy^2` against `threshold^2 Sxx Syy` in rational arithmetic, so a value
    on a boundary (0.80, 0.70) is never moved by a float quotient, with or without ties. Returns 1, 0 or -1.
    `ValueError` for a series with no spread or series of different length."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("x and y must be one-dimensional of the same length (at least 2)")
    n = a.size

    def centred(v: np.ndarray) -> list[int]:
        r2 = [int(round(2 * t)) for t in stats.rankdata(v)]
        tot = sum(r2)
        return [n * t - tot for t in r2]

    cx, cy = centred(a), centred(b)
    sxx, syy = sum(t * t for t in cx), sum(t * t for t in cy)
    if sxx == 0 or syy == 0:
        raise ValueError("a series has zero spread: the rank correlation is undefined")
    sxy = sum(p * q for p, q in zip(cx, cy, strict=True))
    t = Fraction(str(float(threshold)))
    lhs, rhs = Fraction(sxy * sxy), t * t * sxx * syy
    if t >= 0:
        if sxy < 0:
            return -1
        return (lhs > rhs) - (lhs < rhs)
    if sxy >= 0:
        return 1
    return (lhs < rhs) - (lhs > rhs)


def pla_zone_exact(hpl: np.ndarray, rtpl: np.ndarray) -> str:
    """The zone of MAR32.42 for two 250-day series, decided on integers: the Spearman boundaries 0.80 and 0.70 by
    `compare_spearman` (exact rational arithmetic) and the KS boundaries 0.09 and 0.12 on the step gap (`gap / 250 <
    0.09` is `gap <= 22`, `gap / 250 > 0.12` is `gap >= 31`). Agrees with `pla_zone(pla_spearman, pla_ks)` except
    where a float quotient would sit on a boundary."""
    h, r = _pla_series(hpl, "hpl"), _pla_series(rtpl, "rtpl")
    if np.ptp(h) == 0.0 or np.ptp(r) == 0.0:
        raise ValueError("a PLA series has zero spread: the rank correlation is undefined")
    gap = _pla_gap(h, r)
    if compare_spearman(h, r, 0.70) < 0 or gap * 100 > 12 * PLA_N:
        return "red"
    if compare_spearman(h, r, 0.80) > 0 and gap * 100 < 9 * PLA_N:
        return "green"
    return "amber"
