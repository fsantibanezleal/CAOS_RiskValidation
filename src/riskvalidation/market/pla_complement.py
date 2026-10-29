"""A scale-and-level complement to the FRTB profit-and-loss attribution test (BS-13, beyond the regulatory state of
the art).

The regulatory PLA (`riskvalidation.market.frtb`, MAR32.36 to MAR32.42) reads two statistics on the 250-day pair of
hypothetical P&L (HPL) and risk-theoretical P&L (RTPL): the Spearman correlation of the levels and the
Kolmogorov-Smirnov distance of the two distributions. Both are blind to a model whose volatility input is stale and,
for Spearman, to a model whose correlation is wrong. This module adds three statistics on the same 250-day pair, each
aimed at what the PLA does not see, and a decision calibrated to a stated type I error. The complement is added to
the regulatory zone, never substituted for it: `pla_complement` returns the regulatory zone beside its own decision.
It has no regulatory standing.

Statistics (all on 250 finite daily P&L figures, in the same unit for HPL and RTPL, a decimal of the desk notional):

- the Mincer-Zarnowitz regression `HPL_t = a + b RTPL_t + e_t` and the joint Wald test of `(a, b) = (0, 1)` with the
  heteroskedasticity-consistent HC3 covariance (MacKinnon and White 1985), which sees a mis-scaled or shifted model; the
  P&L volatility is time-varying, so the classical covariance would overstate the evidence. Mincer and Zarnowitz (1969),
  "The evaluation of economic forecasts", in Mincer (ed.), Economic Forecasts and Expectations, NBER. The p-value is the
  asymptotic chi-square with 2 degrees of freedom and is a reference only: the decision uses a simulated null;
- the Spearman correlation of `|HPL|` and `|RTPL|`, which sees a risk model that misreads the size of the day's move;
- the Spearman correlation of the `window`-day rolling means of `|HPL|` and `|RTPL|` (default 20 days), which sees a
  volatility that lags: a model fed a 20-day-old volatility moves its scale late, the daily shock stays the same, and
  the rolling mean removes the shared shock and leaves the two scales.

Decision: the Wald statistic flags when it is above its critical value, the two Spearman statistics flag when they
are below theirs. The critical values come from `pla_complement_calibrate` on null pairs (a model with nothing
planted, simulated from the user's own generator), the exceedance count of the null being at most `size` times the
number of null pairs (so the stated type I error is not exceeded by construction, up to the Monte Carlo error of the
null). With `correction="bonferroni"` each statistic is calibrated at `size / 3` and the union of the three flags has
a size of at most `size`.

Missing or degenerate input raises `ValueError` (as the neighbouring `frtb` statistics do). Deterministic: no random
numbers are drawn here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import frtb

DEFAULT_WINDOW = 20
_CORRECTIONS = ("none", "bonferroni")
_N_STATISTICS = 3


@dataclass(frozen=True)
class MZResult:
    """The Mincer-Zarnowitz regression of HPL on RTPL: `intercept` `a` and `slope` `b` (HPL units and unitless), the
    Wald statistic `wald` of `(a, b) = (0, 1)` (HC3 covariance, unitless), its asymptotic chi-square(2) `p_value` (a
    reference only) and the number of observations `n`."""

    intercept: float
    slope: float
    wald: float
    p_value: float
    n: int


@dataclass(frozen=True)
class PLAComplementStatistics:
    """The three statistics of the complement on one desk-year: `wald`, `p_value`, `intercept` and `slope` of the
    Mincer-Zarnowitz regression, `spearman_abs` (Spearman of `|HPL|`, `|RTPL|`), `spearman_roll` (Spearman of their
    `window`-day rolling means), the `window` in days and the number of observations `n`."""

    wald: float
    p_value: float
    intercept: float
    slope: float
    spearman_abs: float
    spearman_roll: float
    window: int
    n: int


@dataclass(frozen=True)
class PLAComplementThresholds:
    """The critical values calibrated on a null: flag when `wald` is above `wald`, when `spearman_abs` is below
    `spearman_abs`, when `spearman_roll` is below `spearman_roll`. `size` is the stated type I error (of the union under
    the Bonferroni `correction`, of each statistic under `none`), `per_statistic_size` the size each statistic was
    calibrated at, `n_null` the number of null desk-years and `window` the rolling window in days."""

    wald: float
    spearman_abs: float
    spearman_roll: float
    size: float
    per_statistic_size: float
    n_null: int
    window: int
    correction: str


@dataclass(frozen=True)
class PLAComplementDecision:
    """Which statistics flag the desk-year against the thresholds, and `flagged`, true when any does."""

    flag_wald: bool
    flag_spearman_abs: bool
    flag_spearman_roll: bool
    flagged: bool


@dataclass(frozen=True)
class PLAComplement:
    """The complement beside the regulatory test for one desk-year: its `statistics`, its `decision` and the regulatory
    `pla_zone` of `frtb.pla_zone_exact` (green, amber or red), unchanged by the complement."""

    statistics: PLAComplementStatistics
    decision: PLAComplementDecision
    pla_zone: str


def _series(hpl: np.ndarray, rtpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, r = frtb._pla_series(hpl, "hpl"), frtb._pla_series(rtpl, "rtpl")
    if np.ptp(h) == 0.0 or np.ptp(r) == 0.0:
        raise ValueError("a PLA series has zero spread: the statistics are undefined")
    return h, r


def _window(window: int) -> int:
    if int(window) != window or not (2 <= window <= frtb.PLA_N - 3):
        raise ValueError(f"window must be an integer from 2 to {frtb.PLA_N - 3} days, got {window}")
    return int(window)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    if den == 0.0:
        raise ValueError("a transformed series has zero spread: the rank correlation is undefined")
    return float(np.clip((ra @ rb) / den, -1.0, 1.0))


def pla_mz_wald(hpl: np.ndarray, rtpl: np.ndarray) -> MZResult:
    """Regress `HPL = a + b RTPL + e` by least squares and test `(a, b) = (0, 1)` with the Wald statistic
    `(beta - (0, 1))' V^-1 (beta - (0, 1))`, `V` the HC3 covariance `(X'X)^-1 X' diag(e_t^2 / (1 - h_t)^2) X (X'X)^-1`
    (`h_t` the leverage). HPL and RTPL in the same unit (decimal of the desk notional), exactly 250 finite
    observations each (MAR32.20); `ValueError` otherwise or when a series has no spread. A model that reproduces the
    HPL exactly (zero residual) gives `wald = 0` for `(0, 1)` and `inf` for any other fit. The p-value is the
    asymptotic chi-square(2) tail, a reference; the decision of this module uses a simulated null."""
    h, r = _series(hpl, rtpl)
    x = np.column_stack([np.ones(h.size), r])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ h)
    e = h - x @ beta
    d = beta - np.array([0.0, 1.0])
    if not np.any(e):
        wald = 0.0 if np.allclose(d, 0.0, atol=1e-12) else float("inf")
    else:
        lev = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
        xs = x * (e / (1.0 - lev))[:, None]
        cov = xtx_inv @ (xs.T @ xs) @ xtx_inv
        wald = float(d @ np.linalg.solve(cov, d))
    return MZResult(float(beta[0]), float(beta[1]), wald, float(stats.chi2.sf(wald, 2)), int(h.size))


def pla_abs_spearman(hpl: np.ndarray, rtpl: np.ndarray) -> float:
    """The Spearman rank correlation (average ranks) of `|HPL|` and `|RTPL|` over the 250 days, in [-1, 1]; unit free.
    `ValueError` as `pla_mz_wald`, or when an absolute series is constant."""
    h, r = _series(hpl, rtpl)
    return _rank_corr(np.abs(h), np.abs(r))


def pla_rolling_abs_spearman(hpl: np.ndarray, rtpl: np.ndarray, window: int = DEFAULT_WINDOW) -> float:
    """The Spearman rank correlation of the `window`-day rolling means of `|HPL|` and `|RTPL|` (250 - window + 1
    overlapping points, `window` in days from 2 to 247, default 20), in [-1, 1]; unit free. A risk model that uses a
    volatility `L` days old moves its scale `L` days late, and this statistic falls as `L` approaches the window.
    `ValueError` as `pla_abs_spearman`, or for a window out of range."""
    w = _window(window)
    h, r = _series(hpl, rtpl)
    kernel = np.ones(w) / w
    return _rank_corr(np.convolve(np.abs(h), kernel, mode="valid"), np.convolve(np.abs(r), kernel, mode="valid"))


def pla_complement_statistics(
        hpl: np.ndarray, rtpl: np.ndarray, window: int = DEFAULT_WINDOW
) -> PLAComplementStatistics:
    """The three statistics of the complement on one 250-day pair (units and refusals as the functions it calls)."""
    mz = pla_mz_wald(hpl, rtpl)
    return PLAComplementStatistics(
        wald=mz.wald, p_value=mz.p_value, intercept=mz.intercept, slope=mz.slope,
        spearman_abs=pla_abs_spearman(hpl, rtpl), spearman_roll=pla_rolling_abs_spearman(hpl, rtpl, window),
        window=_window(window), n=mz.n)


def pla_complement_calibrate(
        null: Sequence[PLAComplementStatistics], size: float = 0.01, correction: str = "bonferroni"
) -> PLAComplementThresholds:
    """Critical values from a simulated null: the statistics of `N` desk-years of a model with nothing planted.

    Each statistic is calibrated at `p = size` (`correction="none"`) or `p = size / 3` (`"bonferroni"`, the union of the
    three flags then has a size of at most `size`). With `k = floor(p N)`, the Wald critical value is the `(k+1)`-th
    largest null Wald and the Spearman critical values the `(k+1)`-th smallest, so at most `k <= p N` null desk-years
    are flagged by each statistic (conservative by construction). `ValueError` for a `size` outside (0, 1), an
    unknown `correction`, an empty null, null statistics of different windows or sizes, non-finite null statistics,
    or a null too small to resolve `p` (`k < 1`, which needs `N >= 1 / p`)."""
    if not (0.0 < float(size) < 1.0):
        raise ValueError(f"size must lie in (0, 1), got {size}")
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be one of {_CORRECTIONS}, got {correction!r}")
    if len(null) == 0:
        raise ValueError("the null needs at least one desk-year")
    windows = {s.window for s in null}
    if len(windows) != 1 or {s.n for s in null} != {frtb.PLA_N}:
        raise ValueError("the null statistics must share one window and be computed on 250 days")
    per = float(size) / (_N_STATISTICS if correction == "bonferroni" else 1)
    n = len(null)
    k = int(np.floor(per * n + 1e-9))
    if k < 1:
        need = int(np.ceil(1 / per))
        raise ValueError(f"{n} null desk-years cannot resolve a per-statistic size of {per:.6g} (need at least {need})")
    wald = np.sort(np.array([s.wald for s in null], dtype=float))[::-1]
    sab = np.sort(np.array([s.spearman_abs for s in null], dtype=float))
    srl = np.sort(np.array([s.spearman_roll for s in null], dtype=float))
    if not (np.all(np.isfinite(wald)) and np.all(np.isfinite(sab)) and np.all(np.isfinite(srl))):
        raise ValueError("the null statistics contain a non-finite value")
    return PLAComplementThresholds(
        wald=float(wald[k]), spearman_abs=float(sab[k]), spearman_roll=float(srl[k]), size=float(size),
        per_statistic_size=per, n_null=n, window=windows.pop(), correction=correction)


def pla_complement_decide(
        statistics: PLAComplementStatistics, thresholds: PLAComplementThresholds
) -> PLAComplementDecision:
    """Flag a statistic when it is beyond its critical value (Wald above, Spearman statistics below). `ValueError`
    when the statistics and the thresholds use different rolling windows."""
    if statistics.window != thresholds.window:
        raise ValueError(f"the statistics use a {statistics.window}-day window and the thresholds {thresholds.window}")
    fw = bool(statistics.wald > thresholds.wald)
    fa = bool(statistics.spearman_abs < thresholds.spearman_abs)
    fr = bool(statistics.spearman_roll < thresholds.spearman_roll)
    return PLAComplementDecision(fw, fa, fr, fw or fa or fr)


def pla_complement(hpl: np.ndarray, rtpl: np.ndarray, thresholds: PLAComplementThresholds) -> PLAComplement:
    """The complement beside the regulatory test for one 250-day pair: the statistics at the window of `thresholds`, the
    decision against them and the regulatory zone `frtb.pla_zone_exact(hpl, rtpl)`, which the complement does not
    change."""
    stat = pla_complement_statistics(hpl, rtpl, thresholds.window)
    return PLAComplement(stat, pla_complement_decide(stat, thresholds), frtb.pla_zone_exact(hpl, rtpl))


__all__ = [
    "DEFAULT_WINDOW",
    "MZResult",
    "PLAComplement",
    "PLAComplementDecision",
    "PLAComplementStatistics",
    "PLAComplementThresholds",
    "pla_abs_spearman",
    "pla_complement",
    "pla_complement_calibrate",
    "pla_complement_decide",
    "pla_complement_statistics",
    "pla_mz_wald",
    "pla_rolling_abs_spearman",
]
