"""The dependence-uncertainty band every aggregate is shown with (dossier 05 G.3, l.366; the SDD): the lower bound,
the simple
sum, the variance-covariance, the Gaussian copula, one Student t copula per degrees-of-freedom value and the worst
case, on
the same marginals. The band is returned whole, never one diversified number.

Every row is economic capital on the unexpected-loss basis, `EC = VaR - EL` (dossier 04 l.652-653); the expected loss is
additive, so the simple sum of the marginal EC equals the comonotonic EC of the total. The simple sum is the headline
capital
(`status = "headline-capital"`); every other row is information, not capital (`"information-not-capital"`, dossier 05
l.519). A row above the headline is a true result (VaR is not subadditive, dossier 05 l.366): it is reported by
`validation.aggregation.diversification_sign`, never repaired and never refused here. The lower-bound row is signed:
the best
VaR less the sum of the expected losses may be negative and is reported as computed.

A marginal with an infinite-mean severity has no expected loss: the caller passes it as `ec = VaR` and `el = 0.0`, so
that
every row is built on the EC the document carries. The caller converts a marginal read at another level or horizon,
never the
engine. Contract: `engine-aggregation-api` 1.0.0, module aggregation.band.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.aggregation.bounds import worst_var_bounds
from riskvalidation.aggregation.copulas import DEFAULT_CHUNK, copula_aggregate
from riskvalidation.aggregation.simple import _matrix, check_correlation, simple_sum, variance_covariance
from riskvalidation.portfolio.portfolio import check_level, float_vector

Ppf = Callable[[np.ndarray], np.ndarray]
METHODS = ("lower-bound", "simple-sum", "variance-covariance", "gaussian-copula", "t-copula", "worst-case")
HEADLINE = "headline-capital"
INFORMATION = "information-not-capital"
CAPITAL_BASIS = "economic-capital-var-less-el"


@dataclass(frozen=True)
class BandRow:
    """One row of the band; maps onto the document row `outputs.aggregation.band[*]` (`method`, `nu`, `capital_clp_m`,
    `diversification_factor`, `status`, `mc_se_clp_m`). `capital` is money (EC), `diversification_factor` is
    capital / simple sum, `mc_se` the Monte Carlo standard error where the row is simulated."""

    method: str
    nu: float | None
    capital: float
    diversification_factor: float
    status: str
    mc_se: float | None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.status != (HEADLINE if self.method == "simple-sum" else INFORMATION):
            raise ValueError(f"status {self.status!r} does not follow method {self.method!r}")
        if (self.nu is not None) != (self.method == "t-copula"):
            raise ValueError("nu must be set exactly for the t-copula row")


@dataclass(frozen=True)
class Band:
    """The rows (lower bound, simple sum, variance-covariance, Gaussian, one t row per nu, worst case), the `headline`
    (the
    capital of the single simple-sum row), the horizon in years, the confidence level and the capital basis."""

    rows: tuple[BandRow, ...]
    headline: float
    horizon_years: int
    confidence_level: float
    capital_basis: str


def _check_nu_set(nu_set: Sequence[float]) -> tuple[float, ...]:
    nus = tuple(float(v) for v in nu_set)
    if not nus:
        raise ValueError("nu_set is empty")
    if not all(math.isfinite(v) and v > 2.0 for v in nus):
        raise ValueError(f"nu_set values must be finite and above 2, got {list(nu_set)}")
    if any(b <= a for a, b in zip(nus, nus[1:], strict=False)):
        raise ValueError(f"nu_set must be ascending and distinct, got {list(nu_set)}")
    return nus


# A convention, not a derived constant: `marginal_ppf_sample` interpolates while `var_from_sample` takes the order
# statistic, so ec and ppf(level) - el agree only to sampling resolution; 1 % relative (absolute below 1).
EC_CONSISTENCY_TOLERANCE = 0.01


def dependence_band(ppfs: Sequence[Ppf], ec: np.ndarray, el: np.ndarray, corr: np.ndarray, nu_set: Sequence[float],
                    level: float = 0.999, horizon_years: int = 1, n_draws: int = 1_000_000, seed: int = 0,
                    chunk_size: int = DEFAULT_CHUNK, bounds_n: int = 100_000, bounds_epsilon: float = 1e-3,
                    ec_consistency_tolerance: float = EC_CONSISTENCY_TOLERANCE) -> Band:
    """The band on the marginals `ppfs` (money quantile functions), their economic capital `ec` and expected loss `el`
    (K,), the correlation `corr` of the copulas and the t degrees of freedom `nu_set` (above 2, ascending, distinct).

    The rows: `simple_sum`, `variance_covariance`, `copula_aggregate` for the Gaussian copula and for each nu (the same
    `seed` on every row), and `worst_var_bounds` (its `lower_range[0]` and `upper_range[1]` less the sum of `el`) for
    the
    lower and the worst case. One horizon and one confidence level apply to every row: `level` lies in (0, 1) and
    `horizon_years` is a positive integer. Golden: ec (3, 4), el (0, 0), R = [[1, 0.5], [0.5, 1]] gives the simple-sum
    row 7.0
    and the variance-covariance row 6.0828, with lower <= 6.0828 <= worst."""
    lv = check_level(level, "level")
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, (int, np.integer)) or horizon_years < 1:
        raise ValueError(f"horizon_years must be a positive integer, got {horizon_years}")
    nus = _check_nu_set(nu_set)
    capital_ec = float_vector(ec, "ec")
    if np.any(capital_ec < 0.0):
        raise ValueError("ec must not contain a negative element")
    losses = float_vector(el, "el")
    if np.any(losses < 0.0):
        raise ValueError("el must not contain a negative element")
    r = _matrix(corr, "corr")
    check_correlation(r)
    k = capital_ec.size
    if not (len(ppfs) == losses.size == r.shape[0] == k):
        raise ValueError(f"ppfs, ec, el and corr must have the same length, got {len(ppfs)}, {k}, {losses.size} and "
                         f"{r.shape[0]}")
    headline = simple_sum(capital_ec)
    if not headline > 0.0:
        raise ValueError("ec must have a positive sum: the diversification factors are undefined")
    tol = float(ec_consistency_tolerance)
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"ec_consistency_tolerance must be a nonnegative number, got {ec_consistency_tolerance}")
    for idx, ppf in enumerate(ppfs):
        implied = float(np.ravel(ppf(np.array([lv])))[0]) - float(losses[idx])
        if not abs(float(capital_ec[idx]) - implied) <= tol * max(1.0, abs(float(capital_ec[idx]))):
            raise ValueError(f"ec does not match the marginals: marginal {idx} has ec {float(capital_ec[idx])} but "
                             f"ppf(level) - el is {implied} (tolerance {tol})")
    el_total = float(losses.sum())
    bounds = worst_var_bounds(ppfs, lv, bounds_n, bounds_epsilon, seed=seed)
    rows: list[tuple[str, float | None, float, float | None]] = [
        ("lower-bound", None, bounds.lower_range[0] - el_total, None),
        ("simple-sum", None, headline, None),
        ("variance-covariance", None, variance_covariance(capital_ec, r), None),
    ]
    for family, nu in (("gaussian", None), *(("t", v) for v in nus)):
        agg = copula_aggregate(ppfs, r, family, nu, el_total, lv, n_draws, seed, chunk_size)
        rows.append(("gaussian-copula" if family == "gaussian" else "t-copula", nu, agg.ec, agg.var_se))
    rows.append(("worst-case", None, bounds.upper_range[1] - el_total, None))
    band_rows = tuple(
        BandRow(method=method, nu=nu, capital=float(capital), diversification_factor=float(capital) / headline,
                status=HEADLINE if method == "simple-sum" else INFORMATION, mc_se=se)
        for method, nu, capital, se in rows)
    return Band(rows=band_rows, headline=headline, horizon_years=int(horizon_years), confidence_level=lv,
                capital_basis=CAPITAL_BASIS)
