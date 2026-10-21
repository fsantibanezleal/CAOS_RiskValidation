"""The portfolio as data and its loss measures (dossier 04 H.1, contract `engine-portfolio-api` 1.0.0).

The default-mode loss is L = sum_i EAD_i LGD_i D_i with the expected LGD deterministic; EL = sum_i EAD_i E[LGD_i]
PD_i, VaR_q(L) = inf{l : P(L <= l) >= q}, ES_q(L) = (1/(1 - q)) int_q^1 VaR_u du (equal to E[L | L >= VaR_q] for a
continuous loss; Acerbi and Tasche 2002, doi:10.1016/S0378-4266(02)00283-2) and the unexpected-loss capital EC_q =
VaR_q - EL (BCBS 2005 explanatory note, section 4.4).

For a sample of S losses sorted ascending L_(1) <= ... <= L_(S) and k = ceil(q S - 1e-9) the estimators are VaR =
L_(k) and ES = (1/(1 - q)) [sum_{i>k} L_(i) / S + (k / S - q) L_(k)] (the Acerbi-Tasche weights: the boundary scenario
carries the fractional mass, so the tail has exactly mass 1 - q). The tolerance guards 0.9 * 10 = 9.000000000000002.

The large homogeneous portfolio anchors come from `riskvalidation.engines.vasicek` and are never recomputed here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import integrate
from scipy.special import ndtri

from riskvalidation.engines.vasicek import conditional_pd, lhp_quantile

QUANTILE_GUARD = 1e-9           # k = ceil(q S - guard): 0.9 * 10 = 9.000000000000002 must give k = 9
LHP_ES_STEPS = 4000             # Simpson steps of the large-portfolio expected shortfall
LHP_ES_SPAN = 12.0              # the quadrature starts this far below G(1 - q)
_CURRENCY = re.compile(r"^[A-Z]{3}$")


# ---------------------------------------------------------------------------------------------------------------------
# input checks shared by the portfolio modules


def float_vector(x: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    """A new one-dimensional finite float64 array."""
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def check_level(q: Any, name: str = "q") -> float:
    """A confidence level as a float in (0, 1); ``ValueError`` naming ``name`` otherwise."""
    v = float(q)
    if not (math.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must lie in (0, 1), got {q}")
    return v


def check_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """Each level of ``levels`` checked by :func:`check_level`, as a tuple; an empty sequence is refused."""
    out = tuple(check_level(q, "level") for q in levels)
    if not out:
        raise ValueError("levels is empty")
    return out


def positive_int(x: Any, name: str) -> int:
    """``x`` as a positive integer; ``ValueError`` naming ``name`` otherwise."""
    v = int(x)
    if v != x or v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {x}")
    return v


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class Portfolio:
    """N obligors with F systematic factors; money in `currency`, PD, LGD and loadings decimals.

    One factor (F = 1): loadings[i, 0] = sqrt(rho_i) and the asset correlation of two names is loadings[i, 0]
    loadings[j, 0]. A name's asset return is A_i = beta_i' Z + sqrt(1 - beta_i' beta_i) e_i with Z ~ N(0,
    factor_corr).
    """

    ead: np.ndarray
    pd: np.ndarray
    lgd: np.ndarray
    loadings: np.ndarray
    factor_corr: np.ndarray
    group: np.ndarray
    group_ids: tuple[str, ...]
    lgd_gamma: float
    currency: str
    label: str

    def __post_init__(self) -> None:
        ead = float_vector(self.ead, "ead")
        n = ead.size
        pd = float_vector(self.pd, "pd")
        lgd = float_vector(self.lgd, "lgd")
        if pd.size != n or lgd.size != n:
            raise ValueError(f"ead, pd and lgd must have the same length, got {n}, {pd.size} and {lgd.size}")
        loadings = np.array(self.loadings, dtype=float)
        if loadings.ndim != 2 or loadings.shape[0] != n or loadings.shape[1] < 1:
            raise ValueError(f"loadings must have shape (N, F) with F >= 1, got {loadings.shape}")
        if not np.all(np.isfinite(loadings)):
            raise ValueError("loadings contains non-finite values")
        f = loadings.shape[1]
        corr = np.array(self.factor_corr, dtype=float)
        if corr.shape != (f, f):
            raise ValueError(f"factor_corr must have shape ({f}, {f}), got {corr.shape}")
        if not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, atol=1e-12, rtol=0):
            raise ValueError("factor_corr must be a finite symmetric matrix")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12, rtol=0):
            raise ValueError("factor_corr must have a unit diagonal")
        if float(np.linalg.eigvalsh(corr).min()) <= 1e-12:
            raise ValueError("factor_corr must be positive definite")
        group = np.array(self.group)
        if group.ndim != 1 or group.size != n:
            raise ValueError(f"group must have shape ({n},), got {group.shape}")
        if group.dtype.kind not in "iu":
            if not (group.dtype.kind == "f" and np.all(np.mod(group, 1) == 0)):
                raise ValueError("group must hold integer indexes")
        group = group.astype(np.int64)
        ids = tuple(self.group_ids)
        if not ids or any((not isinstance(g, str)) or not g.strip() for g in ids):
            raise ValueError("group_ids must hold non-empty strings")
        if np.any(group < 0) or np.any(group >= len(ids)):
            raise ValueError("group holds an index outside group_ids")
        if np.any(ead <= 0.0):
            raise ValueError("ead must be strictly positive")
        if np.any(pd <= 0.0) or np.any(pd >= 1.0):
            raise ValueError("pd must lie strictly between 0 and 1")
        if np.any(lgd < 0.0) or np.any(lgd > 1.0):
            raise ValueError("lgd must lie in [0, 1]")
        if np.any((loadings**2).sum(axis=1) >= 1.0):
            raise ValueError("every loading row must have a sum of squares below 1")
        if np.any(np.einsum("ij,jk,ik->i", loadings, corr, loadings) >= 1.0):
            raise ValueError("every loading row must have a systematic variance beta' factor_corr beta below 1")
        gamma = float(self.lgd_gamma)
        if not (math.isfinite(gamma) and gamma >= 0.0):
            raise ValueError("lgd_gamma must be a non-negative number")
        if not isinstance(self.currency, str) or not _CURRENCY.match(self.currency):
            raise ValueError(f"currency must be three capital letters (ISO 4217), got {self.currency!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        for name, value in (("ead", ead), ("pd", pd), ("lgd", lgd), ("loadings", loadings), ("factor_corr", corr),
                            ("group", group)):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "group_ids", ids)
        object.__setattr__(self, "lgd_gamma", gamma)


def idiosyncratic_scale(p: Portfolio) -> np.ndarray:
    """sqrt(1 - beta_i' factor_corr beta_i) per name: the weight of the idiosyncratic term that keeps the asset
    return at unit variance, so that the marginal default probability is PD_i. It equals sqrt(1 - beta_i' beta_i) of the
    contract for one factor and for uncorrelated factors (and for every portfolio `portfolio_from_groups` builds,
    where a name loads on one factor)."""
    return np.sqrt(1.0 - np.einsum("ij,jk,ik->i", p.loadings, p.factor_corr, p.loadings))


def portfolio_from_groups(groups: Sequence[Mapping[str, Any]], currency: str, lgd_gamma: float,
                          factor_corr: np.ndarray | None = None, one_factor: bool = False,
                          label: str = "illustrative") -> Portfolio:
    """Expand groups of identical names (keys id, n_names, ead_each, pd, lgd, loading, sector) in group order.

    With `one_factor` every name loads on the single factor with its group's loading; otherwise there is one factor
    per distinct sector in order of first appearance, a name loads on its sector's factor, and the correlation between
    the sectors is an input (`factor_corr`, required for more than one sector, never defaulted).
    """
    if not groups:
        raise ValueError("groups is empty")
    keys = ("id", "n_names", "ead_each", "pd", "lgd", "loading", "sector")
    for g in groups:
        missing = [k for k in keys if k not in g]
        if missing:
            raise ValueError(f"group {g.get('id', '?')!r} lacks {missing}")
    counts = [positive_int(g["n_names"], f"n_names of group {g['id']!r}") for g in groups]
    sectors: list[str] = []
    for g in groups:
        if g["sector"] not in sectors:
            sectors.append(g["sector"])
    if one_factor:
        f, corr = 1, np.eye(1)
    else:
        f = len(sectors)
        if f > 1 and factor_corr is None:
            raise ValueError("factor_corr is required for more than one sector: the correlation between sectors is an "
                             "input, never a default")
        corr = np.eye(1) if factor_corr is None else np.array(factor_corr, dtype=float)
    n = int(sum(counts))
    ead, pd, lgd = np.empty(n), np.empty(n), np.empty(n)
    loadings = np.zeros((n, f))
    group = np.empty(n, dtype=np.int64)
    start = 0
    for idx, (g, c) in enumerate(zip(groups, counts, strict=True)):
        sl = slice(start, start + c)
        ead[sl], pd[sl], lgd[sl], group[sl] = float(g["ead_each"]), float(g["pd"]), float(g["lgd"]), idx
        loadings[sl, 0 if one_factor else sectors.index(g["sector"])] = float(g["loading"])
        start += c
    return Portfolio(ead=ead, pd=pd, lgd=lgd, loadings=loadings, factor_corr=corr, group=group,
                     group_ids=tuple(str(g["id"]) for g in groups), lgd_gamma=lgd_gamma, currency=currency, label=label)


def expected_loss(p: Portfolio) -> float:
    """EL = sum_i EAD_i LGD_i PD_i (dossier 04 H.1, the independent-LGD case), money in the portfolio currency."""
    return float(np.sum(p.ead * p.lgd * p.pd))


@dataclass(frozen=True)
class LossMeasures:
    """EL, VaR, ES and EC at one level, money in the portfolio currency.

    `es` is None where the method defines none; `var_se` and `es_se` (Monte Carlo standard errors) are both None or
    both set.
    """

    level: float
    el: float
    var: float
    es: float | None
    ec: float
    var_se: float | None
    es_se: float | None


def lhp_measures(total_ead: float, lgd: float, pd: float, rho: float,
                 levels: Sequence[float]) -> tuple[LossMeasures, ...]:
    """The large homogeneous portfolio (rung C1, dossier 04 C.4): EL, VaR, ES and EC per level, no standard error.

    VaR = total_ead lgd lhp_quantile(q, pd, rho) from `engines.vasicek`. The loss fraction is p(Z) with p the
    conditional PD and p decreasing in Z, so the tail of mass 1 - q is Z <= G(1 - q) and ES = total_ead lgd E[p(Z) | Z
    <= G(1 - q)], computed by Simpson's rule on [G(1 - q) - 12, G(1 - q)] with 4000 steps.
    """
    total = float(total_ead)
    lg = float(lgd)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError("total_ead must be positive")
    if not (math.isfinite(lg) and 0.0 <= lg <= 1.0):
        raise ValueError("lgd must lie in [0, 1]")
    lv = check_levels(levels)
    lhp_quantile(0.5, pd, rho)  # validates pd and rho (open interval) through the engine
    el = total * lg * pd
    out = []
    for q in lv:
        var = total * lg * float(lhp_quantile(q, pd, rho))
        z_q = float(ndtri(1.0 - q))
        z = np.linspace(z_q - LHP_ES_SPAN, z_q, LHP_ES_STEPS + 1)
        dens = np.asarray(conditional_pd(pd, rho, z)) * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        es = total * lg * float(integrate.simpson(dens, x=z)) / (1.0 - q)
        out.append(LossMeasures(level=q, el=el, var=var, es=es, ec=var - el, var_se=None, es_se=None))
    return tuple(out)


def _sorted_losses(losses: Any) -> np.ndarray:
    return np.sort(float_vector(losses, "losses"))


def quantile_rank(q: float, size: int) -> int:
    """The rank ``k = ceil(q size - 1e-9)`` of the order statistic at level ``q``, clipped to ``[1, size]``."""
    return min(max(int(math.ceil(q * size - QUANTILE_GUARD)), 1), size)


def var_from_sample(losses: np.ndarray, q: float) -> float:
    """VaR = L_(k) with k = ceil(q S - 1e-9); independent of the order of the losses."""
    q = check_level(q)
    x = _sorted_losses(losses)
    return float(x[quantile_rank(q, x.size) - 1])


def es_from_sample(losses: np.ndarray, q: float) -> float:
    """The Acerbi-Tasche estimator of the module docstring: the tail mass is exactly 1 - q."""
    q = check_level(q)
    x = _sorted_losses(losses)
    s = x.size
    k = quantile_rank(q, s)
    boundary = max(0.0, k / s - q) * x[k - 1]
    return float((x[k:].sum() / s + boundary) / (1.0 - q))


def measures_from_sample(losses: np.ndarray, levels: Sequence[float], el: float,
                         n_batches: int = 20) -> tuple[LossMeasures, ...]:
    """VaR, ES, EC = VaR - EL and the batch-means standard errors of a loss sample.

    `el` is the portfolio's expected loss, supplied so the measure does not re-estimate it from the sample. A sample
    whose VaR falls below `el` raises ValueError rather than returning a negative capital. A sample too short for
    `n_batches` batches of 10 losses has no standard error: `var_se` and `es_se` are then both None, never a number.
    """
    from riskvalidation.portfolio.copula import mc_standard_errors  # copula imports this module

    x = float_vector(losses, "losses")
    lv = check_levels(levels)
    e = float(el)
    if not math.isfinite(e):
        raise ValueError("el must be finite")
    nb = positive_int(n_batches, "n_batches")
    if nb < 2:
        raise ValueError("n_batches must be at least 2")
    out = []
    for q in lv:
        var, es = var_from_sample(x, q), es_from_sample(x, q)
        if var < e - 1e-9 * max(1.0, abs(e)):
            raise ValueError(f"the sample VaR {var} at level {q} is below the expected loss {e}: no capital to report "
                             "(the sample is too small or the expected loss is wrong)")
        se_v = se_e = None
        if x.size >= 10 * nb:
            se_v, se_e = mc_standard_errors(x, q, nb)
        out.append(LossMeasures(level=q, el=e, var=var, es=es, ec=var - e, var_se=se_v, es_se=se_e))
    return tuple(out)

__all__ = [
    "LossMeasures",
    "Portfolio",
    "check_level",
    "check_levels",
    "es_from_sample",
    "expected_loss",
    "float_vector",
    "idiosyncratic_scale",
    "lhp_measures",
    "measures_from_sample",
    "portfolio_from_groups",
    "positive_int",
    "quantile_rank",
    "var_from_sample",
]
