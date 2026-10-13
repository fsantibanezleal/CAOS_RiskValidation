"""CCF and EAD (dossier 04 G.1 to G.3), in the EU convention.

    CCF = (EAD - E(t_r)) / (L(t_r) - E(t_r)),    EAD = E(t_r) + CCF * (L(t_r) - E(t_r))

with t_r the reference date before the default t_d, E the drawn amount and L the limit: the share of the undrawn amount
drawn by default (the LEQ in US terms). In Qi (2009) terms "CCF" is EAD over the outstanding amount and "EADF" is
EAD over the
limit; `to_us_terms` returns all three so that a page can show them without mixing. The three reference-date designs are
Moral (2006) as summarised in dossier 04 G.1 (the chapter was not read: detail UNVERIFIED). A CCF below 0 or above 1 is
kept; an undefined CCF (a fully drawn or overdrawn line) is NaN, never 0 or 1. Contract `engine-lgd-ead-api` 1.0.0,
module ccf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special

from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec
from riskvalidation.lgd_ead.lgd_models import logit_quasi_mle, lstsq

DESIGNS = ("fixed-horizon", "cohort", "variable-horizon")
TREATMENTS = ("none", "censor", "floor-cap", "bounded-scale")


def reference_indices(design: str, default_index: int, cohort_start_index: int | None = None, horizon: int = 12,
                      n_reference: int = 4, step: int = 3) -> np.ndarray:
    """Reference months on the facility path: fixed horizon [t_d - h]; cohort [start of the calendar cohort containing
    t_d]; variable horizon t_d - h + k step for k = 0 .. n_reference - 1. A negative index or one not before the default
    raises."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    td = int(default_index)
    if int(horizon) < 1 or int(n_reference) < 1 or int(step) < 1:
        raise ValueError("horizon, n_reference and step must be at least 1")
    if design == "fixed-horizon":
        idx = [td - int(horizon)]
    elif design == "cohort":
        if cohort_start_index is None:
            raise ValueError("cohort_start_index is required for the cohort design")
        idx = [int(cohort_start_index)]
    else:
        idx = [td - int(horizon) + k * int(step) for k in range(int(n_reference))]
    out = np.array(idx, dtype=np.int64)
    if np.any(out < 0):
        raise ValueError(f"a negative reference index {out.min()}: the reference date precedes the first observation")
    if np.any(out >= td):
        raise ValueError(f"the reference index must be before the default index {td}, got {out.max()}")
    return out


def ccf_observations(drawn: np.ndarray, limit: np.ndarray, ead: float,
                     reference_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ccf, undrawn) at each reference month: undrawn = L - E, ccf = (EAD - E) / undrawn where undrawn > 0 and NaN
    otherwise (the CCF is undefined for a fully drawn or overdrawn line, dossier 04 G.2). Not clipped."""
    d, lim = vec(drawn, "drawn"), vec(limit, "limit")
    same_length(drawn=d, limit=lim)
    e = scalar(ead, "ead")
    idx = np.asarray(reference_index)
    if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in "iu":
        raise ValueError("reference_index must be a non-empty one-dimensional integer array")
    if np.any(idx < 0) or np.any(idx >= d.shape[0]):
        raise ValueError(f"reference_index must lie in [0, {d.shape[0] - 1}]")
    undrawn = lim[idx] - d[idx]
    ccf = np.full(undrawn.shape, np.nan)
    ok = undrawn > 0.0
    ccf[ok] = (e - d[idx][ok]) / undrawn[ok]
    return ccf, undrawn


def aggregate_ccf(ccf: np.ndarray, undrawn: np.ndarray, aggregator: str) -> float:
    """`mean`: the unweighted mean of the finite CCFs; `pooled`: sum(ccf undrawn) / sum(undrawn), which equals
    sum(EAD - E) / sum(L - E). NaN entries are excluded; all-NaN input raises. Not clipped."""
    if aggregator not in ("mean", "pooled"):
        raise ValueError(f"aggregator must be 'mean' or 'pooled', got {aggregator!r}")
    c = vec(ccf, "ccf", allow_nan=True)
    u = vec(undrawn, "undrawn")
    same_length(ccf=c, undrawn=u)
    ok = np.isfinite(c)
    if not ok.any():
        raise ValueError("no finite CCF to aggregate (all observations are undefined)")
    if aggregator == "mean":
        return float(c[ok].mean())
    total = float(u[ok].sum())
    if total <= 0.0:
        raise ValueError("the undrawn amounts of the finite CCFs do not sum to a positive amount")
    return float(np.sum(c[ok] * u[ok]) / total)


@dataclass(frozen=True)
class TreatedCcf:
    values: np.ndarray    # (M,) the CCFs that remain after the treatment
    n_dropped: int        # observations removed (censor) or NaN removed
    n_below_zero: int     # counted before the treatment
    n_above_one: int      # counted before the treatment


def treat_ccf(ccf: np.ndarray, treatment: str) -> TreatedCcf:
    """The treatments of a CCF outside [0, 1] (dossier 04 G.2): `none` keeps every value (flag it with `range_flag`),
    `censor` drops the observations outside [0, 1], `floor-cap` clips to [0, 1] (it biases EAD) and `bounded-scale`
    returns
    the clipped values as the input of `fit_fractional_logit`. NaN entries are dropped first and counted; the counts of
    values below 0 and above 1 are taken on the input whatever the treatment."""
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment!r}")
    c = vec(ccf, "ccf", allow_nan=True, allow_empty=True)
    nan = np.isnan(c)
    v = c[~nan]
    below, above = int((v < 0.0).sum()), int((v > 1.0).sum())
    dropped = int(nan.sum())
    if treatment == "none":
        out = v.copy()
    elif treatment == "censor":
        keep = (v >= 0.0) & (v <= 1.0)
        out = v[keep]
        dropped += int((~keep).sum())
    else:
        out = np.clip(v, 0.0, 1.0)
    return TreatedCcf(values=out, n_dropped=dropped, n_below_zero=below, n_above_one=above)


def ead_from_ccf(drawn: np.ndarray, limit: np.ndarray, ccf: np.ndarray) -> np.ndarray:
    """EAD = drawn + ccf (limit - drawn), not clipped."""
    d, lim, c = vec(drawn, "drawn"), vec(limit, "limit"), vec(ccf, "ccf")
    same_length(drawn=d, limit=lim, ccf=c)
    return d + c * (lim - d)


def to_us_terms(drawn: np.ndarray, limit: np.ndarray, ead: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(leq, ccf_us, eadf): the EU CCF (EAD - E) / (L - E) (the loan equivalent), Qi (2009) CCF = EAD / E and EADF
    = EAD / L.
    Undefined entries (a line not above the drawn amount, a zero drawn amount, a zero limit) are NaN."""
    d, lim, e = vec(drawn, "drawn"), vec(limit, "limit"), vec(ead, "ead")
    same_length(drawn=d, limit=lim, ead=e)
    with np.errstate(divide="ignore", invalid="ignore"):
        leq = np.where(lim > d, (e - d) / (lim - d), np.nan)
        ccf_us = np.where(d != 0.0, e / d, np.nan)
        eadf = np.where(lim != 0.0, e / lim, np.nan)
    return leq, ccf_us, eadf


def small_undrawn_rule(drawn: np.ndarray, limit: np.ndarray, ead: np.ndarray,
                       threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Qi (2009) switches the ratio below a small undrawn amount: (switched, ratio) with switched = (limit - drawn <
    threshold) and ratio = EAD / limit where switched and the EU CCF elsewhere. The threshold is a parameter in the
    currency of the amounts."""
    d, lim, e = vec(drawn, "drawn"), vec(limit, "limit"), vec(ead, "ead")
    same_length(drawn=d, limit=lim, ead=e)
    thr = scalar(threshold, "threshold")
    if thr <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold!r}")
    if np.any(lim <= 0.0):
        raise ValueError("limit must be strictly positive")
    switched = (lim - d) < thr
    with np.errstate(divide="ignore", invalid="ignore"):
        eu = (e - d) / (lim - d)
    return switched, np.where(switched, e / lim, eu)


@dataclass(frozen=True)
class EadFit:
    kind: str                  # "direct-ead" or "utilisation-change"
    names: tuple[str, ...]
    coefficients: np.ndarray   # (K,)
    extras: dict[str, Any] = field(default_factory=dict)   # direct-ead: n_clipped_below, n_clipped_above


def _names(x: np.ndarray, names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(f"x{i}" for i in range(x.shape[1]))
    if len(names) != x.shape[1]:
        raise ValueError(f"names must have one entry per column of X ({x.shape[1]}), got {len(names)}")
    return tuple(str(n) for n in names)


def fit_direct_ead(X: np.ndarray, ead: np.ndarray, limit: np.ndarray, names: tuple[str, ...] | None = None) -> EadFit:
    """Fractional logit of the EADF = EAD / limit (an intercept column in X). An EADF outside [0, 1] (an overlimit EAD)
    is clipped to [0, 1] for the fit and counted in `extras`, never silently. The zero-adjusted gamma mixture of Tong et
    al. (2016) is not in this version."""
    x = mat(X, "X")
    e, lim = vec(ead, "ead"), vec(limit, "limit")
    same_length(X=x, ead=e, limit=lim)
    if np.any(lim <= 0.0):
        raise ValueError("limit must be strictly positive")
    eadf = e / lim
    below, above = int((eadf < 0.0).sum()), int((eadf > 1.0).sum())
    coef = logit_quasi_mle(x, np.clip(eadf, 0.0, 1.0))
    return EadFit(kind="direct-ead", names=_names(x, names), coefficients=coef,
                  extras={"n_clipped_below": below, "n_clipped_above": above})


def fit_utilisation_change(X: np.ndarray, drawn_ref: np.ndarray, drawn_default: np.ndarray, limit_ref: np.ndarray,
                           names: tuple[str, ...] | None = None) -> EadFit:
    """Least squares of the change of utilisation (E(t_d) - E(t_r)) / L(t_r): a ratio-free model (dossier 04 G.3),
    with no
    undrawn amount in the denominator."""
    x = mat(X, "X")
    d0, d1, lim = vec(drawn_ref, "drawn_ref"), vec(drawn_default, "drawn_default"), vec(limit_ref, "limit_ref")
    same_length(X=x, drawn_ref=d0, drawn_default=d1, limit_ref=lim)
    if np.any(lim <= 0.0):
        raise ValueError("limit_ref must be strictly positive")
    return EadFit(kind="utilisation-change", names=_names(x, names), coefficients=lstsq(x, (d1 - d0) / lim), extras={})


def predict_ead(fit: EadFit, X: np.ndarray, drawn_ref: np.ndarray, limit_ref: np.ndarray) -> np.ndarray:
    """`direct-ead`: limit_ref times the fitted EADF, inside [0, limit_ref]; `utilisation-change`: drawn_ref + limit_ref
    times the fitted change, not clipped (flag it in the document)."""
    x = mat(X, "X")
    if x.shape[1] != len(fit.names):
        raise ValueError(f"X must have {len(fit.names)} columns (those of the fit), got {x.shape[1]}")
    d0, lim = vec(drawn_ref, "drawn_ref"), vec(limit_ref, "limit_ref")
    same_length(X=x, drawn_ref=d0, limit_ref=lim)
    if np.any(lim <= 0.0):
        raise ValueError("limit_ref must be strictly positive")
    z = x @ fit.coefficients
    if fit.kind == "direct-ead":
        return lim * special.expit(z)
    if fit.kind == "utilisation-change":
        return d0 + lim * z
    raise ValueError(f"unknown kind {fit.kind!r}")


def ccf_bias(estimate: float, truth: float) -> float:
    """estimate - truth, signed and not clipped."""
    return scalar(estimate, "estimate") - scalar(truth, "truth")
