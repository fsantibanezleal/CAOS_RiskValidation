"""Realised (workout) LGD of dossier 04 F.1, its range flags, the treatment of incomplete workouts and the LGD
distribution.

For a defaulted exposure i with exposure at default EAD_i, recoveries R, direct and indirect costs C and additional
drawings
Add after default at times t (years since the default date), discounted at r_i:

    LGD_i = 1 - sum_t (R_t - C_t - Add_t) (1 + r_i)^(-t) / EAD_i

A realised LGD may lie outside [0, 1] (costs, fees, additional drawings): it is kept, never clipped, and flagged by
`range_flag`. Contract: `engine-lgd-ead-api` 1.0.0, module workout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from riskvalidation.lgd_ead._checks import mat, scalar, vec


def eba_discount_rate(interbank_rate: float, add_on: float = 0.05) -> float:
    """The EU workout discount rate convention: an interbank rate at the default date plus an add-on (EBA/GL/2017/16 as
    read in dossier 04 F.1; the paragraph number is UNVERIFIED). The caller may pass any other rate to `workout_lgd`."""
    r = scalar(interbank_rate, "interbank_rate")
    s = scalar(add_on, "add_on")
    if s < 0.0:
        raise ValueError(f"add_on must be non-negative, got {s}")
    return r + s


def workout_lgd(ead: np.ndarray, times: np.ndarray, recoveries: np.ndarray, costs: np.ndarray, drawings: np.ndarray,
                rate: float | np.ndarray) -> np.ndarray:
    """LGD_i = 1 - sum_k (R - C - Add) (1 + r)^(-t) / EAD_i for each exposure; the result (A,) is not clipped."""
    e = vec(ead, "ead")
    if np.any(e <= 0.0):
        raise ValueError("ead must be strictly positive")
    t = mat(times, "times")
    flows = {"recoveries": mat(recoveries, "recoveries"), "costs": mat(costs, "costs"),
             "drawings": mat(drawings, "drawings")}
    for name, f in flows.items():
        if f.shape != t.shape:
            raise ValueError(f"times, recoveries, costs and drawings must have the same shape, got {t.shape} and "
                             f"{f.shape} for {name}")
        if np.any(f < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if t.shape[0] != e.shape[0]:
        raise ValueError(f"ead and times must have the same number of exposures, got {e.shape[0]} and {t.shape[0]}")
    if np.any(t <= 0.0):
        raise ValueError("times must be strictly positive (years since the default date)")
    r = np.array(rate, dtype=float)
    if r.ndim == 0:
        r = np.full(e.shape, float(r))
    if r.shape != e.shape or not np.all(np.isfinite(r)) or np.any(r <= -1.0):
        raise ValueError("rate must be a finite scalar or an (A,) array of decimals above -1")
    net = flows["recoveries"] - flows["costs"] - flows["drawings"]
    pv = (net * (1.0 + r[:, None]) ** (-t)).sum(axis=1)
    return 1.0 - pv / e


def range_flag(values: np.ndarray) -> np.ndarray:
    """`below-zero` for a value under 0, `above-one` for a value over 1, otherwise `in-range`; a NaN raises."""
    v = np.array(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {v.shape}")
    if np.any(np.isnan(v)):
        raise ValueError("values contains NaN")
    return np.where(v < 0.0, "below-zero", np.where(v > 1.0, "above-one", "in-range"))


@dataclass(frozen=True)
class ResolvedWorkouts:
    lgd: np.ndarray        # (M,) the LGDs that remain
    n_excluded: int        # incomplete workouts dropped
    n_projected: int       # incomplete workouts replaced by a projection
    treatment: str         # "excluded" or "projected"


def resolve_incomplete(lgd: np.ndarray, complete: np.ndarray, treatment: str,
                       projected: np.ndarray | None = None) -> ResolvedWorkouts:
    """Treat the incomplete workouts explicitly (dossier 04 F.1): exclude them (biases LGD down) or replace them by a
    projection. The treatment is a required argument and the counts are returned."""
    if treatment not in ("excluded", "projected"):
        raise ValueError(f"treatment must be 'excluded' or 'projected', got {treatment!r}")
    x = vec(lgd, "lgd", allow_nan=True)
    c = np.array(complete, dtype=bool)
    if c.ndim != 1:
        raise ValueError("complete must be one-dimensional")
    if x.shape[0] != c.shape[0]:
        raise ValueError(f"lgd and complete must have the same length, got {x.shape[0]} and {c.shape[0]}")
    if np.any(np.isnan(x[c])):
        raise ValueError("lgd must be finite at every complete workout")
    inc = ~c
    n_inc = int(inc.sum())
    if treatment == "excluded":
        return ResolvedWorkouts(lgd=x[c].copy(), n_excluded=n_inc, n_projected=0, treatment="excluded")
    out = x.copy()
    if n_inc:
        if projected is None:
            raise ValueError("projected is required for the treatment 'projected' when a workout is incomplete")
        p = vec(projected, "projected", allow_nan=True)
        if p.shape != x.shape:
            raise ValueError("projected must have the same length as lgd")
        if not np.all(np.isfinite(p[inc])):
            raise ValueError("projected must be finite at every incomplete workout")
        out[inc] = p[inc]
    return ResolvedWorkouts(lgd=out, n_excluded=0, n_projected=n_inc, treatment="projected")


def closed_before(close_dates: np.ndarray, cutoff: str) -> np.ndarray:
    """True only when the workout closed strictly before the cutoff; an open workout (NaT) is false. The
    training-set rule
    of dossier 04 (leakage rule 5)."""
    d = np.asarray(close_dates)
    if d.ndim != 1 or d.dtype.kind != "M":
        raise ValueError("close_dates must be a one-dimensional datetime64 array")
    try:
        cut = np.datetime64(cutoff, "D")
    except ValueError as exc:
        raise ValueError(f"cutoff must be an ISO date, got {cutoff!r}") from exc
    if np.isnat(cut):
        raise ValueError(f"cutoff must be an ISO date, got {cutoff!r}")
    dd = d.astype("datetime64[D]")
    return (~np.isnat(dd)) & (dd < cut)


@dataclass(frozen=True)
class LgdDistribution:
    edges: np.ndarray      # (B + 1,) strictly increasing, edges[0] = 0.0 and edges[-1] = 1.0
    counts: np.ndarray     # (B,) int64 values inside [0, 1], bin i is [edges[i], edges[i+1]), the last bin is closed
    n: int
    n_below_zero: int
    n_above_one: int
    mass_zero: int         # values exactly 0
    mass_one: int          # values exactly 1


def lgd_distribution(lgd: np.ndarray, edges: np.ndarray) -> LgdDistribution:
    """Counts per bin inside [0, 1], the values outside counted apart, and the exact masses at 0 and 1 (dossier 04
    F.2)."""
    x = np.array(lgd, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("lgd must be a non-empty one-dimensional array")
    if np.any(np.isnan(x)):
        raise ValueError("lgd contains NaN")
    e = _edges(edges)
    inside = (x >= 0.0) & (x <= 1.0)
    counts, _ = np.histogram(x[inside], bins=e)
    return LgdDistribution(
        edges=e, counts=counts.astype(np.int64), n=int(x.size), n_below_zero=int((x < 0.0).sum()),
        n_above_one=int((x > 1.0).sum()), mass_zero=int((x == 0.0).sum()), mass_one=int((x == 1.0).sum()),
    )


def _edges(edges: Any) -> np.ndarray:
    e = np.array(edges, dtype=float)
    if (e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0.0)
            or e[0] != 0.0 or e[-1] != 1.0):
        raise ValueError("edges must be strictly increasing with edges[0] = 0.0 and edges[-1] = 1.0")
    return e
