"""PD term structures: marginal, cumulative and survival from a hazard, competing risks, TTC to PIT.

Identities (Botha and Verster 2025, arXiv:2507.15441, eqs. 7 to 10), with S(0) = 1:

    S(t) = prod_{u <= t} (1 - h(u)),  q(t) = S(t - 1) h(t),  F(t) = sum_{u <= t} q(u) = 1 - S(t),
    h(t) = q(t) / S(t - 1).

Forward PD over (a, b]: PD(a, b) = (S(a) - S(b)) / S(a) (dossier 02 B.2). With a competing exit the default incidence is
q_D(t) = S_all(t - 1) h_D(t) and F_D = cumsum(q_D), which is not 1 - S_all. The one-factor conditional PD
PD(z) = Phi((Phi^-1(PD_TTC) - sqrt(rho) z) / sqrt(1 - rho)) is dossier 02 C.1 (BCBS 2005; Vasicek 2002; Gordy 2003).

Conventions of the contract `engine-provisions-api`: PDs at an estimate site are open (0, 1); months are strictly
increasing integers, period ends; every function is pure and returns new arrays. This module states no compliance claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from riskvalidation.provisions._checks import (
    months_array,
    open_unit,
    scalar_open_unit,
    weights_array,
)


@dataclass(frozen=True)
class TermStructure:
    """Hazard, survival, marginal and cumulative PD per period, shape (..., T); `months` (T,) int64."""

    months: np.ndarray
    hazard: np.ndarray
    survival: np.ndarray
    marginal: np.ndarray
    cumulative: np.ndarray


@dataclass(frozen=True)
class CompetingRisks:
    """Default and prepayment as competing exits. Not a `TermStructure`: `cumulative_default` is the incidence of
    default, not 1 - `survival_all` (the probability of default or prepayment)."""

    months: np.ndarray
    hazard_default: np.ndarray
    hazard_prepay: np.ndarray
    survival_all: np.ndarray
    marginal_default: np.ndarray
    cumulative_default: np.ndarray


def _curve(x: object, months: object, name: str) -> tuple[np.ndarray, np.ndarray]:
    m = months_array(months)
    a = open_unit(x, name)
    if a.ndim < 1 or a.shape[-1] != m.size:
        raise ValueError(f"{name} must have trailing length {m.size} (len(months)), got shape {a.shape}")
    return a, m


def _previous(survival: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones(survival.shape[:-1] + (1,)), survival[..., :-1]], axis=-1)


def from_hazard(hazard: np.ndarray, months: np.ndarray) -> TermStructure:
    """A term structure from the conditional PD per period, 0 < h < 1; S = cumprod(1 - h)."""
    h, m = _curve(hazard, months, "hazard")
    survival = np.cumprod(1.0 - h, axis=-1)
    marginal = _previous(survival) * h
    cumulative = np.cumsum(marginal, axis=-1)
    return TermStructure(months=m, hazard=h, survival=survival, marginal=marginal, cumulative=cumulative)


def from_marginal(marginal: np.ndarray, months: np.ndarray) -> TermStructure:
    """A term structure from the unconditional PD per period; h(t) = q(t) / S(t - 1). Needs 0 < q and sum(q) < 1."""
    q, m = _curve(marginal, months, "marginal")
    total = np.cumsum(q, axis=-1)
    if np.any(total[..., -1] >= 1.0):
        raise ValueError("marginal must sum to less than 1 along the last axis")
    survival_prev = _previous(1.0 - total)
    return from_hazard(q / survival_prev, m)


def from_cumulative(cumulative: np.ndarray, months: np.ndarray) -> TermStructure:
    """A term structure from the cumulative PD from the reporting date; strictly increasing inside (0, 1)."""
    f = np.array(cumulative, dtype=float)
    m = months_array(months)
    if f.ndim < 1 or f.shape[-1] != m.size:
        raise ValueError(f"cumulative must have trailing length {m.size} (len(months)), got shape {f.shape}")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0) or np.any(f >= 1.0) or np.any(np.diff(f, axis=-1) <= 0.0):
        raise ValueError("cumulative must be strictly increasing inside (0, 1) (a monotone cumulative PD)")
    q = np.diff(f, axis=-1, prepend=0.0)
    return from_hazard(q / _previous(1.0 - f), m)


def competing_risks(hazard_default: np.ndarray, hazard_prepay: np.ndarray, months: np.ndarray) -> CompetingRisks:
    """Default and prepayment hazards (cause-specific, per period) to the default incidence.

    `hazard_default` is open (0, 1). `hazard_prepay` lies in [0, 1): zero is allowed so that the single-risk
    structure is the limit of the record. The two hazards sum to less than one.
    """
    hd, m = _curve(hazard_default, months, "hazard_default")
    hp = np.array(hazard_prepay, dtype=float)
    if hp.shape != hd.shape:
        raise ValueError(f"hazard_prepay must have the shape of hazard_default {hd.shape}, got {hp.shape}")
    if not np.all(np.isfinite(hp)) or np.any(hp < 0.0) or np.any(hp >= 1.0):
        raise ValueError("hazard_prepay must lie in [0, 1)")
    if np.any(hd + hp >= 1.0):
        raise ValueError("hazard_default + hazard_prepay must be below 1")
    survival_all = np.cumprod(1.0 - hd - hp, axis=-1)
    marginal_default = _previous(survival_all) * hd
    return CompetingRisks(
        months=m, hazard_default=hd, hazard_prepay=hp, survival_all=survival_all,
        marginal_default=marginal_default, cumulative_default=np.cumsum(marginal_default, axis=-1),
    )


def _at(values: np.ndarray, months: np.ndarray, month: int, name: str, at_zero: float) -> np.ndarray:
    """The value at an end month; month 0 gives `at_zero` (the state at the reporting date)."""
    if month == 0:
        return np.full(values.shape[:-1], at_zero)
    idx = np.flatnonzero(months == month)
    if idx.size == 0:
        raise ValueError(f"{name}={month} is neither 0 nor a member of months {months.tolist()}")
    return values[..., idx[0]]


def forward_pd(ts: TermStructure | CompetingRisks, start_month: int, end_month: int) -> np.ndarray:
    """The conditional PD over (start, end] given survival to start.

    Term structure: (S(a) - S(b)) / S(a). Competing risks: (F_D(b) - F_D(a)) / S_all(a), default only, given survival of
    both exits to a. Shape: the leading shape of the structure.
    """
    if int(end_month) <= int(start_month):
        raise ValueError(f"end_month ({end_month}) must be after start_month ({start_month})")
    if int(start_month) < 0:
        raise ValueError(f"start_month must be 0 or a member of months, got {start_month}")
    a, b = int(start_month), int(end_month)
    if isinstance(ts, CompetingRisks):
        fb = _at(ts.cumulative_default, ts.months, b, "end_month", 0.0)
        fa = _at(ts.cumulative_default, ts.months, a, "start_month", 0.0)
        sa = _at(ts.survival_all, ts.months, a, "start_month", 1.0)
        return (fb - fa) / sa
    sb = _at(ts.survival, ts.months, b, "end_month", 1.0)
    sa = _at(ts.survival, ts.months, a, "start_month", 1.0)
    return (sa - sb) / sa


def annualise_pd(pd: np.ndarray, horizon_years: float) -> np.ndarray:
    """1 - (1 - pd)^(1 / horizon_years): Contraste's documented convention for the SICR comparison (dossier 02 B.5),
    not asserted as the EBA's."""
    if not float(horizon_years) > 0.0:
        raise ValueError(f"horizon_years must be positive, got {horizon_years!r}")
    p = open_unit(pd, "pd")
    return 1.0 - (1.0 - p) ** (1.0 / float(horizon_years))


def _rho(rho: float) -> float:
    return scalar_open_unit(rho, "rho")


def ttc_to_pit(pd_ttc: np.ndarray, rho: float, z: np.ndarray) -> np.ndarray:
    """PD(z) = Phi((Phi^-1(pd_ttc) - sqrt(rho) z) / sqrt(1 - rho)), `z` in standard-normal units (dossier 02 C.1)."""
    p = open_unit(pd_ttc, "pd_ttc")
    r = _rho(rho)
    zz = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(zz)):
        raise ValueError("z contains non-finite values")
    return ndtr((ndtri(p) - math.sqrt(r) * zz) / math.sqrt(1.0 - r))


def implied_z(default_rate: np.ndarray, pd_ttc: float, rho: float) -> np.ndarray:
    """Z = (Phi^-1(pd_ttc) - sqrt(1 - rho) Phi^-1(DR)) / sqrt(rho): the inverse of `ttc_to_pit` (dossier 02 C.1)."""
    dr = open_unit(default_rate, "default_rate")
    p = scalar_open_unit(pd_ttc, "pd_ttc")
    r = _rho(rho)
    return (ndtri(p) - math.sqrt(1.0 - r) * ndtri(dr)) / math.sqrt(r)


def weighted_pit_pd(pd_ttc: float, rho: float, z: np.ndarray, weights: np.ndarray) -> float:
    """sum_s w_s ttc_to_pit(pd_ttc, rho, z_s) over at least two scenarios with probability weights (B5.5.42)."""
    zz = np.array(z, dtype=float)
    if zz.ndim != 1:
        raise ValueError(f"z must be one-dimensional (S,), got shape {zz.shape}")
    w = weights_array(weights, zz.size)
    return float(np.sum(w * ttc_to_pit(pd_ttc, rho, zz)))


def gauss_hermite_pit_pd(pd_ttc: float, rho: float, n_nodes: int = 64) -> float:
    """E_Z[ttc_to_pit(pd_ttc, rho, Z)] for Z ~ N(0, 1) by probabilists' Gauss-Hermite nodes: the non-linearity cross
    check of dossier 02 I.1 item 6. Equals `pd_ttc` to 1e-9 from 32 nodes."""
    n = int(n_nodes)
    if n < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes!r}")
    nodes, w = np.polynomial.hermite_e.hermegauss(n)
    return float(np.sum(w / math.sqrt(2.0 * math.pi) * ttc_to_pit(pd_ttc, rho, nodes)))
