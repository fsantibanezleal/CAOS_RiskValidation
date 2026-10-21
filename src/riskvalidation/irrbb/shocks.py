"""The six SRP31 interest-rate shock shapes, the shock-size tables as data, the CMF Annex 1 table and the post-shock
floors (dossier 01 H.2, H.3 and H.5; dossier 03 F.2).

Shock sizes are basis points. A size outside the transcribed tables is never invented: ``shock_size`` refuses a
currency without an entry unless the caller supplies the size with a bilingual label and the method. Contract:
``engine-irrbb-api`` 1.1.0, module shocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.lgd_ead._checks import scalar, vec

SHOCKS = ("parallel-up", "parallel-down", "steepener", "flattener", "short-up", "short-down")
"""The order of SRP31.119 Table 5 and SRP31.125 Table 6; every per-shock array and list follows it."""

_CURRENCIES = (
    "ARS", "AUD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR", "INR", "JPY", "KRW", "MXN", "RUB", "SAR",
    "SEK", "SGD", "TRY", "USD", "ZAR",
)

# SRP31.90 Table 2, the 2024 calibration (dossier 01 H.3; in force 2026-01-01; data January 2000 to December 2023;
# 99.9th percentile; rounding 25 bp). Rows: parallel, short, long, in the order of _CURRENCIES.
_T2024 = (
    (400, 350, 400, 200, 175, 225, 225, 275, 225, 400, 325, 100, 225, 400, 400, 275, 275, 175, 400, 200, 325),
    (500, 425, 500, 275, 250, 300, 350, 425, 375, 500, 475, 100, 350, 500, 500, 375, 425, 250, 500, 300, 500),
    (300, 300, 300, 175, 200, 150, 200, 250, 200, 300, 225, 100, 225, 200, 300, 250, 200, 225, 300, 225, 300),
)
# BCBS d368 Annex 2 Table 1, the 2016 calibration (dossier 01 H.3; data 2000 to 2015; 99th percentile of global factors;
# rounding 50 bp). The 2016 PDF text layer interleaves rows; the row assignment was checked against the column count
# and the EUR 200/250/100 and USD 200/300/150 values (verified with that caveat).
_T2016 = (
    (400, 300, 400, 200, 100, 250, 200, 250, 200, 400, 400, 100, 300, 400, 400, 200, 200, 150, 400, 200, 400),
    (500, 450, 500, 300, 150, 300, 250, 300, 250, 500, 500, 100, 400, 500, 500, 300, 300, 200, 500, 300, 500),
    (300, 200, 300, 150, 100, 150, 100, 150, 100, 350, 300, 100, 200, 300, 300, 150, 150, 100, 300, 150, 300),
)
# RAN 21-13 Annex 1 Hoja 9 (dossier 03 F.2); a dash in the source is None.
_CMF = {
    "canasta-1": (200.0, 300.0, 150.0),
    "canasta-2": (400.0, 500.0, 300.0),
    "CLP": (300.0, 400.0, 200.0),
    "UR-nii": (300.0, None, None),
    "UR-eve": (200.0, 250.0, 150.0),
}
_CALIBRATIONS = {
    "2024": ("srp31-table-2024", {c: tuple(float(r[i]) for r in _T2024) for i, c in enumerate(_CURRENCIES)}),
    "2016": ("srp31-table-2016", {c: tuple(float(r[i]) for r in _T2016) for i, c in enumerate(_CURRENCIES)}),
    "cmf-annex1": ("cmf-annex1", _CMF),
}
_UR_ALIAS = {"CLF": "UR-eve"}  # the UF currency takes the delta EVE size of the UR column


def shock_scalars(midpoints: np.ndarray, x: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """``(alpha_short, alpha_long)`` with ``alpha_short = exp(-t / x)`` and ``alpha_long = 1 - exp(-t / x)``
    (SRP31.91 and SRP31.92; ``x`` years, 4 for most currencies)."""
    t = vec(midpoints, "midpoints")
    if not scalar(x, "x") > 0.0:
        raise ValueError("x must be positive")
    a_short = np.exp(-t / x)
    return a_short, 1.0 - a_short


def shock_curve(shock: str, parallel_bp: float, short_bp: float, long_bp: float, midpoints: np.ndarray,
                x: float = 4.0) -> np.ndarray:
    """The shock in basis points at each midpoint (SRP31.90 to SRP31.92): parallel ``+-parallel_bp``; short
    ``+-short_bp alpha_short``; steepener ``-0.65 |short| + 0.9 |long|``; flattener ``+0.8 |short| - 0.6 |long|``,
    with ``|short| = short_bp alpha_short`` and ``|long| = long_bp alpha_long``."""
    if shock not in SHOCKS:
        raise ValueError(f"unknown shock {shock!r}; expected one of {SHOCKS}")
    sizes = [scalar(parallel_bp, "parallel_bp"), scalar(short_bp, "short_bp"), scalar(long_bp, "long_bp")]
    if min(sizes) < 0.0:
        raise ValueError("shock sizes must be nonnegative")
    a_short, a_long = shock_scalars(midpoints, x)
    par, short, long_ = sizes
    if shock == "parallel-up":
        return np.full(a_short.shape, par)
    if shock == "parallel-down":
        return np.full(a_short.shape, -par)
    if shock == "short-up":
        return short * a_short
    if shock == "short-down":
        return -short * a_short
    if shock == "steepener":
        return -0.65 * short * a_short + 0.9 * long_ * a_long
    return 0.8 * short * a_short - 0.6 * long_ * a_long


def shock_table(calibration: str) -> dict[str, tuple[float, float | None, float | None]]:
    """``{currency: (parallel_bp, short_bp, long_bp)}`` of a calibration, ``None`` where the source has a dash. A new
    dictionary at every call. ``calibration`` in ``2024`` (SRP31.90 Table 2), ``2016`` (BCBS d368) and ``cmf-annex1``
    (RAN 21-13 Annex 1 Hoja 9, whose keys are the baskets and ``CLP``, ``UR-nii``, ``UR-eve``)."""
    if calibration not in _CALIBRATIONS:
        raise ValueError(f"unknown calibration {calibration!r}; expected one of {tuple(_CALIBRATIONS)}")
    return dict(_CALIBRATIONS[calibration][1])


@dataclass(frozen=True)
class ShockSize:
    currency: str
    parallel_bp: float
    short_bp: float
    long_bp: float
    origin: str                  # srp31-table-2024 | srp31-table-2016 | cmf-annex1 | contraste-estimate
    label: dict | None           # {"en": ..., "es": ...}, set only for contraste-estimate
    method: str | None


def shock_size(currency: str, calibration: str,
               estimate: tuple[float, float, float, dict, str] | None = None) -> ShockSize:
    """The size of a currency under a calibration. A currency with an entry returns it with the origin of the
    calibration (``CLF`` takes the ``UR-eve`` row of the CMF table). A currency without an entry raises ``ValueError``
    unless ``estimate = (parallel_bp, short_bp, long_bp, label, method)`` is supplied, which gives ``origin =
    "contraste-estimate"``; the label must hold a non-empty ``en`` and ``es`` text and the method must be non-empty.
    An estimate for a currency that has a tabled size is refused: a tabled size never carries a label."""
    if calibration not in _CALIBRATIONS:
        raise ValueError(f"unknown calibration {calibration!r}; expected one of {tuple(_CALIBRATIONS)}")
    origin, table = _CALIBRATIONS[calibration]
    key = _UR_ALIAS.get(currency, currency) if calibration == "cmf-annex1" else currency
    if key in table:
        if estimate is not None:
            raise ValueError(f"{currency} has a tabled size under {calibration}; an estimate is refused for it")
        par, short, long_ = table[key]
        if short is None or long_ is None:
            raise ValueError(f"{currency} under {calibration} has no short and long size "
                             f"({key} is a parallel-only row)")
        return ShockSize(currency, float(par), float(short), float(long_), origin, None, None)
    if estimate is None:
        hint = "; CLP: use the CMF prescription or label a Contraste estimate" if currency == "CLP" else ""
        raise ValueError(f"{currency} has no shock size under {calibration}: supply the size with its label and "
                         f"method (a size is never invented{hint})")
    par, short, long_, label, method = estimate
    sizes = [scalar(par, "parallel_bp"), scalar(short, "short_bp"), scalar(long_, "long_bp")]
    if min(sizes) < 0.0:
        raise ValueError("shock sizes must be nonnegative")
    if not isinstance(label, dict) or not all(isinstance(label.get(k), str) and label[k].strip() for k in ("en", "es")):
        raise ValueError("an estimate needs a label with non-empty 'en' and 'es' texts")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("an estimate needs a non-empty method")
    return ShockSize(currency, sizes[0], sizes[1], sizes[2], "contraste-estimate", dict(label), method)


def apply_floor(base_rates: np.ndarray, shock_bp: np.ndarray, floor: float | np.ndarray | None) -> np.ndarray:
    """``max(r0 + shock / 1e4, min(floor, r0))`` (SRP31.93: a floor not above zero; RAN 21-13 Annex 1: -2 percent for
    UR rates, 0 for other currencies). ``floor`` is a decimal, an array (a maturity-dependent floor) or ``None`` (no
    floor). A floor never lifts a rate above its pre-shock value."""
    r0 = np.asarray(base_rates, dtype=float)
    s = np.asarray(shock_bp, dtype=float)
    if not (np.all(np.isfinite(r0)) and np.all(np.isfinite(s))):
        raise ValueError("base_rates and shock_bp must be finite")
    shocked = r0 + s / 1e4
    if floor is None:
        return shocked
    f = np.asarray(floor, dtype=float)
    if not np.all(np.isfinite(f)):
        raise ValueError("floor must be finite")
    if r0.shape != s.shape or (f.ndim > 0 and f.shape != r0.shape):
        raise ValueError("base_rates, shock_bp and floor must have the same shape")
    return np.maximum(shocked, np.minimum(f, r0))


def eu_floor(maturity_years: np.ndarray) -> np.ndarray:
    """The maturity-dependent post-shock floor of Commission Delegated Regulation (EU) 2024/856 Article 3(7): -0.015
    at immediate maturity, rising 0.0003 per year to 0 at 50 years (dossier 01 H.5)."""
    m = vec(maturity_years, "maturity_years", allow_empty=True)
    if np.any(m < 0.0):
        raise ValueError("maturity_years must be nonnegative")
    return np.minimum(0.0, -0.015 + 0.0003 * m)

__all__ = [
    "ShockSize",
    "apply_floor",
    "eu_floor",
    "shock_curve",
    "shock_scalars",
    "shock_size",
    "shock_table",
]
