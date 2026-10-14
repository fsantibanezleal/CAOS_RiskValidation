"""The Basel III standardised approach of OPE25 (dossier 01 section F; OPE25 in force 2023-01-01, published 2024-07-05;
the version in force 2027-01-01 only rewords OPE25.10 and OPE25.27(5), formulas unchanged).

``ORC = BIC x ILM`` and ``RWA_op = 12.5 x ORC`` (OPE25.2); ``BI = ILDC + SC + FC`` (OPE25.4);
``ILDC = min(avg |II - IE|, 2.25% x avg IEA) + avg DI``, ``SC = max(avg OOI, avg OOE) + max(avg FI, avg FE)``,
``FC = avg |Net P&L_TB| + avg |Net P&L_BB|``, a bar being the three-year average over ``t, t-1, t-2`` with the absolute
values of net items taken year by year before averaging (OPE25.5 and footnote 1); the BIC with the marginal coefficients
12 percent to EUR 1bn, 15 percent from 1bn to 30bn and 18 percent above (OPE25.7 Table 1); ``ILM = ln(exp(1) - 1 +
(LC / BIC)^0.8)`` with ``LC = 15 x`` the average annual operational risk losses over the previous 10 years (OPE25.8);
ILM = 1 when LC = BIC and ILM > 1 when LC > BIC (OPE25.9). Loss data rules and discretions: 10 years of high-quality
loss data, a transitional minimum of 5 years, and under 5 years the capital is the BIC only (OPE25.10, OPE25.15); losses
net of recoveries (FAQ1 to OPE25.10); bucket 1 banks have ILM = 1 unless national discretion allows inclusion, and
supervisors may set ILM = 1 for all banks of the jurisdiction (OPE25.11); collection threshold EUR 20,000, national
discretion EUR 100,000 for buckets 2 and 3 (OPE25.18).

The engine computes and states no compliance claim; disclosures stay the caller's. Money is in the caller's currency
(the thresholds of the buckets are in EUR). Contract: ``engine-oprisk-api`` 1.0.0, module sma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from riskvalidation.oprisk._checks import scalar, vec

BUCKET_1_LIMIT = 1e9
BUCKET_2_LIMIT = 30e9
COEFFICIENTS = (0.12, 0.15, 0.18)
ILDC_CAP = 0.0225
LC_MULTIPLE = 15.0
ILM_EXPONENT = 0.8
RWA_FACTOR = 12.5
OPE25_MINIMUM_YEARS = 5


def _three(x: np.ndarray, name: str) -> np.ndarray:
    a = vec(x, name)
    if a.size != 3:
        raise ValueError(f"{name} must hold the 3 years t, t-1, t-2, got {a.size}")
    return a


def ildc(ii: np.ndarray, ie: np.ndarray, iea: np.ndarray, di: np.ndarray) -> float:
    """``min(mean(|ii - ie|), 0.0225 mean(iea)) + mean(di)`` (OPE25.5), the absolute value taken year by year."""
    ii, ie, iea, di = _three(ii, "ii"), _three(ie, "ie"), _three(iea, "iea"), _three(di, "di")
    return float(min(np.mean(np.abs(ii - ie)), ILDC_CAP * np.mean(iea)) + np.mean(di))


def service_component(ooi: np.ndarray, ooe: np.ndarray, fi: np.ndarray, fe: np.ndarray) -> float:
    """``max(mean(ooi), mean(ooe)) + max(mean(fi), mean(fe))`` (OPE25.5)."""
    ooi, ooe, fi, fe = _three(ooi, "ooi"), _three(ooe, "ooe"), _three(fi, "fi"), _three(fe, "fe")
    return float(max(np.mean(ooi), np.mean(ooe)) + max(np.mean(fi), np.mean(fe)))


def financial_component(pl_tb: np.ndarray, pl_bb: np.ndarray) -> float:
    """``mean(|pl_tb|) + mean(|pl_bb|)`` with the absolute value taken year by year (OPE25.5 footnote 1)."""
    return float(np.mean(np.abs(_three(pl_tb, "pl_tb"))) + np.mean(np.abs(_three(pl_bb, "pl_bb"))))


def business_indicator(ildc_value: float, sc_value: float, fc_value: float) -> float:
    """``BI = ILDC + SC + FC`` (OPE25.4); a negative component raises."""
    parts = (scalar(ildc_value, "ildc_value"), scalar(sc_value, "sc_value"), scalar(fc_value, "fc_value"))
    if min(parts) < 0.0:
        raise ValueError("the components of the business indicator must not be negative")
    return float(sum(parts))


def _bi(bi: float) -> float:
    v = scalar(bi, "bi")
    if v < 0.0:
        raise ValueError(f"bi must not be negative, got {v}")
    return v


def bic_bucket(bi: float) -> int:
    """1 for ``bi <= 1e9``, 2 for ``1e9 < bi <= 30e9``, 3 above (OPE25.7 Table 1)."""
    v = _bi(bi)
    return 1 if v <= BUCKET_1_LIMIT else 2 if v <= BUCKET_2_LIMIT else 3


def bic(bi: float) -> float:
    """``0.12 min(bi, 1e9) + 0.15 (min(bi, 30e9) - 1e9)^+ + 0.18 (bi - 30e9)^+`` (OPE25.7 Table 1)."""
    v = _bi(bi)
    c1, c2, c3 = COEFFICIENTS
    middle = max(min(v, BUCKET_2_LIMIT) - BUCKET_1_LIMIT, 0.0)
    return c1 * min(v, BUCKET_1_LIMIT) + c2 * middle + c3 * max(v - BUCKET_2_LIMIT, 0.0)


def loss_component(annual_losses: np.ndarray, window_years: int = 10) -> float:
    """``LC = 15 x`` the mean of the most recent ``window_years`` annual losses (net of recoveries, oldest first;
    OPE25.8).
    The minimum-years rule is applied by ``sma_capital``, not here."""
    x = vec(annual_losses, "annual_losses")
    if int(window_years) != window_years or window_years < 1:
        raise ValueError(f"window_years must be a positive integer, got {window_years}")
    if np.any(x < 0.0):
        raise ValueError("annual losses must not be negative")
    return float(LC_MULTIPLE * np.mean(x[-int(window_years):]))


def ilm(loss_component_value: float, bic_value: float) -> float:
    """``ILM = ln(e - 1 + (LC / BIC)^0.8)`` (OPE25.8); exactly 1 when ``LC = BIC`` (OPE25.9)."""
    lc, b = scalar(loss_component_value, "loss_component_value"), scalar(bic_value, "bic_value")
    if b <= 0.0:
        raise ValueError("bic_value must be positive")
    if lc < 0.0:
        raise ValueError("loss_component_value must not be negative")
    return math.log(math.e - 1.0 + (lc / b) ** ILM_EXPONENT)


@dataclass(frozen=True)
class SmaResult:
    """The standardised approach for one bank: money fields in EUR; ``basis`` is ``"bic-times-ilm"`` or ``"bic-only"``;
    ``discretion_applied`` is ``"none"``, ``"bucket-1-ilm-one"``, ``"national-ilm-one"`` or ``"transitional-bic-only"``;
    ``rwa = 12.5 x orc``. ``basis == "bic-only"`` exactly when ``loss_component`` and ``ilm`` are None, ``reason`` is a
    string and the discretion is ``"transitional-bic-only"``."""

    bi: float
    bic: float
    bucket: int
    years_used: int
    basis: str
    loss_component: float | None
    ilm: float | None
    discretion_applied: str
    orc: float
    rwa: float
    reason: str | None


def sma_capital(bi: float, annual_losses: np.ndarray, bucket1_uses_losses: bool = False, national_ilm_one: bool = False,
                minimum_years: int = 5, window_years: int = 10) -> SmaResult:
    """Operational risk capital of OPE25: ``ORC = BIC x ILM``, ``RWA = 12.5 ORC``.

    Order of the rules: (1) fewer than ``minimum_years`` years of loss data: the capital is the BIC only (OPE25.10,
    OPE25.15), no LC and no ILM, a reason, discretion ``transitional-bic-only`` (never an ILM on too little data);
    (2) ``national_ilm_one``: ILM = 1, discretion ``national-ilm-one`` (OPE25.11; disclosure stays the caller's);
    (3) bucket 1 and not ``bucket1_uses_losses``: ILM = 1, discretion ``bucket-1-ilm-one`` (OPE25.11); (4) otherwise
    ``ILM = ilm(LC, BIC)``. In (2) and (3) the loss component is still computed and reported. ``minimum_years`` below 5
    raises (5 years is the transitional minimum of OPE25.15; a national discretion would be a different, explicit
    function).
    """
    b = _bi(bi)
    x = vec(annual_losses, "annual_losses", allow_empty=True)
    if int(minimum_years) != minimum_years or minimum_years < OPE25_MINIMUM_YEARS:
        raise ValueError(f"minimum_years must be an integer of at least {OPE25_MINIMUM_YEARS} (the transitional "
                         f"minimum of OPE25.15), got {minimum_years}: an ILM on fewer years is unreachable through "
                         "this function")
    bic_value = bic(b)
    bucket = bic_bucket(b)
    years = int(x.size)
    if years < minimum_years:
        reason = (f"{years} years of loss data, fewer than the {int(minimum_years)} required (OPE25.10, OPE25.15): "
                  "the capital is the business indicator component only")
        return SmaResult(b, bic_value, bucket, years, "bic-only", None, None, "transitional-bic-only", bic_value,
                         RWA_FACTOR * bic_value, reason)
    lc = loss_component(x, window_years)
    used = min(years, int(window_years))
    if national_ilm_one:
        value, discretion = 1.0, "national-ilm-one"
    elif bucket == 1 and not bucket1_uses_losses:
        value, discretion = 1.0, "bucket-1-ilm-one"
    else:
        value, discretion = ilm(lc, bic_value), "none"
    orc = bic_value * value
    return SmaResult(b, bic_value, bucket, used, "bic-times-ilm", lc, value, discretion, orc, RWA_FACTOR * orc, None)
