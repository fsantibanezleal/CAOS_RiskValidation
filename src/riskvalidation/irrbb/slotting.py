"""The 19 time buckets, the slotting of repricing cash flows, prepayment and term-deposit multipliers, SMM and CPR,
the PSA curve (SRP31.96 to SRP31.126; dossier 01 H.4; dossier 05 D.4).

Contract: ``engine-irrbb-api`` 1.1.0, module slotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.irrbb.shocks import SHOCKS
from riskvalidation.lgd_ead._checks import same_length, scalar, vec

_MIDPOINTS = (
    0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25.0,
)
"""SRP31.96 Table 3, all midpoints (dossier 01 H.4; dossier 03 F.2)."""
_EDGES = (
    0.0, 1 / 360, 1 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,
)
"""Lower edges (exclusive), derived from the range labels of Table 3. The upper edge of the overnight bucket as 1/360
year is UNVERIFIED (the source gives the label and the midpoint 0.0028 only)."""
_LABELS = (
    "overnight", "O/N to 1M", "1M to 3M", "3M to 6M", "6M to 9M", "9M to 1Y", "1Y to 1.5Y", "1.5Y to 2Y", "2Y to 3Y",
    "3Y to 4Y", "4Y to 5Y", "5Y to 6Y", "6Y to 7Y", "7Y to 8Y", "8Y to 9Y", "9Y to 10Y", "10Y to 15Y", "15Y to 20Y",
    "over 20Y",
)
_GAMMA = dict(zip(SHOCKS, (0.8, 1.2, 0.8, 1.2, 0.8, 1.2), strict=True))
"""SRP31.119 Table 5: the multiplier of the prepayment rate by shock."""
_U = dict(zip(SHOCKS, (1.2, 0.8, 0.8, 1.2, 1.2, 0.8), strict=True))
"""SRP31.125 Table 6: the multiplier of the term-deposit early redemption rate by shock."""


@dataclass(frozen=True)
class BucketGrid:
    midpoints: np.ndarray    # (19,) years
    lower: np.ndarray        # (19,) years, exclusive
    upper: np.ndarray        # (19,) years, inclusive; the last is infinity
    labels: tuple[str, ...]


def bucket_grid() -> BucketGrid:
    """The 19 buckets of SRP31.96 Table 3 with their midpoints, edges (``(lower, upper]``) and labels."""
    lower = np.array(_EDGES, dtype=float)
    return BucketGrid(np.array(_MIDPOINTS, dtype=float), lower, np.r_[lower[1:], np.inf], _LABELS)


def bucket_of(time_years: float, grid: BucketGrid | None = None) -> int:
    """The bucket number 1 to 19 of a time: intervals ``(lower, upper]``, time 0 in bucket 1, beyond 20 years
    bucket 19."""
    t = scalar(time_years, "time_years")
    if t < 0.0:
        raise ValueError("time_years must be nonnegative")
    g = grid or bucket_grid()
    return int(np.searchsorted(g.upper, t, side="left")) + 1


def slot_cash_flows(times: np.ndarray, amounts: np.ndarray, grid: BucketGrid | None = None) -> np.ndarray:
    """(19,) net notional repricing cash flows by bucket: ``amounts`` (assets positive, liabilities negative) at
    ``times`` years. The sum of the result equals the sum of the amounts."""
    t, a = vec(times, "times", allow_empty=True), vec(amounts, "amounts", allow_empty=True)
    same_length(times=t, amounts=a)
    if np.any(t < 0.0):
        raise ValueError("times must be nonnegative")
    g = grid or bucket_grid()
    idx = np.searchsorted(g.upper, t, side="left")
    return np.bincount(idx, weights=a, minlength=g.upper.size).astype(float)


_NMD_CAPS = {
    "retail-transactional": (0.9, 5.0),
    "retail-non-transactional": (0.7, 4.5),
    "wholesale": (0.5, 4.0),
}
"""SRP31 Table 4 (dossier 01 H.4): the cap on the core share and on the average maturity of the core, in years."""
_NMD_ALIAS = {"retail-operational": "retail-transactional", "retail-non-operational": "retail-non-transactional"}
"""The RAN 21-13 Annex 1 names of the first two categories (dossier 03 F.2; the same numbers)."""


@dataclass(frozen=True)
class NmdSlotting:
    flows: np.ndarray                    # (19,) money magnitudes, the sign is the caller's
    requested_core_share: float
    core_share_cap: float
    core_share_used: float
    requested_core_maturity_years: float
    core_maturity_cap_years: float
    core_maturity_used_years: float
    cap_applied: bool                    # true when the request exceeded a cap and the used value was cut to it


def nmd_split(balance: float, category: str, core_share: float, core_maturity_years: float, enforce_caps: bool = True,
              grid: BucketGrid | None = None) -> NmdSlotting:
    """Slot a non-maturity deposit (SRP31.107 to SRP31.112, Table 4): the non-core part ``(1 - share) balance`` goes
    to bucket 1 and the core part to the two buckets whose midpoints bracket the core maturity, weighted so that the
    weighted average maturity is the core maturity (a maturity below the first midpoint or above the last goes to that
    bucket alone).

    With ``enforce_caps`` (the standardised treatment) the used share and maturity are ``min(request, cap)`` and
    ``cap_applied`` is true when either was cut; without it the request is used as stated, ``cap_applied`` is false
    and the caller owns the assumption beyond the cap. The result carries the request, the cap and the used value of
    both."""
    b = scalar(balance, "balance")
    if b < 0.0:
        raise ValueError("balance must be nonnegative")
    key = _NMD_ALIAS.get(category, category)
    if key not in _NMD_CAPS:
        raise ValueError(f"unknown NMD category {category!r}; expected one of {tuple(_NMD_CAPS) + tuple(_NMD_ALIAS)}")
    share, maturity = scalar(core_share, "core_share"), scalar(core_maturity_years, "core_maturity_years")
    if not 0.0 <= share <= 1.0:
        raise ValueError("core_share must lie in [0, 1]")
    if maturity < 0.0:
        raise ValueError("core_maturity_years must be nonnegative")
    cap_share, cap_maturity = _NMD_CAPS[key]
    used_share = min(share, cap_share) if enforce_caps else share
    used_maturity = min(maturity, cap_maturity) if enforce_caps else maturity
    applied = enforce_caps and (share > cap_share or maturity > cap_maturity)
    g = grid or bucket_grid()
    flows = np.zeros(g.midpoints.size)
    flows[0] += (1.0 - used_share) * b
    core = used_share * b
    lo = int(np.searchsorted(g.midpoints, used_maturity, side="right")) - 1
    if lo < 0:
        flows[0] += core
    elif lo >= g.midpoints.size - 1:
        flows[-1] += core
    else:
        w_lo = (g.midpoints[lo + 1] - used_maturity) / (g.midpoints[lo + 1] - g.midpoints[lo])
        flows[lo] += w_lo * core
        flows[lo + 1] += (1.0 - w_lo) * core
    return NmdSlotting(flows, share, cap_share, used_share, maturity, cap_maturity, used_maturity, bool(applied))


def _shock_name(shock: str) -> str:
    if shock not in SHOCKS:
        raise ValueError(f"unknown shock {shock!r}; expected one of {SHOCKS}")
    return shock


def _unit(x: float, name: str) -> float:
    v = scalar(x, name)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return v


def prepayment_cpr(shock: str, cpr0: float) -> float:
    """``min(1, gamma_shock cpr0)`` with gamma from SRP31.119 Table 5. The cash flows ``CF^p(k) = CF^s(k) + CPR
    N^p(k-1)`` of SRP31.121 are the caller's slotting; the definition of the outstanding notional ``N^p(k-1)`` is
    UNVERIFIED."""
    return min(1.0, _GAMMA[_shock_name(shock)] * _unit(cpr0, "cpr0"))


def term_deposit_redemption_rate(shock: str, tdrr0: float) -> float:
    """``min(1, u_shock tdrr0)`` with u from SRP31.125 Table 6. SRP31.126 puts the redeemed share ``TD x TDRR`` in
    bucket 1; reading the remainder as scaled by ``1 - TDRR`` in its contractual bucket is UNVERIFIED."""
    return min(1.0, _U[_shock_name(shock)] * _unit(tdrr0, "tdrr0"))


def smm_from_cpr(cpr: float) -> float:
    """``SMM = 1 - (1 - CPR)^(1/12)`` (Fuster, Lucca and Vickery 2022; dossier 05 D.4)."""
    c = _unit(cpr, "cpr")
    return 1.0 if c >= 1.0 else float(-np.expm1(np.log1p(-c) / 12.0))


def cpr_from_smm(smm: float) -> float:
    """``CPR = 1 - (1 - SMM)^12``."""
    s = _unit(smm, "smm")
    return float(-np.expm1(12.0 * np.log1p(-s))) if s < 1.0 else 1.0


def psa_cpr(month: int | np.ndarray, psa_percent: float = 100.0) -> float | np.ndarray:
    """``psa_percent / 100 min(0.002 month, 0.06)``: 100 percent PSA is 0.2 percent in month 1, rising 0.2 percent a
    month to 6 percent at month 30 (dossier 05 D.4; the SIFMA primary text was not retrieved, UNVERIFIED against it)."""
    m = np.asarray(month, dtype=float)
    if not np.all(np.isfinite(m)) or np.any(m < 1.0):
        raise ValueError("month must be at least 1")
    p = scalar(psa_percent, "psa_percent")
    if p < 0.0:
        raise ValueError("psa_percent must be nonnegative")
    out = p / 100.0 * np.minimum(0.002 * m, 0.06)
    return float(out) if out.ndim == 0 else out

__all__ = [
    "BucketGrid",
    "NmdSlotting",
    "bucket_grid",
    "bucket_of",
    "cpr_from_smm",
    "nmd_split",
    "prepayment_cpr",
    "psa_cpr",
    "slot_cash_flows",
    "smm_from_cpr",
    "term_deposit_redemption_rate",
]
