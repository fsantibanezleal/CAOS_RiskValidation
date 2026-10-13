"""The IRB LGD and EAD floors and the supervisory CCFs of dossier 01 section A.5 (Basel III final, in force 2023-01-01;
Basel II is not implemented). Every cell is transcribed from the dossier table; the paragraph text itself was not
re-read.
Contract `engine-lgd-ead-api` 1.0.0, module floors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.lgd_ead._checks import same_length, scalar, vec

PORTFOLIOS = ("corporate", "retail")
COLLATERALS = ("unsecured", "financial", "receivables", "real-estate", "other-physical", "qrre", "mortgage")

# dossier 01 A.5: "A-IRB corporate LGD floors" (CRE32.16) and "Retail LGD floors" (CRE32.58 to CRE32.59). The order
# of the retail secured cells (financial, receivables, real estate, other physical) is read from the corporate row:
# UNVERIFIED. The dossier cites CRE32.58 to CRE32.59 jointly, so the split of the seven retail cells between the two
# paragraphs is not asserted.
LGD_FLOOR_CELLS: dict[tuple[str, str], float] = {
    ("corporate", "unsecured"): 0.25,       # CRE32.16
    ("corporate", "financial"): 0.00,       # CRE32.16
    ("corporate", "receivables"): 0.10,     # CRE32.16
    ("corporate", "real-estate"): 0.10,     # CRE32.16
    ("corporate", "other-physical"): 0.15,  # CRE32.16
    ("retail", "mortgage"): 0.05,           # CRE32.58 to CRE32.59
    ("retail", "qrre"): 0.50,               # CRE32.58 to CRE32.59
    ("retail", "unsecured"): 0.30,          # CRE32.58 to CRE32.59
    ("retail", "financial"): 0.00,          # CRE32.58 to CRE32.59
    ("retail", "receivables"): 0.10,        # CRE32.58 to CRE32.59
    ("retail", "real-estate"): 0.10,        # CRE32.58 to CRE32.59
    ("retail", "other-physical"): 0.15,     # CRE32.58 to CRE32.59
}

# dossier 01 A.5 and B.2: the F-IRB CCFs equal the standardised ones (CRE32.33, CRE20.94 to CRE20.101)
SUPERVISORY_CCF: dict[str, float] = {
    "direct-credit-substitutes": 1.0,
    "nifs-rufs-transaction-related-contingents": 0.5,
    "commitments": 0.4,                                              # CRE20.98
    "short-term-self-liquidating-trade-letters-of-credit": 0.2,
    "unconditionally-cancellable": 0.1,                              # CRE20.100
}


def lgd_floor(portfolio: str, collateral: str) -> float:
    """The A-IRB LGD floor of a portfolio and a collateral type (sovereign exposures are exempt, CRE32.16, and are not
    passed here). A combination without a cell raises."""
    if portfolio not in PORTFOLIOS:
        raise ValueError(f"portfolio must be one of {PORTFOLIOS}, got {portfolio!r}")
    if collateral not in COLLATERALS:
        raise ValueError(f"collateral must be one of {COLLATERALS}, got {collateral!r}")
    try:
        return LGD_FLOOR_CELLS[(portfolio, collateral)]
    except KeyError:
        raise ValueError(f"dossier 01 A.5 has no floor cell for {portfolio} / {collateral}") from None


@dataclass(frozen=True)
class FloorBite:
    applied: np.ndarray      # (A,) the estimates after the floor
    binding: np.ndarray      # (A,) bool, the floor raised the estimate
    share_binding: float
    mean_before: float
    mean_after: float
    uplift: float            # mean_after - mean_before, never negative


def _bite(before: np.ndarray, applied: np.ndarray) -> FloorBite:
    mb, ma = float(before.mean()), float(applied.mean())
    return FloorBite(applied=applied, binding=applied > before, share_binding=float((applied > before).mean()),
                     mean_before=mb, mean_after=ma, uplift=max(0.0, ma - mb))


def apply_lgd_floor(lgd: np.ndarray, floor: float) -> FloorBite:
    """applied = max(lgd, floor); a floor of 0 binds nowhere."""
    x = vec(lgd, "lgd")
    f = scalar(floor, "floor")
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"floor must lie in [0, 1], got {floor!r}")
    return _bite(x, np.maximum(x, f))


def supervisory_ccf(item: str) -> float:
    """The F-IRB / standardised CCF of an off-balance sheet item (CRE32.33; CRE20.94 to CRE20.101)."""
    try:
        return SUPERVISORY_CCF[item]
    except KeyError:
        raise ValueError(f"item must be one of {tuple(SUPERVISORY_CCF)}, got {item!r}") from None


def ead_floor(drawn: np.ndarray, limit: np.ndarray, sa_ccf: np.ndarray, off_balance_share: float = 0.5) -> np.ndarray:
    """The EAD floor, 'on-balance amount + 50% of off-balance exposure using the SA CCF' (CRE32.36 corporate, CRE32.64
    retail; dossier 01 A.5): drawn + off_balance_share sa_ccf (limit - drawn). The reading that the 50% applies to the
    SA-CCF-weighted off-balance exposure follows the dossier wording and is UNVERIFIED against the paragraph text.

    The off-balance amount is max(limit - drawn, 0): an overdrawn line (drawn above limit) has no off-balance amount and
    its floor is the drawn amount."""
    d, lim, c = vec(drawn, "drawn"), vec(limit, "limit"), vec(sa_ccf, "sa_ccf")
    same_length(drawn=d, limit=lim, sa_ccf=c)
    s = scalar(off_balance_share, "off_balance_share")
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"off_balance_share must lie in [0, 1], got {off_balance_share!r}")
    if np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("sa_ccf must lie in [0, 1]")
    return d + s * c * np.maximum(lim - d, 0.0)


def apply_ead_floor(ead: np.ndarray, floor: np.ndarray) -> FloorBite:
    """applied = max(ead, floor) elementwise."""
    e, f = vec(ead, "ead"), vec(floor, "floor")
    same_length(ead=e, floor=f)
    return _bite(e, np.maximum(e, f))
