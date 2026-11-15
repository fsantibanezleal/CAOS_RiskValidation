"""The IRB LGD and EAD floors and the supervisory CCFs of dossier 01 section A.5 (Basel III final, in force 2023-01-01;
Basel II is not implemented), the CRR3 LGD input floors (Regulation (EU) 2024/1623, applying from 2025-01-01), the
partly secured floor of CRE32.17 and the resolution of the floor of one exposure for
`regulatory.irb.capital_requirement`. The Basel cells were confirmed against the paragraph text (CRE32.16, CRE32.58 to
CRE32.59) on 2026-11-11, the CRR3 cells against Articles 161(4) to (6) and 164(4) to (4a).
Contract `engine-lgd-ead-api` 1.0.0, module floors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskvalidation.lgd_ead._checks import same_length, scalar, vec

PORTFOLIOS = ("corporate", "retail", "public-sector")
COLLATERALS = ("unsecured", "financial", "receivables", "real-estate", "other-physical", "qrre", "mortgage")
FLOOR_REGIMES = ("basel3", "crr3")

# CRE32.16 (corporate; the floors do not apply to the sovereign asset class) and the CRE32.58 table (retail). The order
# of the retail secured cells (financial, receivables, real estate, other physical) is CONFIRMED against the CRE32.58
# table (the 2026-11-11 reading). CRE32.59 fixes the residential mortgage floor at 5% irrespective of the collateral.
LGD_FLOOR_CELLS: dict[tuple[str, str], float] = {
    ("corporate", "unsecured"): 0.25,       # CRE32.16
    ("corporate", "financial"): 0.00,       # CRE32.16
    ("corporate", "receivables"): 0.10,     # CRE32.16
    ("corporate", "real-estate"): 0.10,     # CRE32.16
    ("corporate", "other-physical"): 0.15,  # CRE32.16
    ("retail", "mortgage"): 0.05,           # CRE32.58 table, fixed by CRE32.59
    ("retail", "qrre"): 0.50,               # CRE32.58
    ("retail", "unsecured"): 0.30,          # CRE32.58
    ("retail", "financial"): 0.00,          # CRE32.58
    ("retail", "receivables"): 0.10,        # CRE32.58
    ("retail", "real-estate"): 0.10,        # CRE32.58
    ("retail", "other-physical"): 0.15,     # CRE32.58
}

# CRR3: Article 161(4) Table 1 (corporate exposure classes of Article 147(2)(c)(i) to (iii)), Article 161(5) (regional
# governments, local authorities and public sector entities, Article 147(2)(aa)(i) and (ii): 5% without eligible funded
# credit protection, no secured cell) and Article 164(4) Table 1 (retail; the residential mortgage 5% is fixed by
# Article 164(4a)(d)). Applies from 2025-01-01 (Article 2 of Regulation (EU) 2024/1623).
LGD_FLOOR_CELLS_CRR3: dict[tuple[str, str], float] = {
    ("corporate", "unsecured"): 0.25,       # Article 161(4)
    ("corporate", "financial"): 0.00,       # Article 161(4)
    ("corporate", "receivables"): 0.10,     # Article 161(4)
    ("corporate", "real-estate"): 0.10,     # Article 161(4)
    ("corporate", "other-physical"): 0.15,  # Article 161(4)
    ("public-sector", "unsecured"): 0.05,   # Article 161(5)
    ("retail", "mortgage"): 0.05,           # Article 164(4), fixed by Article 164(4a)(d)
    ("retail", "qrre"): 0.50,               # Article 164(4)
    ("retail", "unsecured"): 0.30,          # Article 164(4)
    ("retail", "financial"): 0.00,          # Article 164(4)
    ("retail", "receivables"): 0.10,        # Article 164(4)
    ("retail", "real-estate"): 0.10,        # Article 164(4)
    ("retail", "other-physical"): 0.15,     # Article 164(4)
}


def _crr3_source(cell: tuple[str, str]) -> str:
    if cell[0] == "corporate":
        return "CRR3 Article 161(4)"
    if cell[0] == "public-sector":
        return "CRR3 Article 161(5)"
    return "CRR3 Article 164(4a)(d)" if cell[1] == "mortgage" else "CRR3 Article 164(4)"


def _basel_source(cell: tuple[str, str]) -> str:
    if cell[0] == "corporate":
        return "CRE32.16"
    return "CRE32.59" if cell[1] == "mortgage" else "CRE32.58"


#: the paragraph of each cell, per regime
LGD_FLOOR_SOURCES: dict[str, dict[tuple[str, str], str]] = {
    "basel3": {cell: _basel_source(cell) for cell in LGD_FLOOR_CELLS},
    "crr3": {cell: _crr3_source(cell) for cell in LGD_FLOOR_CELLS_CRR3},
}
_CELLS = {"basel3": LGD_FLOOR_CELLS, "crr3": LGD_FLOOR_CELLS_CRR3}

# dossier 01 A.5 and B.2: the F-IRB CCFs equal the standardised ones (CRE32.33, CRE20.94 to CRE20.101)
SUPERVISORY_CCF: dict[str, float] = {
    "direct-credit-substitutes": 1.0,
    "nifs-rufs-transaction-related-contingents": 0.5,
    "commitments": 0.4,                                              # CRE20.98
    "short-term-self-liquidating-trade-letters-of-credit": 0.2,
    "unconditionally-cancellable": 0.1,                              # CRE20.100
}


def lgd_floor(portfolio: str, collateral: str, regime: str = "basel3") -> float:
    """The A-IRB LGD floor of a portfolio and a collateral type under a regime, ``basel3`` (CRE32.16, CRE32.58 to
    CRE32.59) or ``crr3`` (Articles 161(4), 161(5), 164(4)); sovereign exposures are exempt (CRE32.16) and are not
    passed here. A combination without a cell raises. The floor of a partly secured exposure is
    :func:`partly_secured_lgd_floor` of the unsecured and the secured cell."""
    if regime not in _CELLS:
        raise ValueError(f"regime must be one of {FLOOR_REGIMES}, got {regime!r}")
    if portfolio not in PORTFOLIOS:
        raise ValueError(f"portfolio must be one of {PORTFOLIOS}, got {portfolio!r}")
    if collateral not in COLLATERALS:
        raise ValueError(f"collateral must be one of {COLLATERALS}, got {collateral!r}")
    try:
        return _CELLS[regime][(portfolio, collateral)]
    except KeyError:
        raise ValueError(f"{regime} has no floor cell for {portfolio} / {collateral}") from None


def _denominator(e: Any, e_s: Any, h_e: Any) -> float:
    ev, esv, hv = scalar(e, "e"), scalar(e_s, "e_s"), scalar(h_e, "h_e")
    if ev <= 0.0:
        raise ValueError(f"e must be positive, got {e!r}")
    if esv < 0.0:
        raise ValueError(f"e_s must not be negative, got {e_s!r}")
    if hv < 0.0:
        raise ValueError(f"h_e must not be negative, got {h_e!r}")
    return ev * (1.0 + hv)


def _unit(x: Any, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [0, 1], got {x!r}") from None
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {x!r}")
    return v


def partly_secured_lgd_floor(unsecured_floor: float, secured_floor: float, e: float, e_s: float,
                             h_e: float = 0.0) -> float:
    r"""The LGD floor of a partly secured exposure (CRE32.17; CRR3 Article 161(6)):

    $\text{Floor}=\text{LGD}_{U}\,E_U/(E(1+H_E))+\text{LGD}_{S}\,E_S/(E(1+H_E))$ with $E_S$ capped at $E(1+H_E)$ and
    $E_U=E(1+H_E)-E_S$ (CRE32.10). ``e`` is the exposure value, ``e_s`` the collateral value after its haircuts, ``h_e``
    the exposure haircut. ``e_s = 0`` gives the unsecured floor; ``e_s`` at or above $E(1+H_E)$ gives the secured
    floor."""
    u, s_ = _unit(unsecured_floor, "unsecured floor"), _unit(secured_floor, "secured floor")
    share = min(float(e_s) / _denominator(e, e_s, h_e), 1.0)
    return u * (1.0 - share) + s_ * share


@dataclass(frozen=True)
class LGDFloorInput:
    """What the floor of one exposure needs beyond its asset class: the collateral type and the CRE32.10 terms.

    ``collateral`` is ``unsecured`` (the default), a collateral type of CRE32.16 (``financial``, ``receivables``,
    ``real-estate``, ``other-physical``) or ``mortgage``; ``e`` the exposure value, ``e_s`` the collateral value after
    the haircuts of CRE32.11 (capped at ``e (1 + h_e)``), ``h_e`` the exposure haircut. ``portfolio="public-sector"``
    selects the CRR3 Article 161(5) cell for an exposure to a regional government, a local authority or a public
    sector entity (the engine has no asset class of its own for them: pass ``corporate``)."""

    collateral: str = "unsecured"
    e: float = 1.0
    e_s: float = 0.0
    h_e: float = 0.0
    portfolio: str | None = None

    def __post_init__(self) -> None:
        if self.collateral not in COLLATERALS:
            raise ValueError(f"collateral must be one of {COLLATERALS}, got {self.collateral!r}")
        if self.portfolio not in (None, "public-sector"):
            raise ValueError(f"portfolio must be None or 'public-sector', got {self.portfolio!r}")
        _denominator(self.e, self.e_s, self.h_e)


@dataclass(frozen=True)
class LGDFloorApplied:
    """The floor of one exposure and where it came from. ``applied`` is False for an exposure the floors do not reach
    (a sovereign, a defaulted exposure): ``floor`` is then None, ``lgd_used`` equals ``lgd_input`` and ``reason`` says
    why."""

    applied: bool
    regime: str
    portfolio: str | None
    collateral: str | None
    lgd_input: float
    floor: float | None
    lgd_used: float
    binding: bool
    secured_share: float | None
    source: str
    reason: str


_CORPORATE_CLASSES = ("corporate", "sme", "hvcre")
_RETAIL_CLASSES = ("residential_mortgage", "qrre", "other_retail")
_DEFAULTED_SOURCE = {"basel3": "CRE31.7 and CRE31.20 (the capital requirement of a defaulted exposure)",
                     "crr3": "CRR3 Article 158(5) (the expected loss of a defaulted exposure is ELBE)"}
_SOVEREIGN_SOURCE = {"basel3": "CRE32.16 (the floors do not apply to the sovereign asset class)",
                     "crr3": "CRR3 Articles 161(4) and 164(4) (corporate and retail exposure classes only)"}


def _applied(regime: str, portfolio: str, collateral: str, lgd: float, floor: float, share: float | None,
             source: str) -> LGDFloorApplied:
    used = max(lgd, floor)
    return LGDFloorApplied(applied=True, regime=regime, portfolio=portfolio, collateral=collateral, lgd_input=lgd,
                           floor=floor, lgd_used=used, binding=used > lgd, secured_share=share, source=source,
                           reason="")


def _exempt(regime: str, lgd: float, source: str, reason: str) -> LGDFloorApplied:
    return LGDFloorApplied(applied=False, regime=regime, portfolio=None, collateral=None, lgd_input=lgd, floor=None,
                           lgd_used=lgd, binding=False, secured_share=None, source=source, reason=reason)


def resolve_lgd_floor(asset_class: str, lgd: float, regime: str, spec: LGDFloorInput,
                      defaulted: bool = False) -> LGDFloorApplied:
    """The floor that binds the own-estimate LGD of one exposure, and the LGD to use.

    Basel III final (``basel3``, CRE32.16, CRE32.17, CRE32.58, CRE32.59) or CRR3 (``crr3``, Articles 161(4) to (6) and
    164(4) to (4a)); any other regime raises. The floor applies to the LGD input of the risk-weight and expected-loss
    formulas of a non-defaulted, non-sovereign exposure: a sovereign and a defaulted exposure are returned unfloored
    with the source (whether the floor binds the default-time LGD is UNVERIFIED: CRE36.85 was not read). A residential
    mortgage takes 5% whatever the collateral and the cover. An asset class the research could not place (``bank``,
    ``financial_institution``) raises."""
    if regime not in _CELLS:
        raise ValueError(f"LGD floors exist for the regimes {FLOOR_REGIMES}, not for regime {regime!r}")
    lgd_v = scalar(lgd, "lgd")
    if not 0.0 <= lgd_v <= 1.0:
        raise ValueError("lgd must lie in [0, 1]")
    public = spec.portfolio == "public-sector"
    if public and regime != "crr3":
        raise ValueError("portfolio 'public-sector' (Article 161(5)) exists only under regime crr3")
    if asset_class == "sovereign":
        if public:
            raise ValueError("a public-sector exposure is not a sovereign one: pass asset_class 'corporate'")
        return _exempt(regime, lgd_v, _SOVEREIGN_SOURCE[regime], "sovereign exposure: the floors do not apply")
    if defaulted:
        return _exempt(regime, lgd_v, _DEFAULTED_SOURCE[regime],
                       "defaulted exposure: the floors bind the non-defaulted formula; whether they bind the "
                       "default-time LGD is UNVERIFIED, so none is applied")
    if asset_class in _RETAIL_CLASSES:
        if public:
            raise ValueError("portfolio 'public-sector' is not a retail exposure class")
        portfolio = "retail"
    elif asset_class in _CORPORATE_CLASSES:
        portfolio = "public-sector" if public else "corporate"
    else:
        raise ValueError(f"the LGD floor of asset class {asset_class!r} is UNVERIFIED (CRE32.16 and Article 161(4) say "
                         "'corporate'; how banks and financial institutions are classed was not read): not applied")
    if spec.collateral == "mortgage" and asset_class != "residential_mortgage":
        raise ValueError("collateral 'mortgage' belongs to asset class residential_mortgage")
    sources = LGD_FLOOR_SOURCES[regime]
    if asset_class == "residential_mortgage":  # CRE32.59, Article 164(4a)(d): 5% irrespective of the collateral
        return _applied(regime, portfolio, "mortgage", lgd_v, lgd_floor("retail", "mortgage", regime), None,
                        sources[("retail", "mortgage")])
    if asset_class == "qrre":
        if spec.collateral != "unsecured":
            raise ValueError(f"{regime} has no floor cell for qrre / {spec.collateral} (QRRE has no secured cell)")
        return _applied(regime, portfolio, "unsecured", lgd_v, lgd_floor("retail", "qrre", regime), None,
                        sources[("retail", "qrre")])
    if spec.collateral == "unsecured":
        if spec.e_s != 0.0:
            raise ValueError("e_s must be 0 for an unsecured exposure (give the collateral type with it)")
        return _applied(regime, portfolio, "unsecured", lgd_v, lgd_floor(portfolio, "unsecured", regime), None,
                        sources[(portfolio, "unsecured")])
    u = lgd_floor(portfolio, "unsecured", regime)
    sec = lgd_floor(portfolio, spec.collateral, regime)
    share = min(spec.e_s / _denominator(spec.e, spec.e_s, spec.h_e), 1.0)
    floor = partly_secured_lgd_floor(u, sec, spec.e, spec.e_s, spec.h_e)
    if regime == "basel3":
        src = "CRE32.16 and CRE32.17" if portfolio == "corporate" else "CRE32.17 and CRE32.59"
    else:
        src = "CRR3 Article 161(4) and 161(6)" if portfolio == "corporate" else "CRR3 Article 164(4) and 164(4a)(c)"
    return _applied(regime, portfolio, spec.collateral, lgd_v, floor, share, src)


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

__all__ = [
    "LGD_FLOOR_CELLS_CRR3",
    "LGD_FLOOR_SOURCES",
    "FloorBite",
    "LGDFloorApplied",
    "LGDFloorInput",
    "apply_ead_floor",
    "apply_lgd_floor",
    "ead_floor",
    "lgd_floor",
    "partly_secured_lgd_floor",
    "resolve_lgd_floor",
    "supervisory_ccf",
]
