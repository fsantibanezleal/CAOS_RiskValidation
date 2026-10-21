"""The basic approach to CVA risk, BA-CVA (MAR50.9, MAR50.14 to MAR50.25, in force 2023-01-01, published 2020-07-08;
dossier 01 E.1).

Reduced version (MAR50.14 to MAR50.16), capital = DS K_reduced with DS = 0.65:

    K_reduced = sqrt((rho sum_c SCVA_c)^2 + (1 - rho^2) sum_c SCVA_c^2),  rho = 50 percent
    SCVA_c = (1 / alpha) RW_c sum_NS M_NS EAD_NS DF_NS,  alpha = 1.4
    DF_NS = (1 - exp(-0.05 M_NS)) / (0.05 M_NS)  (1 for IMM banks)

Full version (MAR50.20 to MAR50.25): K_full = beta K_reduced + (1 - beta) K_hedged, beta = 0.25, with

    K_hedged = sqrt((rho sum_c (SCVA_c - SNH_c) - IH)^2 + (1 - rho^2) sum_c (SCVA_c - SNH_c)^2 + sum_c HMA_c).

The counterparty risk weights are MAR50.16 Table 1. There is no verified official numeric example (research addendum
R8): the tests rest on the invariants of dossier 01 M.1 (rho = 1 gives the sum of the SCVA, hedges never take K_full
below beta K_reduced). Not implemented: the computation of SNH_c, IH and HMA_c from the hedge instruments (the caller
passes the terms), the MAR50.18 eligible-hedge rule, and SA-CVA (MAR50.27 onward). The engine computes and states no
compliance claim.
"""

from __future__ import annotations

import math

from riskvalidation.regulatory.standardised import RegulatoryParameter, _finite, _nonnegative

MAR50_IN_FORCE = "2023-01-01"
MATERIALITY_THRESHOLD_EUR = 1e11  # MAR50.9: EUR 100 billion
ALPHA = 1.4  # MAR50.15
RHO = 0.5  # MAR50.14
BETA = 0.25  # MAR50.20
DS = 0.65  # MAR50.14, MAR50.20
DISCOUNT_RATE = 0.05  # MAR50.15

#: MAR50.16 Table 1: (investment grade, high yield or not rated) risk weights by sector, as decimals
_TABLE_1 = {
    "sovereigns-central-banks-mdbs": (0.005, 0.02),
    "local-government-public": (0.01, 0.04),
    "financials": (0.05, 0.12),
    "basic-materials-energy-industrials": (0.03, 0.07),
    "consumer-transport-administrative": (0.03, 0.085),
    "technology-telecoms": (0.02, 0.055),
    "health-care-utilities-professional": (0.015, 0.05),
    "other": (0.05, 0.12),
}
SECTOR_RISK_WEIGHTS: dict[tuple[str, str], RegulatoryParameter] = {}
for _sector, (_ig, _hy) in _TABLE_1.items():
    SECTOR_RISK_WEIGHTS[(_sector, "risk_weight_investment_grade")] = RegulatoryParameter(
        _ig, "MAR50.16", "MAR50.16 Table 1", MAR50_IN_FORCE
    )
    SECTOR_RISK_WEIGHTS[(_sector, "risk_weight_high_yield_or_not_rated")] = RegulatoryParameter(
        _hy, "MAR50.16", "MAR50.16 Table 1", MAR50_IN_FORCE
    )
del _sector, _ig, _hy


def materiality_threshold_applies(aggregate_notional_eur: float) -> bool:
    """MAR50.9: true at or below EUR 100 billion of aggregate non-centrally-cleared derivative notional."""
    return _nonnegative("aggregate_notional_eur", aggregate_notional_eur) <= MATERIALITY_THRESHOLD_EUR


def sector_risk_weight(sector: str, investment_grade: bool) -> RegulatoryParameter:
    """MAR50.16 Table 1: the counterparty risk weight by sector and credit quality (investment grade or not)."""
    column = "risk_weight_investment_grade" if investment_grade else "risk_weight_high_yield_or_not_rated"
    if (sector, column) not in SECTOR_RISK_WEIGHTS:
        raise ValueError(f"sector {sector!r} is not a row of MAR50.16 Table 1 (MAR50.16)")
    return SECTOR_RISK_WEIGHTS[(sector, column)]


def discount_factor(m: float, imm_bank: bool = False) -> float:
    """MAR50.15: (1 - exp(-0.05 M)) / (0.05 M), the effective maturity M in years; 1 for a bank using the IMM."""
    maturity = _finite("m", m)
    if maturity <= 0.0:
        raise ValueError(f"m must be positive, got {m!r}")
    if imm_bank:
        return 1.0
    return -math.expm1(-DISCOUNT_RATE * maturity) / (DISCOUNT_RATE * maturity)  # expm1: no cancellation for a short M


def scva(
    risk_weight: float, netting_sets: list[tuple[float, float]], imm_bank: bool = False, alpha: float = ALPHA
) -> float:
    """MAR50.15: (1 / alpha) RW sum_NS M_NS EAD_NS DF_NS over the counterparty's netting sets given as (M in years,
    EAD)."""
    rw = _nonnegative("risk_weight", risk_weight)
    a = _finite("alpha", alpha)
    if a <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    total = 0.0
    for m, ead in netting_sets:
        maturity = _finite("netting_sets", m)
        if maturity <= 0.0:
            raise ValueError(f"netting_sets must hold a positive M in years, got {m!r}")
        total += maturity * _nonnegative("netting_sets", ead) * discount_factor(maturity, imm_bank)
    return rw * total / a


def _rho(rho: float) -> float:
    r = _finite("rho", rho)
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho!r}")
    return r


def k_reduced(scvas: list[float], rho: float = RHO) -> float:
    """MAR50.14: sqrt((rho sum SCVA)^2 + (1 - rho^2) sum SCVA^2); with rho = 1 the sum of the SCVA."""
    r = _rho(rho)
    x = [_nonnegative("scvas", v) for v in scvas]
    return math.sqrt((r * sum(x)) ** 2 + (1.0 - r * r) * sum(v * v for v in x))


def k_hedged(scva_minus_snh: list[float], ih: float, hma: list[float], rho: float = RHO) -> float:
    """MAR50.21 to MAR50.25: sqrt((rho sum (SCVA - SNH) - IH)^2 + (1 - rho^2) sum (SCVA - SNH)^2 + sum HMA)."""
    r = _rho(rho)
    x = [_finite("scva_minus_snh", v) for v in scva_minus_snh]
    index_hedge = _finite("ih", ih)
    misalignment = [_nonnegative("hma", v) for v in hma]
    if len(misalignment) != len(x):
        raise ValueError(
            f"hma has {len(misalignment)} entries for {len(x)} counterparties: one term per counterparty is required"
        )
    radicand = (r * sum(x) - index_hedge) ** 2 + (1.0 - r * r) * sum(v * v for v in x) + sum(misalignment)
    return math.sqrt(max(0.0, radicand))


def k_full(k_reduced_value: float, k_hedged_value: float, beta: float = BETA) -> float:
    """MAR50.20: beta K_reduced + (1 - beta) K_hedged."""
    b = _finite("beta", beta)
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta!r}")
    return b * _nonnegative("k_reduced_value", k_reduced_value) + (1.0 - b) * _nonnegative(
        "k_hedged_value", k_hedged_value
    )


def capital(k: float, ds: float = DS) -> float:
    """MAR50.14 and MAR50.20: DS x K with the discount scalar DS = 0.65."""
    return _nonnegative("ds", ds) * _nonnegative("k", k)

__all__ = [
    "capital",
    "discount_factor",
    "k_full",
    "k_hedged",
    "k_reduced",
    "materiality_threshold_applies",
    "scva",
    "sector_risk_weight",
]
