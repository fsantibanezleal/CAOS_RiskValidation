"""The repricing gap, the duration gap, the standardised EVE and NII, and the RAN 21-13 Annex 1 variant (dossier 05 D.1
and D.2; dossier 01 H.4 and H.5; dossier 03 F.2).

A positive delta EVE or delta NII is a **loss** (SRP31.129). Discounting is continuously compounded. Money is a float
in the currency of the call; the engine never converts between currencies. Contract: ``engine-irrbb-api`` 1.0.0,
module eve_nii.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.irrbb.shocks import SHOCKS
from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec


def _buckets(**arrays: np.ndarray) -> None:
    same_length(**arrays)


# ---------------------------------------------------------------------------------------------------- gaps


def repricing_gap(asset_flows: np.ndarray, liability_flows: np.ndarray) -> np.ndarray:
    """``G_k = RSA_k - RSL_k`` (dossier 05 D.1): money magnitudes, both nonnegative."""
    a, liab = vec(asset_flows, "asset_flows"), vec(liability_flows, "liability_flows")
    _buckets(asset_flows=a, liability_flows=liab)
    if np.any(a < 0.0) or np.any(liab < 0.0):
        raise ValueError("asset_flows and liability_flows are magnitudes and must be nonnegative")
    return a - liab


def delta_nii_gap(gap: np.ndarray, midpoints: np.ndarray, shift_bp: float) -> float:
    """Loss-positive earnings-gap approximation on a constant balance sheet: ``-sum over t_k < 1 of G_k dr (1 - t_k)``
    with ``dr = shift_bp / 1e4`` (dossier 05 D.1, with the sign of the contract)."""
    g, t = vec(gap, "gap"), vec(midpoints, "midpoints")
    _buckets(gap=g, midpoints=t)
    dr = scalar(shift_bp, "shift_bp") / 1e4
    near = t < 1.0
    return float(-np.sum(g[near] * dr * (1.0 - t[near])))


@dataclass(frozen=True)
class DurationGap:
    asset_value: float
    liability_value: float
    asset_duration_years: float
    liability_duration_years: float
    duration_gap_years: float
    equity_change_100bp_up: float


def duration_gap(asset_flows: np.ndarray, liability_flows: np.ndarray, rates: np.ndarray,
                 midpoints: np.ndarray) -> DurationGap:
    """Present values at the continuously compounded rates; durations are the value-weighted times (with continuous
    compounding the modified duration equals the Macaulay duration); ``DGAP = D_A - (L / A) D_L`` and the first-order
    ``equity_change_100bp_up = -DGAP A 0.01`` (dossier 05 D.2). A book with no assets raises ``ValueError``."""
    a, liab = vec(asset_flows, "asset_flows"), vec(liability_flows, "liability_flows")
    r, t = vec(rates, "rates"), vec(midpoints, "midpoints")
    _buckets(asset_flows=a, liability_flows=liab, rates=r, midpoints=t)
    if np.any(a < 0.0) or np.any(liab < 0.0):
        raise ValueError("asset_flows and liability_flows are magnitudes and must be nonnegative")
    df = np.exp(-r * t)
    pv_a, pv_l = a * df, liab * df
    value_a, value_l = float(pv_a.sum()), float(pv_l.sum())
    if value_a <= 0.0:
        raise ValueError("the book has no assets: the duration gap is undefined")
    d_a = float(np.sum(t * pv_a) / value_a)
    d_l = float(np.sum(t * pv_l) / value_l) if value_l > 0.0 else 0.0
    dgap = d_a - (value_l / value_a) * d_l
    return DurationGap(value_a, value_l, d_a, d_l, dgap, -dgap * value_a * 0.01)


# ---------------------------------------------------------------------------------------------------- EVE and NII


def eve(cash_flows: np.ndarray, rates: np.ndarray, midpoints: np.ndarray) -> float:
    """``EVE = sum CF_k exp(-R_k t_k)`` (SRP31.129); assets positive, liabilities negative."""
    cf, r, t = vec(cash_flows, "cash_flows"), vec(rates, "rates"), vec(midpoints, "midpoints")
    _buckets(cash_flows=cf, rates=r, midpoints=t)
    return float(np.sum(cf * np.exp(-r * t)))


def delta_eve(cf_base: np.ndarray, cf_shock: np.ndarray, rates_base: np.ndarray, rates_shock: np.ndarray,
              midpoints: np.ndarray, kao: float = 0.0) -> float:
    """Loss-positive: ``sum CF0 DF0 - sum CFi DFi + KAO`` (SRP31.129; dossier 01 H.4)."""
    return eve(cf_base, rates_base, midpoints) - eve(cf_shock, rates_shock, midpoints) + scalar(kao, "kao")


def eve_risk_measure(delta_eve_by_shock_currency: np.ndarray) -> float:
    """``max over shocks of max(0, sum over currencies with delta > 0 of delta)``: (S, C) loss-positive money in one
    reporting currency, one row per shock (six in the order of ``SHOCKS``; any number of rows is accepted, the
    contract's golden has three) and one column per currency with material exposure (SRP31.129). A gain in one
    currency never offsets a
    loss in another."""
    m = mat(delta_eve_by_shock_currency, "delta_eve_by_shock_currency")
    return float(max(0.0, np.max(np.where(m > 0.0, m, 0.0).sum(axis=1))))


def delta_nii(cash_flows: np.ndarray, base_rates: np.ndarray, shocked_rates: np.ndarray,
              midpoints: np.ndarray) -> float:
    """Loss-positive: ``sum over t_k < 1 of CF_k (r0_k - ri_k)(1 - t_k)`` (RAN 21-13 Annex 1 Hoja 10; dossier 03 F.2;
    the same form serves SRP31.90's two parallel scenarios on a constant balance sheet)."""
    cf, r0, ri, t = (vec(cash_flows, "cash_flows"), vec(base_rates, "base_rates"), vec(shocked_rates, "shocked_rates"),
                     vec(midpoints, "midpoints"))
    _buckets(cash_flows=cf, base_rates=r0, shocked_rates=ri, midpoints=t)
    near = t < 1.0
    return float(np.sum(cf[near] * (r0[near] - ri[near]) * (1.0 - t[near])))


def automatic_option_addon(delta_fv_sold: np.ndarray, delta_fv_bought: np.ndarray) -> float:
    """``KAO = sum sold - sum bought`` (SRP31.127; dossier 01 H.4): changes in fair value under one shock."""
    sold = vec(delta_fv_sold, "delta_fv_sold", allow_empty=True)
    bought = vec(delta_fv_bought, "delta_fv_bought", allow_empty=True)
    return float(sold.sum() - bought.sum())


# ---------------------------------------------------------------------------------------------------- Annex 1


def cmf_pair(delta_clp: float, delta_clf: float, offset: bool = True, share: float = 0.5) -> float:
    """The CLP and CLF pair of RAN 21-13 Annex 1 (dossier 03 F.2), loss-positive. With ``offset`` and deltas of
    opposite signs (``delta_clp * delta_clf < 0``): if ``delta_clp <= 0`` the result is ``max(share delta_clp +
    delta_clf, 0)``, otherwise
    ``max(delta_clp + share delta_clf, 0)``, so the share applies to the gain leg only. Otherwise (same sign, either
    zero or no offset) the result is ``max(delta_clp, 0) + max(delta_clf, 0)``."""
    clp, clf, s = scalar(delta_clp, "delta_clp"), scalar(delta_clf, "delta_clf"), scalar(share, "share")
    if not 0.0 <= s <= 1.0:
        raise ValueError("share must lie in [0, 1]")
    if offset and clp * clf < 0.0:
        if clp <= 0.0:
            return max(s * clp + clf, 0.0)
        return max(clp + s * clf, 0.0)
    return max(clp, 0.0) + max(clf, 0.0)


@dataclass(frozen=True)
class Annex1Nii:
    pairs: np.ndarray            # (2,) used pair, parallel-up then parallel-down
    max_pair: float              # the maximum over the two shocks of the other currencies' positive parts plus the pair
    indexation_charge: float
    delta_nii: float
    worst_shock: str | None


@dataclass(frozen=True)
class Annex1Eve:
    pairs: np.ndarray            # (6,)
    delta_eve: float
    worst_shock: str | None


def _other(other: np.ndarray | None, rows: int) -> np.ndarray:
    if other is None:
        return np.zeros(rows)
    o = mat(other, "other_currencies")
    if o.shape[0] != rows:
        raise ValueError(f"other_currencies must have {rows} rows, got {o.shape[0]}")
    return np.where(o > 0.0, o, 0.0).sum(axis=1)


def _pairs(clp: np.ndarray, ur: np.ndarray, offset: bool, rows: int, what: str) -> np.ndarray:
    if clp.shape != (rows,) or ur.shape != (rows,):
        raise ValueError(f"{what} needs {rows} values per currency, got {clp.shape} and {ur.shape}")
    return np.array([cmf_pair(float(x), float(y), offset) for x, y in zip(clp, ur, strict=True)])


def annex1_delta_nii(nii_clp: np.ndarray, nii_ur: np.ndarray, pn_ur: float, offset: bool = True,
                     other_currencies: np.ndarray | None = None) -> Annex1Nii:
    """``delta_NII = max over the two parallel shocks of {sum over other currencies of max(delta, 0) + pair} + 0.025
    |PN_UR|`` (RAN 21-13 Annex 1 Hoja 10 to 12; dossier 03 F.2). The charge is added once, after the maximum; the
    worst shock is None
    when the maximum is zero."""
    clp, ur = vec(nii_clp, "nii_clp"), vec(nii_ur, "nii_ur")
    pairs = _pairs(clp, ur, offset, 2, "annex1_delta_nii")
    totals = _other(other_currencies, 2) + pairs
    top = float(totals.max())
    charge = indexation_charge(pn_ur)
    worst = None if top <= 0.0 else SHOCKS[int(np.argmax(totals))]  # parallel-up, parallel-down
    return Annex1Nii(pairs, top, charge, top + charge, worst)


def annex1_delta_eve(eve_clp: np.ndarray, eve_ur: np.ndarray, offset: bool = True,
                     other_currencies: np.ndarray | None = None) -> Annex1Eve:
    """``delta_EVE = max over the six shocks of {sum over other currencies of max(delta, 0) + pair}`` with the same
    offset rule (Annex 1 Hoja 12); there is no indexation charge in the EVE measure."""
    clp, ur = vec(eve_clp, "eve_clp"), vec(eve_ur, "eve_ur")
    pairs = _pairs(clp, ur, offset, len(SHOCKS), "annex1_delta_eve")
    totals = _other(other_currencies, len(SHOCKS)) + pairs
    top = float(totals.max())
    return Annex1Eve(pairs, top, None if top <= 0.0 else SHOCKS[int(np.argmax(totals))])


def indexation_charge(net_uf_position: float, rate: float = 0.025) -> float:
    """``rate |PN_UR|``: the standardised indexation (reajustabilidad) charge, an implied 2.5 percent inflation
    surprise (RAN 21-13 Annex 1; dossier 03 F.2). A CPI surprise scenario is a different quantity."""
    r = scalar(rate, "rate")
    if r < 0.0:
        raise ValueError("rate must be nonnegative")
    return r * abs(scalar(net_uf_position, "net_uf_position"))


# ---------------------------------------------------------------------------------------------------- outlier tests


@dataclass(frozen=True)
class OutlierResult:
    ratio: float
    threshold: float
    breached: bool


def outlier_test(measure: float, tier1: float, threshold: float = 0.15) -> OutlierResult:
    """``measure / tier1`` against the threshold (15 percent of Tier 1, SRP31.82 to SRP31.83; dossier 01 H.5);
    breached only when the ratio is strictly greater. A nonpositive measure never breaches."""
    m, t1, th = scalar(measure, "measure"), scalar(tier1, "tier1"), scalar(threshold, "threshold")
    if t1 <= 0.0:
        raise ValueError("tier1 must be positive")
    ratio = m / t1
    return OutlierResult(ratio, th, bool(ratio > th and m > 0.0))


@dataclass(frozen=True)
class PrioritisedBank:
    tests: tuple[dict, ...]      # id, measure, base, ratio, threshold, breached
    prioritised: bool


def prioritised_bank(delta_nii: float, delta_eve: float, tier1: float, margin_12m: float) -> PrioritisedBank:
    """The three tests of RAN 21-13 Annex 1 (dossier 03 F.2), strict inequalities: ``nii-tier1`` (delta NII over Tier
    1 above 5 percent), ``eve-tier1`` (delta EVE over Tier 1 above 15 percent), ``nii-margin`` (delta NII over the
    12-month cumulative net interest and indexation margin above 18 percent). ``prioritised`` is true when any is
    breached."""
    nii, ev = scalar(delta_nii, "delta_nii"), scalar(delta_eve, "delta_eve")
    t1, margin = scalar(tier1, "tier1"), scalar(margin_12m, "margin_12m")
    if t1 <= 0.0 or margin <= 0.0:
        raise ValueError("tier1 and margin_12m must be positive")
    spec = (("nii-tier1", nii, t1, 0.05), ("eve-tier1", ev, t1, 0.15), ("nii-margin", nii, margin, 0.18))
    tests = tuple({"id": i, "measure": m, "base": b, "ratio": m / b, "threshold": th, "breached": bool(m / b > th)}
                  for i, m, b, th in spec)
    return PrioritisedBank(tests, any(t["breached"] for t in tests))
