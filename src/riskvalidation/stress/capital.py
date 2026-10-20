"""The capital engine of the macro stress test (dossier 05 H.1 layer 3) on a static balance sheet under the EBA rules.

Static balance sheet (decision E4; EBA Methodological Note 2025-01-20, addendum F6): paras 33 and 34 maturing assets and
liabilities are replaced with similar instruments, no workout or cure of S3 assets, no capital measures after the
reference date, the same business mix; para 90 S3 exposures are not replaced, new S3 exposures move into the stock of S3
keeping the total constant, no cures, charge-offs or write-offs within the three years; para 91 the initial residual
maturity is kept; para 92 a constant balance sheet for impairments; para 24 year-end starting figures and a three-year
horizon.

The three-year IAPE projection for the base, the stress without mitigation and the stress with mitigation, and the
deficit (dossier 03 l.197-206); the internal capital target `OI >= 8% + c_DSIB + c_P2 + c_CCoB + c_CCyB + c_d` of RWA
(03 l.152-169) with its headroom; the Chilean capital stack (03 G.4, l.911-920); the P2G proxy, a Contraste
approximation (03 l.1001-1005); the rows of the Annex 3 summary (03 l.232-241; addendum F7). The revenue side (decision
Q10): the headline CET1 path is loss-only depletion and a labelled row adds PPNR from a stated constant fee and cost
ratio. Money is one currency and never converted; shares are shares of RWA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

MIN_PE = 0.08  # the 8 percent minimum effective equity over RWA (03 l.152-169)
P2R_CAP = 0.04  # the cap of the art. 66 quinquies requirement over RWA (03 l.52, l.253-254)
CHILEAN_CET1_FLOOR = 0.045  # 4.5 percent CET1 (03 G.4, l.911-920)
CHILEAN_CONSERVATION = 0.025  # 2.5 percent conservation buffer, art. 66 bis (03 G.4)
CCYB_MAX = 0.025  # the largest countercyclical buffer the stack accepts (contract engine-stress-api)
# : the 34 rows of the CMF Annex 3 summary with the labels as printed (addendum F7; Circular 2.365 of 2025-07-08,
# Resolution 6801;
# : https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf, Annex 3 from pdf p.29). The ids 2.T2, 2.AT1
# and 2.CET1
#: are the contract's ids for the three unnumbered components of row 2.
ANNEX3_ROWS: tuple[tuple[str, str], ...] = (
    ("1", "Ratios de capital"),
    ("1.1", "PE / APR"),
    ("1.2", "T1 / APR"),
    ("1.3", "CET1 / APR"),
    ("2", "Patrimonio efectivo"),
    ("2.T2", "T2"),
    ("2.AT1", "AT1"),
    ("2.CET1", "CET1"),
    ("3", "APR"),
    ("4", "Cuantificacion del capital necesario"),
    ("4.1", "Riesgo de credito"),
    ("4.1.1", "Metodo estandar"),
    ("4.1.2", "Modelo interno"),
    ("4.2", "Riesgo de mercado (ME)"),
    ("4.3", "Riesgo operacional (ME)"),
    ("4.4", "Riesgo de concentracion"),
    ("4.4.1", "Individual"),
    ("4.4.2", "Sectorial"),
    ("4.4.3", "Otro tipo"),
    ("4.5", "RMLB"),
    ("4.6", "Riesgo estrategico"),
    ("4.7", "Riesgo reputacional"),
    ("4.8", "Otros riesgos materiales"),
    ("5", "OI"),
    ("6", "Cuantificacion del capital necesario por riesgos no cubiertos en pilar 1 (4.4+4.5+4.6+4.7+4.8)"),
    ("7", "Exigencia minima de PE, en periodo de planificacion (8%+7.1+7.2)"),
    ("7.1", "Exigencia adicional de PE por art.66 quater (banco sistemico)"),
    ("7.2", "Exigencia adicional de PE por art.66 quinquies"),
    (
        "8",
        "Cuantificacion del capital necesario por riesgos no cubiertos en pilar 1, "
        "y no exigidos por art.66 quinquies (6-7.2)",
    ),
    ("9", "Colchones de conservacion y contra ciclico, en periodo de planificacion (art.66 bis y 66 ter)"),
    ("10.1", "Impacto en indice de capital basico de las pruebas de tension severas, sin mitigacion"),
    ("10.2", "Impacto en indice de capital basico de las pruebas de tension severas, con mitigacion"),
    ("11", "Componente d) del OI"),
    ("12", "Holgura de indice de patrimonio efectivo respecto del OI (1.1 - 5)"),
)
#: the keys of `internal_capital` of `annex3_summary`, with the Annex 3 row each fills in column D
INTERNAL_CAPITAL_KEYS = {
    "credit": "4.1.2",
    "concentration_individual": "4.4.1",
    "concentration_sectoral": "4.4.2",
    "concentration_other": "4.4.3",
    "irrbb": "4.5",
    "strategic": "4.6",
    "reputational": "4.7",
    "other": "4.8",
}
PLAN_CASES = ("base", "stress-without-mitigation", "stress-with-mitigation")


def _num(x: float, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a number") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} is non-finite")
    return v


def _series(x: Sequence[float], name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional series, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _share(x: float, name: str) -> float:
    v = _num(x, name)
    if v < 0.0:
        raise ValueError(f"{name} must not be negative, got {v}")
    return v


# ----------------------------------------------------------------------------------------------------------------
# The headline path and the PPNR row


def cet1_path_loss_only(cet1_0: float, losses: Sequence[float]) -> np.ndarray:
    """Loss-only depletion (decision Q10): the CET1 at each year end is the CET1 at the jump-off less the cumulated
    losses of the
    years so far, with no revenue invented. `losses` (Y,) are not negative; returns (Y,)."""
    start = _num(cet1_0, "cet1_0")
    loss = _series(losses, "losses")
    if np.any(loss < 0.0):
        raise ValueError("losses must not be negative")
    return start - np.cumsum(loss)


@dataclass(frozen=True)
class PpnrRow:
    """A labelled PPNR row (never the headline); maps onto the document row `headline_cet1.ppnr.rows[*]`."""

    nii: float
    fee_income: float
    operating_cost: float
    ppnr: float


def ppnr_row(nii: float, fee_ratio: float, cost_ratio: float) -> PpnrRow:
    """`fee_income = fee_ratio * nii`, `operating_cost = cost_ratio * (nii + fee_income)`, `ppnr = nii + fee_income -
    operating_cost`. Both ratios are decimals in [0, 1] and named assumptions of the case (decision Q10); their
    definitions are the contract's proposal (the Federal Reserve's efficiency-ratio element of the 2026 PPNR suite, 05
    H.3, is the precedent for a
    cost ratio; the dossiers give no model for PPNR, 05 l.402)."""
    n = _num(nii, "nii")
    fr, cr = _num(fee_ratio, "fee_ratio"), _num(cost_ratio, "cost_ratio")
    for name, v in (("fee_ratio", fr), ("cost_ratio", cr)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"the ratio {name} must lie in [0, 1], got {v}")
    fee = fr * n
    cost = cr * (n + fee)
    return PpnrRow(nii=n, fee_income=fee, operating_cost=cost, ppnr=n + fee - cost)


# ----------------------------------------------------------------------------------------------------------------
# The internal capital target and the Chilean stack


@dataclass(frozen=True)
class InternalTarget:
    """The internal effective-equity target `OI`; shares of RWA; maps onto the block `internal_capital_target`."""

    min_pe: float  # 0.08
    c_dsib: float
    c_p2: float
    c_ccob: float
    c_ccyb: float
    c_d: float
    oi: float


def internal_capital_target(
    c_dsib: float, c_p2: float, c_ccob: float, c_ccyb: float, c_d: float, p2r: float = 0.0
) -> InternalTarget:
    """`OI = 8% + c_DSIB + c_P2 + c_CCoB + c_CCyB + c_d` of RWA (03 l.152-169). `p2r` is the art. 66 quinquies
    requirement, the floor of `c_p2` (`c_P2,bank >= c_66 quinquies`); `p2r` is bounded by the 4 percent cap of RWA (03
    l.52, l.253-254) and `c_p2` is the bank's own estimate, at least `p2r`, with no cap. `c_p2` and `c_d` are labelled
    assumptions of the case (decision Q4);
    `c_d` may be zero (03 B.2: "podria ser nula")."""
    parts = {
        "c_dsib": _share(c_dsib, "c_dsib"),
        "c_p2": _share(c_p2, "c_p2"),
        "c_ccob": _share(c_ccob, "c_ccob"),
        "c_ccyb": _share(c_ccyb, "c_ccyb"),
        "c_d": _share(c_d, "c_d"),
    }
    req = _share(p2r, "p2r")
    if req > P2R_CAP:
        raise ValueError(
            f"p2r = {req} is above the {P2R_CAP} cap of RWA on the art. 66 quinquies requirement (03 l.253)"
        )
    if parts["c_p2"] < req:
        raise ValueError(
            f"c_p2 = {parts['c_p2']} is below p2r = {req}: the bank's estimate includes the art. 66 quinquies "
            "requirement"
        )
    oi = MIN_PE + parts["c_dsib"] + parts["c_p2"] + parts["c_ccob"] + parts["c_ccyb"] + parts["c_d"]
    return InternalTarget(
        min_pe=MIN_PE,
        c_dsib=parts["c_dsib"],
        c_p2=parts["c_p2"],
        c_ccob=parts["c_ccob"],
        c_ccyb=parts["c_ccyb"],
        c_d=parts["c_d"],
        oi=oi,
    )


def chilean_cet1_stack(c_ccyb: float, c_dsib: float, p2_cet1: float) -> float:
    """The minimum CET1/RWA of the Chilean legal stack `4.5% + 2.5% + RCC + c_DSIB + the CET1 share of c_P2` (03 G.4,
    l.911-920; RAN 21-13): the failure point of the reverse stress (decision Q11). Shares of RWA; the countercyclical
    buffer is at most
    0.025."""
    ccyb, dsib, p2 = _share(c_ccyb, "c_ccyb"), _share(c_dsib, "c_dsib"), _share(p2_cet1, "p2_cet1")
    if ccyb > CCYB_MAX:
        raise ValueError(f"c_ccyb = {ccyb} is above {CCYB_MAX}")
    return CHILEAN_CET1_FLOOR + CHILEAN_CONSERVATION + ccyb + dsib + p2


# ----------------------------------------------------------------------------------------------------------------
# The three-year plan


@dataclass(frozen=True)
class PlanYear:
    """One year of a capital-plan case; maps onto the document rows `capital_plan.cases[*].years[*]`."""

    year: int  # 1 to 3
    rwa: float
    cet1: float
    at1: float
    t2: float
    pe: float  # cet1 + at1 + t2
    requirement_pct: float
    pe_pct: float
    headroom_pct: float  # pe_pct - requirement_pct
    deficit: float  # max(0, requirement_pct * rwa - pe)


def _years(rwa: float, cet1: np.ndarray, at1: float, t2: float, requirement: float) -> tuple[PlanYear, ...]:
    out = []
    for k, c1 in enumerate(cet1, start=1):
        pe = float(c1) + at1 + t2
        pct = pe / rwa
        out.append(
            PlanYear(
                year=k,
                rwa=rwa,
                cet1=float(c1),
                at1=at1,
                t2=t2,
                pe=pe,
                requirement_pct=requirement,
                pe_pct=pct,
                headroom_pct=pct - requirement,
                deficit=max(0.0, requirement * rwa - pe),
            )
        )
    return tuple(out)


def capital_plan(
    rwa: float,
    cet1_0: float,
    at1: float,
    t2: float,
    base_cet1: Sequence[float],
    stress_losses: Sequence[float],
    mitigation_effects: Sequence[float],
    oi: float,
    legal_minimum: float,
) -> dict[str, tuple[PlanYear, ...]]:
    """The three cases of the IAPE projection (03 l.197-206): `base` (CET1 from the caller's baseline, requirement
    `oi`), `stress-without-mitigation` (CET1 from `cet1_path_loss_only`, requirement `legal_minimum`) and `stress-with-
    mitigation` (the loss-only path plus the cumulated mitigation effects, requirement `legal_minimum`).

    Static balance sheet (para 33): `rwa` is the same every year, and AT1 and T2 stay at their jump-off values in every
    case (no capital measure after the reference date). `mitigation_effects` are the effect of the mitigation actions on
    CET1 of each year, not cumulated, and not negative (a mitigation action that lowered CET1 would break the rule that
    the mitigated PE is never
    below the unmitigated one)."""
    r = _num(rwa, "rwa")
    if r <= 0.0:
        raise ValueError("rwa must be positive")
    start = _num(cet1_0, "cet1_0")
    a1, t = _share(at1, "at1"), _share(t2, "t2")
    base = _series(base_cet1, "base_cet1")
    loss = _series(stress_losses, "stress_losses")
    mit = _series(mitigation_effects, "mitigation_effects")
    if not base.size == loss.size == mit.size:
        raise ValueError(
            f"base_cet1, stress_losses and mitigation_effects must have the same length, got {base.size}, "
            f"{loss.size} and {mit.size}"
        )
    if np.any(mit < 0.0):
        raise ValueError("mitigation_effects must not be negative")
    target, legal = _share(oi, "oi"), _share(legal_minimum, "legal_minimum")
    stressed = cet1_path_loss_only(start, loss)
    return {
        "base": _years(r, base, a1, t, target),
        "stress-without-mitigation": _years(r, stressed, a1, t, legal),
        "stress-with-mitigation": _years(r, stressed + np.cumsum(mit), a1, t, legal),
    }


def p2g_proxy(cet1_ratio_0: float, cet1_ratios_severe: Sequence[float], usable_buffers_and_mitigation: float) -> float:
    """`max(0, max_t [(CET1/RWA)_0 - (CET1/RWA)_t severe] - usable)` (03 l.1001-1005, FAQ 24: "the maximum deterioration
    inside the three-year projection horizon", referenced to CET1/RWA). A Contraste approximation: the CMF calibration
    grid is not public
    and the value is not a P2G requirement; the document block carries the label `contraste-approximation`."""
    start = _num(cet1_ratio_0, "cet1_ratio_0")
    severe = _series(cet1_ratios_severe, "cet1_ratios_severe")
    usable = _share(usable_buffers_and_mitigation, "usable_buffers_and_mitigation")
    return max(0.0, float(np.max(start - severe)) - usable)


# ----------------------------------------------------------------------------------------------------------------
# The Annex 3 summary

_NA = {"value": None, "status": "not-applicable"}
_RI = {"value": None, "status": "RI"}
_NTF = {"value": None, "status": "not-to-be-filled"}


def _v(x: float) -> dict:
    return {"value": float(x), "status": "value"}


def _need(mapping: Mapping, key: str, what: str) -> object:
    if key not in mapping:
        raise ValueError(f"{what} has no key {key!r}")
    return mapping[key]


def annex3_summary(
    capital: Mapping[str, float],
    target: InternalTarget,
    rwa_by_risk: Mapping[str, float],
    internal_capital: Mapping[str, float | None],
    plan: Mapping[str, tuple[PlanYear, ...]],
    mitigation_applies: bool,
) -> tuple[dict, ...]:
    """The 34 rows of the CMF Annex 3 summary (`ANNEX3_ROWS`), each `{"row_id", "label_cmf", "a", "b", "c", "d"}` with
    the four columns A (percent of RWA), B (amount), C (legal requirement) and D (internal measurement) as cells
    `{"value", "status"}` of status `value`, `RI` (an immaterial risk), `not-to-be-filled` (market and operational risk
    under the standardised method) or `not-applicable`.

    `capital` holds `cet1`, `at1` and `t2` in money (CLP millions in the product; the form states thousand pesos);
    `rwa_by_risk` holds `credit`, `market` and `operational`, and RWA is their sum; `internal_capital` holds one key per
    `INTERNAL_CAPITAL_KEYS` with the internal capital of the risk in money, or None for an immaterial risk (status
    `RI`); `plan` is the `capital_plan` result (the two stress cases give rows 10.1 and 10.2). Row 7.2 is the target's
    `c_p2` (the bank's own estimate, at least the art. 66 quinquies requirement): `InternalTarget` carries no separate
    requirement. Row 10.1 is the largest fall of CET1/RWA from the jump-off in the stress without mitigation, row 10.2
    the same with mitigation, `not-applicable` when `mitigation_applies` is
    false (the form: "N/A if the stress leads to no mitigation actions")."""
    cet1 = _share(_need(capital, "cet1", "capital"), "capital['cet1']")
    at1 = _share(_need(capital, "at1", "capital"), "capital['at1']")
    t2 = _share(_need(capital, "t2", "capital"), "capital['t2']")
    rwa_risk = {
        k: _share(_need(rwa_by_risk, k, "rwa_by_risk"), f"rwa_by_risk[{k!r}]")
        for k in ("credit", "market", "operational")
    }
    rwa = sum(rwa_risk.values())
    if rwa <= 0.0:
        raise ValueError("the RWA (the sum of rwa_by_risk) must be positive")
    internal: dict[str, float | None] = {}
    for key in INTERNAL_CAPITAL_KEYS:
        raw = _need(internal_capital, key, "internal_capital")
        internal[key] = None if raw is None else _share(raw, f"internal_capital[{key!r}]")
    plans = {}
    for case in (
        ("stress-without-mitigation", "stress-with-mitigation")
        if mitigation_applies
        else ("stress-without-mitigation",)
    ):
        years = _need(plan, case, "plan")
        if not years:
            raise ValueError(f"plan[{case!r}] has no years")
        plans[case] = years
    cet1_ratio = cet1 / rwa
    pe = cet1 + at1 + t2
    legal = {
        "credit": MIN_PE * rwa_risk["credit"],
        "market": MIN_PE * rwa_risk["market"],
        "operational": MIN_PE * rwa_risk["operational"],
    }

    def internal_cell(key: str) -> dict:
        return _RI if internal[key] is None else _v(internal[key])

    quantified = [v for k, v in internal.items() if k != "credit" and v is not None]
    row6 = float(sum(quantified))
    row7a = target.min_pe + target.c_dsib + target.c_p2
    cells: dict[str, tuple[dict, dict, dict, dict]] = {
        "1": (_NA, _NA, _NA, _NA),
        "1.1": (_v(pe / rwa), _v(pe), _NA, _NA),
        "1.2": (_v((cet1 + at1) / rwa), _v(cet1 + at1), _NA, _NA),
        "1.3": (_v(cet1_ratio), _v(cet1), _NA, _NA),
        "2": (_v(pe / rwa), _v(pe), _NA, _NA),
        "2.T2": (_v(t2 / rwa), _v(t2), _NA, _NA),
        "2.AT1": (_v(at1 / rwa), _v(at1), _NA, _NA),
        "2.CET1": (_v(cet1_ratio), _v(cet1), _NA, _NA),
        "3": (_v(1.0), _v(rwa), _NA, _NA),
        "4": (_NA, _NA, _v(MIN_PE * rwa), _NA),
        "4.1": (_NA, _NA, _v(legal["credit"]), _NA),
        "4.1.1": (_NA, _NA, _v(legal["credit"]), _NA),
        "4.1.2": (_NA, _NA, _NA, internal_cell("credit")),
        "4.2": (_NA, _NA, _v(legal["market"]), _NTF),
        "4.3": (_NA, _NA, _v(legal["operational"]), _NTF),
        "4.4": (_NA, _NA, _NA, _NA),
        "4.4.1": (_NA, _NA, _NA, internal_cell("concentration_individual")),
        "4.4.2": (_NA, _NA, _NA, internal_cell("concentration_sectoral")),
        "4.4.3": (_NA, _NA, _NA, internal_cell("concentration_other")),
        "4.5": (_NA, _NA, _NA, internal_cell("irrbb")),
        "4.6": (_NA, _NA, _NA, internal_cell("strategic")),
        "4.7": (_NA, _NA, _NA, internal_cell("reputational")),
        "4.8": (_NA, _NA, _NA, internal_cell("other")),
        "5": (_v(target.oi), _v(target.oi * rwa), _NA, _NA),
        "6": (_v(row6 / rwa), _v(row6), _NA, _NA),
        "7": (_v(row7a), _v(row7a * rwa), _NA, _NA),
        "7.1": (_v(target.c_dsib), _v(target.c_dsib * rwa), _NA, _NA),
        "7.2": (_v(target.c_p2), _v(target.c_p2 * rwa), _NA, _NA),
        "8": (_v(row6 / rwa - target.c_p2), _v(row6 - target.c_p2 * rwa), _NA, _NA),
        "9": (_v(target.c_ccob + target.c_ccyb), _v((target.c_ccob + target.c_ccyb) * rwa), _NA, _NA),
        "11": (_v(target.c_d), _v(target.c_d * rwa), _NA, _NA),
        "12": (_v(pe / rwa - target.oi), _v(pe - target.oi * rwa), _NA, _NA),
    }
    for row_id, case in (("10.1", "stress-without-mitigation"), ("10.2", "stress-with-mitigation")):
        if case not in plans:
            cells[row_id] = (_NA, _NA, _NA, _NA)
        else:
            fall = max(cet1_ratio - y.cet1 / y.rwa for y in plans[case])
            cells[row_id] = (_v(fall), _NA, _NA, _NA)
    return tuple(
        {"row_id": rid, "label_cmf": label, "a": a, "b": b, "c": c, "d": d}
        for (rid, label) in ANNEX3_ROWS
        for (a, b, c, d) in (cells[rid],)
    )


# ----------------------------------------------------------------------------------------------------------------
# The EBA static balance sheet


def eba_static_balance_sheet(stage_balances: np.ndarray, new_s3_flows: np.ndarray) -> np.ndarray:
    """The stage balances after each period under the para 90 rule: the new stage 3 exposures of a period move into the
    stock of stage 3 and reduce stages 1 and 2 pro rata to their balances, the total exposure is constant, nothing
    leaves stage 3 (no cures, charge-offs or write-offs) and nothing is replaced. `stage_balances` (3,) in stages 1, 2,
    3; `new_s3_flows` (H,); returns
    (H, 3). A flow larger than stages 1 plus 2 is refused."""
    s = np.array(stage_balances, dtype=float)
    f = _series(new_s3_flows, "new_s3_flows")
    if s.shape != (3,) or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError(f"stage_balances must be three non-negative balances (stages 1, 2, 3), got {s.tolist()}")
    if np.any(f < 0.0):
        raise ValueError("new_s3_flows must not be negative: nothing leaves stage 3")
    out = np.empty((f.size, 3))
    for h, flow in enumerate(f):
        s12 = s[0] + s[1]
        if flow > s12 + 1e-12 * max(1.0, s12):
            raise ValueError(f"new_s3_flows[{h}] = {flow} is larger than stages 1 plus 2 ({s12})")
        if flow > 0.0:
            share = min(1.0, flow / s12)
            s = np.array([s[0] - s[0] * share, s[1] - s[1] * share, s[2] + flow])
        out[h] = s
    return out
