r"""CMF CNCB Chapter B-1 provisions (Chile): the standard methods for group portfolios, the individual-assessment
categories, the default portfolio, the 0.50% minimum and the additional provisions.

Source: Contraste dossier 03 sections D.2 to D.6, read from the CMF "Compendio de Normas Contables para Bancos"
(CNCB, https://www.cmfchile.cl/portal/normativa/624/articles-29177_doc_pdf.pdf, read 2026-10-04; base Circular 2.243 of
2019-12-20, later circulars to 2.346 of 2024-03-06). Every table below is transcribed cell by cell from that
dossier, with its source paragraph in a comment. Parameters are decimals: 1.0916 per cent is 0.010916. The B-1
provision factor is PE = PI x PDI (B-1 N 3.1).

Chilean loan provisions in local statutory accounts are B-1, not IFRS 9 impairment: CNCB Chapter A-2 N 5 does not
apply IFRS 9 Chapter 5.5 to loans and contingent credits (dossier 03 D.1, dossier 02 F.5). This module keeps the two
regimes apart and computes no expected credit loss in the IFRS 9 sense. The functions compute; they state no
compliance claim. The CMF reform proposal of August 2026 (dossier 03 D.7: recalibrated PI scale, internal methods,
collateral as LGD) is not in force and is not implemented here.

The additional provisions (B-1 N 9) are a separate line (:func:`b1_additional_provisions`); no function of this module
adds them to a comparable B-1 total, and the 0.50% minimum applies to the individually assessed normal portfolio
only (:func:`b1_minimum_normal_check`).
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------------------------------
# B-1 N 2.1 "Hoja 3": individual assessment, Normal (A1 to A6) and Subestandar (B1 to B4). (PI, PDI, PE) decimals.
# Dossier 03 D.3.
_INDIVIDUAL: dict[str, tuple[float, float, float]] = {
    "A1": (0.0004, 0.9, 0.00036),
    "A2": (0.001, 0.825, 0.000825),
    "A3": (0.0025, 0.875, 0.0021875),
    "A4": (0.02, 0.875, 0.0175),
    "A5": (0.0475, 0.9, 0.04275),
    "A6": (0.1, 0.9, 0.09),
    "B1": (0.15, 0.925, 0.13875),
    "B2": (0.22, 0.925, 0.2035),
    "B3": (0.33, 0.975, 0.32175),
    "B4": (0.45, 0.975, 0.43875),
}

# B-1 N 2.2 "Hoja 8": default portfolio. (inclusive upper bound of TPE, PP); C6 has no upper bound. Dossier 03 D.3.
_DEFAULT_BUCKETS: tuple[tuple[str, float | None, float], ...] = (
    ("C1", 0.03, 0.02),
    ("C2", 0.20, 0.10),
    ("C3", 0.30, 0.25),
    ("C4", 0.50, 0.40),
    ("C5", 0.80, 0.65),
    ("C6", None, 0.90),
)

_DPD = ("0", "1-29", "30-59", "60-89", "default")

# B-1 N 3.1.1, Circular 2.346 "Hoja 11": residential mortgages (and residential leasing), dossier 03 D.5.
# PVG buckets <= 40%, <= 80%, <= 90%, > 90%; columns are _DPD. PE is the transcribed cell (it equals PI x PDI within the
# table's own rounding of 5e-7 as a decimal).
_MORTGAGE_PVG_UPPER = (0.40, 0.80, 0.90, None)
_MORTGAGE_PI = (
    (0.010916, 0.213407, 0.460536, 0.751614, 1.0),
    (0.019158, 0.274332, 0.520824, 0.789511, 1.0),
    (0.02515, 0.2793, 0.5258, 0.796952, 1.0),
    (0.0274, 0.2843, 0.5308, 0.803677, 1.0),
)
_MORTGAGE_PDI = (
    (0.000225, 0.000441, 0.000482, 0.000482, 0.000537),
    (0.021955, 0.028233, 0.029192, 0.029192, 0.030413),
    (0.215527, 0.2166, 0.2192, 0.221331, 0.22231),
    (0.272, 0.2903, 0.2959, 0.301558, 0.302436),
)
_MORTGAGE_PE = (
    (0.000002, 0.000094, 0.000222, 0.000362, 0.000537),
    (0.000421, 0.007745, 0.015204, 0.023047, 0.030413),
    (0.005421, 0.060496, 0.115255, 0.17639, 0.22231),
    (0.007453, 0.082532, 0.157064, 0.242355, 0.302436),
)

# B-1 N 3.1.2 (a): commercial leasing, dossier 03 D.5. PI by days past due; PDI by PVB bucket.
_LEASING_PI = {
    True: (0.0079, 0.0794, 0.2876, 0.5876, 1.0),   # real estate
    False: (0.0161, 0.1202, 0.4088, 0.6938, 1.0),  # non real estate
}
_LEASING_PVB_UPPER = (0.40, 0.50, 0.80, 0.90, None)
_LEASING_PDI = {
    True: (0.0005, 0.0005, 0.051, 0.232, 0.362),
    False: (0.182, 0.57, 0.684, 0.751, 0.789),
}

# B-1 N 3.1.2 (b): student loans, dossier 03 D.5. When payment is due: PI by days past due, one PDI; not due: one PI and
# one PDI per scheme.
_STUDENT_PI_DUE = {
    "CAE": (0.052, 0.372, 0.59, 0.728, 1.0),
    "CORFO": (0.029, 0.15, 0.434, 0.719, 1.0),
}
_STUDENT_PDI_DUE = 0.709
_STUDENT_NOT_DUE = {"CAE": (0.416, 0.503), "CORFO": (0.165, 0.458)}  # (PI, PDI)

# B-1 N 3.1.2 (c): generic commercial loans and factoring, dossier 03 D.5.
_COMMERCIAL_PI_GUARANTEE_LE_100 = (0.0186, 0.116, 0.2533, 0.4131, 1.0)
_COMMERCIAL_PI_GUARANTEE_GT_100 = (0.0268, 0.1345, 0.2692, 0.4131, 1.0)
_COMMERCIAL_PI_NO_GUARANTEE = (0.0491, 0.2293, 0.453, 0.6163, 1.0)
_COMMERCIAL_PTVG_UPPER = (0.60, 0.75, 0.90, None)
_COMMERCIAL_PDI_GUARANTEE = (0.05, 0.203, 0.322, 0.43)
_COMMERCIAL_PDI_GUARANTEE_RECOURSE = (0.032, 0.128, 0.203, 0.271)
_COMMERCIAL_PDI_NO_GUARANTEE = 0.569
_COMMERCIAL_PDI_NO_GUARANTEE_RECOURSE = 0.359

# B-1 N 3.1.3, Circular 2.346: consumer loans, dossier 03 D.5. Rows: maximum days past due in the bank 0-7, 8-30, 31-60,
# 61-89; columns: (mortgage, no 30+ system arrears), (mortgage, 30+ system arrears), (no mortgage, no arrears),
# (no mortgage, arrears). Defaulted debtors (90 days or more) have PI 1.
_CONSUMER_ROW_UPPER = (7, 30, 60, 89)
_CONSUMER_PI = (
    (0.033, 0.146, 0.066, 0.198),
    (0.204, 0.416, 0.306, 0.485),
    (0.502, 0.63, 0.651, 0.663),
    (0.626, 0.817, 0.723, 0.869),
)
_CONSUMER_PRODUCTS = ("leasing-auto", "instalment", "cards-lines-other")
_CONSUMER_PDI_MORTGAGE = (0.332, 0.477, 0.495)
_CONSUMER_PDI_NO_MORTGAGE = (0.332, 0.566, 0.603)

_MINIMUM_RATE = 0.005  # B-1 N 2.1.3: minimum provision of the individually assessed Normal portfolio (dossier 03 D.3)

_ROUND = 9  # decimals kept before a bucket comparison, so a float at a boundary lands in its inclusive bucket


def _bucket(value: float, uppers: tuple[float | None, ...], name: str, rule: str) -> int:
    """Index of the first bucket whose inclusive upper bound is at least ``value`` (``None`` is unbounded)."""
    if value != value or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r} ({rule})")
    v = round(float(value), _ROUND)
    for i, upper in enumerate(uppers):
        if upper is None or v <= upper:
            return i
    raise AssertionError("unreachable")  # pragma: no cover


def _dpd_index(dpd_bucket: str, allowed: tuple[str, ...], rule: str) -> int:
    if dpd_bucket not in allowed:
        raise ValueError(f"days-past-due bucket {dpd_bucket!r} is outside {list(allowed)} ({rule})")
    return allowed.index(dpd_bucket)


def _money(value: float, name: str) -> float:
    if value != value or value < 0:
        raise ValueError(f"{name} must be a non-negative amount, got {value!r}")
    return float(value)


def _rate(value: float, name: str) -> float:
    if value != value or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a decimal in [0, 1], got {value!r}")
    return float(value)


def b1_individual_category(category: str) -> tuple[float, float, float]:
    """(PI, PDI, PE) decimals of an individual-assessment category A1..A6, B1..B4 (CNCB B-1 N 2.1, "Hoja 3").

    Raises ``KeyError`` for any other category.
    """
    try:
        return _INDIVIDUAL[category]
    except (KeyError, TypeError):
        raise KeyError(
            f"category {category!r} is outside A1..A6 and B1..B4 (CNCB B-1 N 2.1, individual assessment)"
        ) from None


def b1_individual_provision(eap: float, ea: float, category: str, guarantor_category: str | None = None) -> float:
    r"""Provision of an individually assessed Normal or Substandard debtor (B-1 N 2.1.3).

    $(EAP - EA) PI PDI + EA PI_{aval} PDI_{aval}$: ``eap`` is the exposure subject to provisions (loans plus contingent
    credits less collateral at realisable value) and ``ea`` the guaranteed exposure, which is never deducted from
    the exposure; the guaranteed part takes the guarantor's category.
    """
    eap = _money(eap, "eap")
    ea = _money(ea, "ea")
    if ea > eap:
        raise ValueError(f"guaranteed exposure {ea!r} exceeds the exposure subject to provisions {eap!r} (B-1 N 2.1.3)")
    own = (eap - ea) * _pe(category)
    if ea == 0.0:
        return own
    if guarantor_category is None:
        raise ValueError("a guaranteed exposure needs the guarantor's category (B-1 N 2.1.3 and N 4.1)")
    return own + ea * _pe(guarantor_category)


def _pe(category: str) -> float:
    pi, pdi, _ = b1_individual_category(category)
    return pi * pdi


def b1_default_provision(exposure: float, recoverable: float) -> tuple[float, str, float]:
    """(TPE, category, provision) of a debtor in the default portfolio (B-1 N 2.2, "Hoja 8").

    $TPE = (E - R)/E$, rounded to 1e-9 before the bucket comparison; the provision is $E \\cdot PP$ with the category's
    PP. Upper bounds are inclusive: C1 to 3%, C2 to 20%, C3 to 30%, C4 to 50%, C5 to 80%, C6 above.
    """
    if exposure != exposure or exposure <= 0:
        raise ValueError(f"exposure must be positive, got {exposure!r} (B-1 N 2.2: TPE = (E - R) / E)")
    if recoverable != recoverable or not 0.0 <= recoverable <= exposure:
        raise ValueError(f"recoverable amount {recoverable!r} must lie in [0, exposure={exposure!r}] (B-1 N 2.2)")
    tpe = round((exposure - recoverable) / exposure, _ROUND)
    uppers = tuple(upper for _, upper, _ in _DEFAULT_BUCKETS)
    category, _, pp = _DEFAULT_BUCKETS[_bucket(tpe, uppers, "TPE", "B-1 N 2.2")]
    return tpe, category, exposure * pp


def b1_mortgage_pe(pvg: float, dpd_bucket: str) -> tuple[float, float, float]:
    """(PI, PDI, PE) of a residential mortgage (B-1 N 3.1.1, Circular 2.346 "Hoja 11"; 20 cells).

    ``pvg`` is the outstanding principal over the mortgage collateral value, a decimal; buckets <= 40%, <= 80%,
    <= 90%, > 90%. ``dpd_bucket`` is one of ``0``, ``1-29``, ``30-59``, ``60-89``, ``default``. PE is the transcribed
    cell. Also applies to residential leasing.
    """
    j = _dpd_index(dpd_bucket, _DPD, "B-1 N 3.1.1 residential mortgages")
    i = _bucket(pvg, _MORTGAGE_PVG_UPPER, "pvg", "B-1 N 3.1.1 residential mortgages")
    return _MORTGAGE_PI[i][j], _MORTGAGE_PDI[i][j], _MORTGAGE_PE[i][j]


def b1_leasing_pe(pvb: float, dpd_bucket: str, real_estate: bool) -> tuple[float, float, float]:
    """(PI, PDI, PE = PI x PDI) of commercial leasing (B-1 N 3.1.2 (a)).

    ``pvb`` is the present value of the lease over the asset value; PVB buckets <= 40%, <= 50%, <= 80%, <= 90%, > 90%.
    """
    j = _dpd_index(dpd_bucket, _DPD, "B-1 N 3.1.2 (a) commercial leasing")
    i = _bucket(pvb, _LEASING_PVB_UPPER, "pvb", "B-1 N 3.1.2 (a) commercial leasing")
    key = bool(real_estate)
    pi, pdi = _LEASING_PI[key][j], _LEASING_PDI[key][i]
    return pi, pdi, pi * pdi


def b1_student_loan_pe(scheme: str, dpd_bucket: str) -> tuple[float, float, float]:
    """(PI, PDI, PE = PI x PDI) of a student loan (B-1 N 3.1.2 (b)); ``scheme`` CAE or CORFO.

    ``dpd_bucket`` is ``0``, ``1-29``, ``30-59``, ``60-89``, ``default`` (payment due) or ``not-due``.
    """
    rule = "B-1 N 3.1.2 (b) student loans"
    if scheme not in _STUDENT_PI_DUE:
        raise ValueError(f"scheme {scheme!r} is outside ['CAE', 'CORFO'] ({rule})")
    if dpd_bucket == "not-due":
        pi, pdi = _STUDENT_NOT_DUE[scheme]
    else:
        j = _dpd_index(dpd_bucket, _DPD + ("not-due",), rule)
        pi, pdi = _STUDENT_PI_DUE[scheme][j], _STUDENT_PDI_DUE
    return pi, pdi, pi * pdi


def b1_commercial_pe(dpd_bucket: str, ptvg: float | None,
                     factoring_with_recourse: bool = False) -> tuple[float, float, float]:
    """(PI, PDI, PE = PI x PDI) of a generic commercial loan or factoring (B-1 N 3.1.2 (c)).

    ``ptvg`` is the obligations over the value of real guarantees, a decimal, or ``None`` for no guarantee. With a
    guarantee, PI depends on PTVG at most or above 100% and PDI on the buckets <= 60%, <= 75%, <= 90%, > 90%; factoring
    with recourse to the assignor has its own PDI column.
    """
    rule = "B-1 N 3.1.2 (c) generic commercial loans and factoring"
    j = _dpd_index(dpd_bucket, _DPD, rule)
    if ptvg is None:
        pi = _COMMERCIAL_PI_NO_GUARANTEE[j]
        pdi = _COMMERCIAL_PDI_NO_GUARANTEE_RECOURSE if factoring_with_recourse else _COMMERCIAL_PDI_NO_GUARANTEE
    else:
        i = _bucket(ptvg, _COMMERCIAL_PTVG_UPPER, "ptvg", rule)
        le_100 = round(float(ptvg), _ROUND) <= 1.0
        pi = (_COMMERCIAL_PI_GUARANTEE_LE_100 if le_100 else _COMMERCIAL_PI_GUARANTEE_GT_100)[j]
        pdi = (_COMMERCIAL_PDI_GUARANTEE_RECOURSE if factoring_with_recourse else _COMMERCIAL_PDI_GUARANTEE)[i]
    return pi, pdi, pi * pdi


def b1_consumer_pe(max_dpd_in_bank: int, has_mortgage_in_system: bool, arrears_30_in_system_3m: bool,
                   product: str) -> tuple[float, float, float]:
    """(PI, PDI, PE = PI x PDI) of a consumer loan (B-1 N 3.1.3, Circular 2.346).

    PI from the debtor's maximum days past due in the bank (0-7, 8-30, 31-60, 61-89; 90 or more gives 1), whether
    the debtor holds a residential mortgage in the system and whether it had a debt 30+ days past due in the
    system in the previous three months (16 cells). PDI by ``product`` (``leasing-auto``, ``instalment``,
    ``cards-lines-other``) and the mortgage in the system.
    """
    rule = "B-1 N 3.1.3 consumer loans"
    if product not in _CONSUMER_PRODUCTS:
        raise ValueError(f"product {product!r} is outside {list(_CONSUMER_PRODUCTS)} ({rule})")
    if max_dpd_in_bank != max_dpd_in_bank or max_dpd_in_bank < 0 or int(max_dpd_in_bank) != max_dpd_in_bank:
        raise ValueError(f"max_dpd_in_bank must be a non-negative integer, got {max_dpd_in_bank!r} ({rule})")
    pdis = _CONSUMER_PDI_MORTGAGE if has_mortgage_in_system else _CONSUMER_PDI_NO_MORTGAGE
    pdi = pdis[_CONSUMER_PRODUCTS.index(product)]
    if max_dpd_in_bank >= 90:
        pi = 1.0
    else:
        row = next(r for r, upper in enumerate(_CONSUMER_ROW_UPPER) if max_dpd_in_bank <= upper)
        col = (0 if has_mortgage_in_system else 2) + (1 if arrears_30_in_system_3m else 0)
        pi = _CONSUMER_PI[row][col]
    return pi, pdi, pi * pdi


def b1_group_provision(group_exposure: float, pe_group: float, guaranteed_share: float = 0.0,
                       pe_guarantor: float = 0.0) -> float:
    """Group-portfolio provision with guarantor substitution (B-1 N 4.1).

    $EG (1 - EA) PE_{group} + EG\\, EA\\, PE_{aval}$, ``guaranteed_share`` being EA/100 in [0, 1]. The 0.50% minimum
    is not applied here: it belongs to the individually assessed Normal portfolio only.
    """
    eg = _money(group_exposure, "group_exposure")
    share = _rate(guaranteed_share, "guaranteed_share")
    pe_g = _rate(pe_group, "pe_group")
    pe_a = _rate(pe_guarantor, "pe_guarantor")
    return eg * (1.0 - share) * pe_g + eg * share * pe_a


@dataclass(frozen=True)
class MinimumCheck:
    """Result of the 0.50% minimum on the individually assessed Normal portfolio (B-1 N 2.1.3)."""

    minimum_rate: float
    minimum_amount: float
    binding: bool
    top_up: float


def b1_minimum_normal_check(normal_exposure: float, normal_provision: float) -> MinimumCheck:
    """The 0.50% minimum provision on the individually assessed Normal portfolio, A1 to A6 (B-1 N 2.1.3).

    ``normal_exposure`` (loans plus contingent credits) and ``normal_provision`` are those of that portfolio only:
    not the group-assessed book and not the whole performing book. The minimum is always enforced:
    ``top_up = max(0, minimum - provision)`` and ``binding = top_up > 0``.
    """
    exposure = _money(normal_exposure, "normal_exposure")
    provision = _money(normal_provision, "normal_provision")
    minimum = _MINIMUM_RATE * exposure
    top_up = max(0.0, minimum - provision)
    return MinimumCheck(minimum_rate=_MINIMUM_RATE, minimum_amount=minimum, binding=top_up > 0.0, top_up=top_up)


def b1_group_max_rule(standard: float, internal: float | None, internal_not_objected: bool = False) -> tuple[float, str]:
    """(applied, method) of the group-assessment max rule (B-1 N 3).

    The larger of the standard and the internal method (method ``max``), unless the internal method has the CMF
    "no objection", then the internal one (``internal``); no internal method gives the standard (``standard``).
    """
    standard = _money(standard, "standard")
    if internal is None:
        return standard, "standard"
    internal = _money(internal, "internal")
    if internal_not_objected:
        return internal, "internal"
    return max(standard, internal), "max"


def b1_additional_provisions(amount: float, policy_min: float, policy_max: float) -> float:
    """Additional (counter-cyclical, general) provisions inside the board's policy limits (B-1 N 9).

    Returned as their own line: never part of a comparable B-1 total and never used to cover model deficiencies.
    Raises ``ValueError`` outside ``[policy_min, policy_max]`` or when the limits are inverted.
    """
    if policy_min > policy_max:
        raise ValueError(f"policy limits are inverted: min {policy_min!r} > max {policy_max!r} (B-1 N 9)")
    if amount != amount or not policy_min <= amount <= policy_max:
        raise ValueError(f"additional provisions {amount!r} are outside the board policy [{policy_min!r}, {policy_max!r}] (B-1 N 9)")
    return float(amount)
