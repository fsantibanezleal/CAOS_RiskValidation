"""IRB risk-weight functions: Basel III final (BCBS CRE31, CRE32), EU CRR3, Basel II and the Chilean internal methods.

The capital requirement of a non-defaulted exposure is the loss at the 99.9% quantile of the systematic factor in the
one-factor model (Vasicek 2002; Gordy 2003), less the expected loss, with a maturity adjustment for non-retail
exposures (CRE31.5):

    K = [LGD N(G(PD) / sqrt(1 - R) + sqrt(R / (1 - R)) G(0.999)) - PD LGD] (1 + (M - 2.5) b) / (1 - 1.5 b),
    b = (0.11852 - 0.05478 ln PD)^2,  RWA = 12.5 K EAD,

with N the standard normal distribution and G its inverse. The asset correlation R depends on the asset class
(CRE31.5 to CRE31.16). A regime fixes what differs between frameworks: the 1.06 scaling factor of Basel II (CRE30.4,
absent from Basel III final and from CRR3 Article 153), the PD floors (CRE32.4 and CRE32.58), and, for the CMF's RAN
21-6, its own correlation curves and the regulatory consumer LGD. BCBS CRE99 Table 1 is reproduced cell by cell
(tests/test_irb.py). Nothing here says that a bank uses these methods: in Chile none does (CMF, informe normativo of
August 2026).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

from scipy.special import ndtr, ndtri

Q_999 = 0.999

#: the asset classes, and which of them carry the maturity adjustment
RETAIL = ("residential_mortgage", "qrre", "other_retail", "retail_consumer")
NON_RETAIL = ("corporate", "sovereign", "bank", "financial_institution", "hvcre", "sme")


@dataclass(frozen=True)
class Regime:
    """The parameters a framework fixes, each with its source."""

    id: str
    name: str
    in_force: str
    scaling: float
    pd_floor: float
    pd_floor_qrre: float  # QRRE revolvers (CRE32.58); transactors take pd_floor
    floor_exempt: tuple[str, ...]
    correlations: str  # "bcbs" or "cl_ran21_6"
    classes: tuple[str, ...]
    fixed_lgd: Mapping[str, float] = field(default_factory=dict)
    references: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # a regime is shared module state: freeze its tables so no caller can change them for everyone else
        object.__setattr__(self, "fixed_lgd", MappingProxyType(dict(self.fixed_lgd)))
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))

    def to_dict(self) -> dict:
        """A JSON-safe copy (the tables as plain dictionaries)."""
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out.update(classes=list(self.classes), floor_exempt=list(self.floor_exempt), fixed_lgd=dict(self.fixed_lgd),
                   references=dict(self.references))
        return out


_BCBS_REFS = {
    "risk_weight": "CRE31.5",
    "correlation_corporate": "CRE31.5",
    "large_financial_institution": "CRE31.7",
    "firm_size_adjustment": "CRE31.8",
    "hvcre": "CRE31.11",
    "defaulted": "CRE31.3",
    "residential_mortgage": "CRE31.14",
    "qrre": "CRE31.15",
    "other_retail": "CRE31.16",
    "pd_floor": "CRE32.4 (corporate and bank), CRE32.58 (retail; QRRE revolvers 0.10%)",
    "maturity": "CRE32.46 to CRE32.47 (1 to 5 years)",
}

REGIMES: dict[str, Regime] = {
    "basel3": Regime(
        id="basel3",
        name="Basel III final (BCBS CRE31 and CRE32, in force 2023-01-01)",
        in_force="2023-01-01",
        scaling=1.0,
        pd_floor=0.0005,
        pd_floor_qrre=0.0010,
        floor_exempt=("sovereign",),
        correlations="bcbs",
        classes=NON_RETAIL[:-1] + RETAIL[:-1],
        references={**_BCBS_REFS, "scaling": "no scaling factor in CRE31 (in force 2023-01-01)"},
    ),
    "crr3": Regime(
        id="crr3",
        name="EU CRR3 (Regulation (EU) 2024/1623, Article 153; applies from 2025-01-01)",
        in_force="2025-01-01",
        scaling=1.0,
        pd_floor=0.0005,
        pd_floor_qrre=0.0010,
        floor_exempt=("sovereign",),
        correlations="bcbs",
        classes=NON_RETAIL[:-1] + RETAIL[:-1],
        references={
            **_BCBS_REFS,
            "risk_weight": "CRR3 Article 153(1)(iii), the Basel III final formula without the 1.06 factor",
            "pd_floor": "CRR3 Articles 160 and 163 (0.05%; QRRE revolvers 0.10%)",
            "scaling": "no scaling factor in CRR3 Article 153",
        },
    ),
    "basel2": Regime(
        id="basel2",
        name="Basel II (BCBS CRE30.4 and CRE32, versions in force 2019-12-15)",
        in_force="2019-12-15",
        scaling=1.06,
        pd_floor=0.0003,
        pd_floor_qrre=0.0003,
        floor_exempt=("sovereign",),
        correlations="bcbs",
        classes=NON_RETAIL[:-1] + RETAIL[:-1],
        references={
            **_BCBS_REFS,
            "scaling": "CRE30.4 (in force 2019-12-15): a scaling factor of 1.06 on IRB RWA",
            "pd_floor": "CRE32.3 and CRE32.51 (in force 2019-12-15): 0.03%",
        },
    ),
    "cl_ran21_6": Regime(
        id="cl_ran21_6",
        name="CMF RAN 21-6, internal methodologies (numeral 4.4; Circular N°2.370 of 2026-04-10)",
        in_force="2026-04-10",
        scaling=1.0,
        pd_floor=0.0005,
        pd_floor_qrre=0.0005,
        floor_exempt=(),
        correlations="cl_ran21_6",
        classes=(
            "corporate", "financial_institution", "sme", "hvcre", "residential_mortgage", "other_retail",
            "retail_consumer",
        ),
        fixed_lgd={"retail_consumer": 0.613},
        references={
            "risk_weight": "RAN 21-6 numeral 4.4.1 (Hoja 20)",
            "correlations": "RAN 21-6 numeral 4.4.2: corporates 0.12 to 0.24, non-bank financial institutions 1.25 "
            "times, PYMES 0.08 to 0.20, HVCRE 0.12 to 0.30, retail 0.03 to 0.16, residential mortgages 15%",
            "pd_floor": "RAN 21-6 numeral 4: 'no podrá ser inferior a 0,05%'",
            "consumer_lgd": "RAN 21-6 numeral 4: 'se debe considerar una PDI de 61,3%' for the consumer portfolio",
            "maturity": "RAN 21-6 numeral 4: M floored at 1 and capped at 5 years",
            "defaulted": "RAN 21-6 numeral 4.4.1: K = max(0, PDI - best estimate of expected loss)",
            "scaling": "no scaling factor",
        },
    ),
}


def _w(pd: float, k: float) -> float:
    return (1.0 - math.exp(-k * pd)) / (1.0 - math.exp(-k))


def correlation(asset_class: str, pd: float, *, sales_eur_m: float | None = None, large_fi: bool = False,
                regime: str = "basel3") -> float:
    """The asset correlation R of an exposure (CRE31.5 to CRE31.16, or RAN 21-6 numeral 4.4.2)."""
    reg = _regime(regime, asset_class)
    if reg.correlations == "cl_ran21_6":
        if asset_class == "corporate":
            return 0.12 * _w(pd, 50) + 0.24 * (1.0 - _w(pd, 50))
        if asset_class == "financial_institution":
            return 1.25 * (0.12 * _w(pd, 50) + 0.24 * (1.0 - _w(pd, 50)))
        if asset_class == "sme":
            return 0.08 * _w(pd, 50) + 0.20 * (1.0 - _w(pd, 50))
        if asset_class == "hvcre":
            return 0.12 * _w(pd, 50) + 0.30 * (1.0 - _w(pd, 50))
        if asset_class == "residential_mortgage":
            return 0.15
        return 0.03 * _w(pd, 35) + 0.16 * (1.0 - _w(pd, 35))  # other_retail, retail_consumer
    if asset_class in ("corporate", "sovereign", "bank", "financial_institution"):
        r = 0.12 * _w(pd, 50) + 0.24 * (1.0 - _w(pd, 50))
        if asset_class == "corporate" and sales_eur_m is not None and sales_eur_m < 50.0:
            s = min(max(sales_eur_m, 5.0), 50.0)  # CRE31.8: sales below EUR 5 million are treated as 5
            r -= 0.04 * (1.0 - (s - 5.0) / 45.0)
        if large_fi or asset_class == "financial_institution":
            r *= 1.25  # CRE31.7
        return r
    if asset_class == "hvcre":
        return 0.12 * _w(pd, 50) + 0.30 * (1.0 - _w(pd, 50))
    if asset_class == "residential_mortgage":
        return 0.15
    if asset_class == "qrre":
        return 0.04
    return 0.03 * _w(pd, 35) + 0.16 * (1.0 - _w(pd, 35))  # other_retail


def _regime(regime: str, asset_class: str) -> Regime:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; known: {sorted(REGIMES)}")
    reg = REGIMES[regime]
    if asset_class not in reg.classes:
        raise ValueError(f"{reg.name} has no asset class {asset_class!r}; its classes: {list(reg.classes)}")
    return reg


@dataclass(frozen=True)
class IRBResult:
    """The capital requirement of one exposure and every parameter that produced it."""

    asset_class: str
    regime: str
    pd_input: float
    pd_applied: float
    lgd: float
    maturity: float | None
    correlation: float | None
    maturity_adjustment: float | None
    k: float
    risk_weight: float
    rwa: float | None
    defaulted: bool
    references: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def capital_requirement(
    asset_class: str,
    pd: float,
    lgd: float | None = None,
    *,
    maturity: float = 2.5,
    sales_eur_m: float | None = None,
    large_fi: bool = False,
    ead: float | None = None,
    revolver: bool = False,
    defaulted: bool = False,
    el_best_estimate: float | None = None,
    regime: str = "basel3",
) -> IRBResult:
    """K, the risk weight 12.5 K (times the regime's scaling factor) and, given an EAD, the RWA of one exposure.

    ``pd`` is floored at the regime's floor (sovereigns exempt where the regime says so; a QRRE revolver, ``revolver``,
    takes the higher QRRE floor of CRE32.58, a transactor the retail one). A PD of 0, the estimate of a grade that never
    defaulted (a long-run average over cohorts without a default), is the case the floor exists for: it takes the floor,
    and is refused only where no floor applies (an exempt class), the formula having no value at 0. ``maturity`` is
    clamped to 1 to
    5 years for non-retail classes, and retail classes have no maturity adjustment. A defaulted exposure gets
    K = max(0, LGD - EL best estimate) (CRE31.3). Where a regime fixes the LGD of a class (RAN 21-6 consumer, 61.3%) a
    different ``lgd`` is refused rather than silently replaced.
    """
    reg = _regime(regime, asset_class)
    notes: list[str] = []
    fixed = reg.fixed_lgd.get(asset_class)
    if fixed is not None:
        if lgd is not None and abs(lgd - fixed) > 1e-12:
            raise ValueError(f"{reg.name} fixes the LGD of {asset_class} at {fixed}; got {lgd}")
        lgd = fixed
        notes.append(f"LGD fixed by the regime at {fixed:.1%}")
    if lgd is None:
        raise ValueError("lgd is required for this asset class")
    if not 0.0 <= lgd <= 1.0:
        raise ValueError("lgd must lie in [0, 1]")
    if defaulted:
        if el_best_estimate is None:
            raise ValueError("a defaulted exposure needs its best estimate of expected loss")
        k = max(0.0, lgd - el_best_estimate)
        rw = 12.5 * k * reg.scaling
        return IRBResult(asset_class, reg.id, pd, 1.0, lgd, None, None, None, k, rw, None if ead is None else rw * ead,
                         True, {"defaulted": reg.references.get("defaulted", ""), "scaling": reg.references["scaling"]},
                         tuple(notes))
    floor = reg.pd_floor_qrre if asset_class == "qrre" and revolver else reg.pd_floor
    exempt = asset_class in reg.floor_exempt
    if not 0.0 <= pd < 1.0 or (pd == 0.0 and (exempt or floor <= 0.0)):
        raise ValueError("pd must lie in (0, 1) for a non-defaulted exposure (0 only where the regime floors it)")
    pd_applied = pd if exempt else max(pd, floor)
    if pd_applied != pd:
        notes.append(f"PD floored at {floor:.2%}")
    r = correlation(asset_class, pd_applied, sales_eur_m=sales_eur_m, large_fi=large_fi, regime=regime)
    k = lgd * ndtr((ndtri(pd_applied) + math.sqrt(r) * ndtri(Q_999)) / math.sqrt(1.0 - r)) - pd_applied * lgd
    m_used: float | None = None
    ma: float | None = None
    if asset_class not in RETAIL:
        m_used = min(max(maturity, 1.0), 5.0)
        if m_used != maturity:
            notes.append(f"maturity clamped to {m_used:g} years")
        b = (0.11852 - 0.05478 * math.log(pd_applied)) ** 2
        ma = (1.0 + (m_used - 2.5) * b) / (1.0 - 1.5 * b)
        k *= ma
    k = max(k, 0.0)
    rw = 12.5 * k * reg.scaling
    src = reg.references
    refs = {key: src[key] for key in ("risk_weight", "pd_floor", "scaling") if key in src}
    refs["correlation"] = src.get(asset_class) or src.get("correlation_corporate") or src.get("correlations") or ""
    if asset_class == "corporate" and sales_eur_m is not None and reg.correlations == "bcbs":
        refs["firm_size_adjustment"] = reg.references["firm_size_adjustment"]
    if (large_fi or asset_class == "financial_institution") and reg.correlations == "bcbs":
        refs["large_financial_institution"] = reg.references["large_financial_institution"]
    if m_used is not None:
        refs["maturity"] = reg.references.get("maturity", "")
    return IRBResult(asset_class, reg.id, pd, pd_applied, lgd, m_used, r, ma, k, rw, None if ead is None else rw * ead,
                     False, refs, tuple(notes))


def el_versus_provisions(
    expected_loss: float, eligible_provisions: float, irb_credit_rwa: float
) -> dict[str, float | str]:
    """Expected loss against eligible provisions (CRE35; CAP30.13; CAP10.19): a shortfall is deducted from CET1 in full,
    an excess is recognised in Tier 2 up to 0.6% of IRB credit RWA."""
    if min(expected_loss, eligible_provisions, irb_credit_rwa) < 0:
        raise ValueError("amounts must be non-negative")
    shortfall = max(0.0, expected_loss - eligible_provisions)
    excess = max(0.0, eligible_provisions - expected_loss)
    return {
        "shortfall": shortfall,
        "excess": excess,
        "cet1_deduction": shortfall,
        "tier2_addback": min(excess, 0.006 * irb_credit_rwa),
        "references": "CRE35.2 to CRE35.4; CAP30.13 (shortfall deducted from CET1); CAP10.19 (excess in Tier 2 up to "
        "0.6% of IRB credit RWA)",
    }
