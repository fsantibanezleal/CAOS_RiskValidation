"""The default risk capital requirement for non-securitisations, DRC (MAR22.1 to MAR22.26, in force 2023-01-01,
published 2020-03-27).

For each exposure the gross jump-to-default (JTD) risk position is (MAR22.11, MAR22.13)

    JTD(long) = max(LGD x notional + P&L, 0),   JTD(short) = min(LGD x notional + P&L, 0)

with a long carrying a positive notional and a short a negative one, and a loss already taken recorded negative in the
P&L. It is weighted by the fraction of the one-year horizon, min(1, max(0.25, maturity in years)) (MAR22.15, MAR22.18;
cash equities are assigned the caller's maturity of three months or more than one year, MAR22.16). Exposures to the same
obligor are offset where the short has the same or lower seniority than the long (MAR22.19), giving a net long and a net
short per obligor. Within each of the three buckets (MAR22.22) the hedge benefit ratio is

    HBR = sum net JTD long / (sum net JTD long + sum |net JTD short|)           (MAR22.23)

and the bucket requirement is DRC_b = max(sum RW x net long - HBR x sum RW x |net short|, 0) (MAR22.25), the risk
weights being those of Table 2 by credit quality category (MAR22.24). The total is the plain sum of the buckets
(MAR22.26).

The text leaves the HBR undefined when both sums are zero: the engine defines it as 0, and the bucket's charge is then
0. The seniority ladder behind MAR22.19 (equity, non-senior debt, senior debt, covered bond, the most junior first) is
the engine's ordering of the four instrument classes of MAR22.12, and a caller may pass its own integer ranks instead.
The maturity weight of MAR22.19 is the one already carried by the gross JTD (footnote 1 of MAR22.15 states that the
scaling is of the gross JTD), so it is not applied twice.

Not implemented (stated limits): the default risk of securitisations, both the non-correlation trading portfolio and the
correlation trading portfolio (MAR22.27 to MAR22.45); there is no diversification between the three requirements
(MAR22.4), so this module returns the non-securitisation requirement alone. The look-through of index instruments
(MAR22.5), the guaranteed-bond rule (MAR22.19, CRE22.71), the national discretion of a zero weight on sovereigns
(MAR22.7), the funds of MAR22.8, and the choice of the seniority, rating and bucket of each exposure are the caller's.
The engine computes and states no compliance claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from riskvalidation.regulatory.standardised import RegulatoryParameter, _finite, _nonnegative

MAR22_IN_FORCE = "2023-01-01"

#: MAR22.12: the LGD by instrument class
LGD: dict[str, RegulatoryParameter] = {
    "equity_or_non_senior": RegulatoryParameter(1.0, "MAR22.12", "MAR22.12 equity and non-senior debt", MAR22_IN_FORCE),
    "senior": RegulatoryParameter(0.75, "MAR22.12", "MAR22.12 senior debt", MAR22_IN_FORCE),
    "covered_bond": RegulatoryParameter(0.25, "MAR22.12", "MAR22.12 covered bonds", MAR22_IN_FORCE),
}
#: MAR22.15: the capital horizon, one year
HORIZON_YEARS = RegulatoryParameter(1.0, "MAR22.15", "MAR22.15 capital horizon", MAR22_IN_FORCE)
#: MAR22.18: the maturity weighting is floored at one fourth (three months)
MATURITY_FLOOR_YEARS = RegulatoryParameter(0.25, "MAR22.18", "MAR22.18 maturity floor", MAR22_IN_FORCE)
#: MAR22.22: the three buckets, the value being their ordinal in the list of the paragraph
BUCKETS: dict[str, RegulatoryParameter] = {
    name: RegulatoryParameter(float(i), "MAR22.22", "MAR22.22 buckets", MAR22_IN_FORCE)
    for i, name in enumerate(("corporates", "sovereigns", "local_governments"), start=1)
}

#: MAR22.24 Table 2: the default risk weights by credit quality category, as printed in percent
_TABLE_2_PERCENT = (
    ("AAA", 0.5),
    ("AA", 2.0),
    ("A", 3.0),
    ("BBB", 6.0),
    ("BB", 15.0),
    ("B", 30.0),
    ("CCC", 50.0),
    ("unrated", 15.0),
    ("defaulted", 100.0),
)
RATINGS = tuple(name for name, _ in _TABLE_2_PERCENT)
DEFAULT_RISK_WEIGHTS: dict[str, RegulatoryParameter] = {
    name: RegulatoryParameter(pct / 100.0, "MAR22.24", "MAR22.24 Table 2", MAR22_IN_FORCE)
    for name, pct in _TABLE_2_PERCENT
}

#: the ladder behind MAR22.19, the most junior first (the engine's ordering of the classes of MAR22.12)
SENIORITY = {"equity": 0, "non_senior_debt": 1, "senior_debt": 2, "covered_bond": 3}

_PARAGRAPHS = {
    "gross_jtd": "MAR22.11, MAR22.13",
    "maturity": "MAR22.15, MAR22.18",
    "offsetting": "MAR22.19",
    "hedge_benefit_ratio": "MAR22.23",
    "bucket": "MAR22.25",
    "charge": "MAR22.26 (the plain sum of the bucket requirements, no hedging between buckets)",
    "securitisations": "MAR22.4 (no diversification with the securitisation requirements, which are not computed here)",
}


def maturity_weight(maturity_years: float) -> float:
    """MAR22.15 and MAR22.18: min(1, max(0.25, maturity in years))."""
    m = _nonnegative("maturity_years", maturity_years)
    return min(HORIZON_YEARS.value, max(MATURITY_FLOOR_YEARS.value, m))


def _lgd_value(lgd: float | str) -> float:
    if isinstance(lgd, str):
        if lgd not in LGD:
            raise ValueError(f"lgd {lgd!r} is not one of {sorted(LGD)} (MAR22.12)")
        return LGD[lgd].value
    value = _finite("lgd", lgd)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"lgd must lie in [0, 1], got {lgd!r}")
    return value


def gross_jtd(
    direction: str,
    notional: float,
    pnl: float,
    lgd: float | str,
    maturity_years: float,
    *,
    recovery_linked: bool = True,
    unwind_allowed: bool = False,
) -> float:
    """MAR22.11 to MAR22.18: the gross JTD of one exposure, weighted by its maturity.

    `direction` is "long" or "short" by whether a default results in a loss or a gain (MAR22.10); `notional` is positive
    for a long and negative for a short, `pnl` the cumulative mark-to-market result already taken, a loss negative
    (MAR22.13); `lgd` is a number or one of the keys of `LGD`. With `recovery_linked=False` (a price not linked to the
    recovery rate of the defaulter, MAR22.12) the notional is not multiplied by the LGD; with `unwind_allowed=True`
    (legal terms that allow unwinding with no exposure to default risk, MAR22.13) the JTD is zero."""
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    n = _finite("notional", notional)
    if direction == "long" and n < 0.0:
        raise ValueError(f"notional must be at least 0 for a long exposure (MAR22.13), got {notional!r}")
    if direction == "short" and n > 0.0:
        raise ValueError(f"notional must be at most 0 for a short exposure (MAR22.13), got {notional!r}")
    p = _finite("pnl", pnl)
    factor = _lgd_value(lgd) if recovery_linked else 1.0
    if not recovery_linked:
        _lgd_value(lgd)  # still validated, even though it is not applied
    weight = maturity_weight(maturity_years)
    if unwind_allowed:
        return 0.0
    raw = factor * n + p
    base = max(raw, 0.0) if direction == "long" else min(raw, 0.0)
    return base * weight


@dataclass(frozen=True)
class DrcExposure:
    """One exposure for the netting: its obligor, bucket, rating category, seniority and signed gross JTD (positive
    long, negative short, already weighted by the maturity as `gross_jtd` returns it)."""

    obligor: str
    bucket: str
    rating: str
    seniority: str | int
    jtd: float


@dataclass(frozen=True)
class NetJtd:
    """The net JTD risk positions of one obligor: `net_long` at least 0, `net_short` at most 0."""

    obligor: str
    bucket: str
    rating: str
    net_long: float
    net_short: float


def _rank(seniority: object) -> int:
    if isinstance(seniority, bool):
        raise ValueError(f"seniority must be one of {list(SENIORITY)} or an integer rank, got {seniority!r}")
    if isinstance(seniority, int):
        return seniority
    if isinstance(seniority, str) and seniority in SENIORITY:
        return SENIORITY[seniority]
    raise ValueError(f"seniority {seniority!r} is not one of {list(SENIORITY)} or an integer rank (MAR22.19)")


def _check_bucket(name: object) -> str:
    if name not in BUCKETS:
        raise ValueError(f"bucket {name!r} is not one of {list(BUCKETS)} (MAR22.22)")
    return str(name)


def _check_rating(name: object) -> str:
    if name not in DEFAULT_RISK_WEIGHTS:
        raise ValueError(f"rating {name!r} is not one of {list(RATINGS)} (MAR22.24 Table 2)")
    return str(name)


def net_jtd(exposures: Iterable[DrcExposure]) -> list[NetJtd]:
    """MAR22.19 to MAR22.21: offset per obligor, a short only against a long of the same or higher seniority.

    The offset is the largest the rule allows: the shorts are taken from the most senior to the most junior, each
    against the most junior longs it may offset (a short in an equity may offset a long in a bond, a short in a bond
    cannot offset a long in the equity). Returns one `NetJtd` per obligor in order of first appearance."""
    order: list[str] = []
    seen: dict[str, tuple[str, str]] = {}
    longs: dict[str, list[list[float]]] = {}
    shorts: dict[str, list[list[float]]] = {}
    for e in exposures:
        bucket, rating = _check_bucket(e.bucket), _check_rating(e.rating)
        rank = _rank(e.seniority)
        amount = _finite("jtd", e.jtd)
        if e.obligor not in seen:
            seen[e.obligor] = (bucket, rating)
            order.append(e.obligor)
            longs[e.obligor], shorts[e.obligor] = [], []
        elif seen[e.obligor] != (bucket, rating):
            raise ValueError(
                f"obligor {e.obligor!r} appears with the bucket and rating {seen[e.obligor]} and {(bucket, rating)}: "
                "one obligor has one bucket and one rating category"
            )
        if amount > 0.0:
            longs[e.obligor].append([float(rank), amount])
        elif amount < 0.0:
            shorts[e.obligor].append([float(rank), -amount])
    out = []
    for obligor in order:
        long_list = sorted(longs[obligor], key=lambda x: x[0])  # the most junior longs first
        left_short = 0.0
        for s_rank, s_amount in sorted(shorts[obligor], key=lambda x: -x[0]):  # the most senior shorts first
            remaining = s_amount
            for entry in long_list:
                if remaining <= 0.0:
                    break
                if entry[0] >= s_rank and entry[1] > 0.0:
                    take = min(remaining, entry[1])
                    entry[1] -= take
                    remaining -= take
            left_short += remaining
        bucket, rating = seen[obligor]
        out.append(NetJtd(obligor, bucket, rating, sum(x[1] for x in long_list), -left_short))
    return out


def hedge_benefit_ratio(net_long_sum: float, net_short_abs_sum: float) -> float:
    """MAR22.23: the net long sum over the net long sum plus the absolute net short sum; 0 when both are 0 (the text
    leaves that case undefined: the bucket has no position, and its charge is 0)."""
    long_sum = _nonnegative("net_long_sum", net_long_sum)
    short_sum = _nonnegative("net_short_abs_sum", net_short_abs_sum)
    total = long_sum + short_sum
    return long_sum / total if total > 0.0 else 0.0


def _weighted(name: str, by_rating: Mapping[str, float], short: bool) -> tuple[float, float]:
    """The sum of the amounts and the sum of the risk-weighted absolute amounts of one side of a bucket."""
    total = weighted = 0.0
    for rating, amount in by_rating.items():
        _check_rating(rating)
        v = _finite(name, amount)
        if short and v > 0.0:
            raise ValueError(f"{name} must be at most 0 (a net short JTD is negative), got {amount!r} for {rating!r}")
        if not short and v < 0.0:
            raise ValueError(f"{name} must be at least 0 (a net long JTD is positive), got {amount!r} for {rating!r}")
        total += abs(v)
        weighted += DEFAULT_RISK_WEIGHTS[rating].value * abs(v)
    return total, weighted


def _bucket(long_by_rating: Mapping[str, float], short_by_rating: Mapping[str, float]) -> tuple[float, float]:
    long_sum, long_weighted = _weighted("long_by_rating", long_by_rating, short=False)
    short_sum, short_weighted = _weighted("short_by_rating", short_by_rating, short=True)
    hbr = hedge_benefit_ratio(long_sum, short_sum)
    return max(long_weighted - hbr * short_weighted, 0.0), hbr


def bucket_drc(long_by_rating: Mapping[str, float], short_by_rating: Mapping[str, float]) -> float:
    """MAR22.25: max(sum RW x net long - HBR x sum RW x |net short|, 0) for one bucket, the two mappings giving the net
    long (at least 0) and the net short (at most 0) JTD by credit quality category of Table 2 (MAR22.24)."""
    return _bucket(long_by_rating, short_by_rating)[0]


@dataclass(frozen=True)
class DrcResult:
    """The DRC of the non-securitisation portfolio: the charge of each of the three buckets, their hedge benefit ratios
    and their plain sum, with the paragraphs."""

    bucket_charges: dict[str, float]
    hedge_benefit_ratios: dict[str, float]
    charge: float
    paragraphs: dict[str, str]


def drc_non_securitisation(
    buckets: Mapping[str, tuple[Mapping[str, float], Mapping[str, float]]],
) -> DrcResult:
    """MAR22.26: the plain sum of the bucket requirements; `buckets` maps a bucket name to its (net long by rating, net
    short by rating) mappings, and a bucket not given has no position. No hedging is recognised between buckets, and no
    diversification against securitisations (MAR22.4)."""
    for name in buckets:
        _check_bucket(name)
    charges: dict[str, float] = {}
    ratios: dict[str, float] = {}
    for name in BUCKETS:
        long_by, short_by = buckets.get(name, ({}, {}))
        charges[name], ratios[name] = _bucket(long_by, short_by)
    return DrcResult(charges, ratios, sum(charges.values()), dict(_PARAGRAPHS))


def drc_from_exposures(exposures: Iterable[DrcExposure]) -> DrcResult:
    """From the signed gross JTD of the exposures to the requirement: `net_jtd`, the net positions by bucket and rating
    category, then `drc_non_securitisation`."""
    by_bucket: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for n in net_jtd(exposures):
        long_by, short_by = by_bucket.setdefault(n.bucket, ({}, {}))
        if n.net_long > 0.0:
            long_by[n.rating] = long_by.get(n.rating, 0.0) + n.net_long
        if n.net_short < 0.0:
            short_by[n.rating] = short_by.get(n.rating, 0.0) + n.net_short
    return drc_non_securitisation(by_bucket)


__all__ = [
    "BUCKETS",
    "DEFAULT_RISK_WEIGHTS",
    "HORIZON_YEARS",
    "LGD",
    "MATURITY_FLOOR_YEARS",
    "RATINGS",
    "SENIORITY",
    "DrcExposure",
    "DrcResult",
    "NetJtd",
    "bucket_drc",
    "drc_from_exposures",
    "drc_non_securitisation",
    "gross_jtd",
    "hedge_benefit_ratio",
    "maturity_weight",
    "net_jtd",
]
