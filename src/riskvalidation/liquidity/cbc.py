"""The counterbalancing capacity (CBC): the liquidity value of the unencumbered assets after haircuts and the amount
usable on each day.

Sources: dossier 05 E.1 line 238 (ECB LiST 2019 methodological note, paras 73 to 75 and Table 2.1: the liquidity value
of an asset is its market value less a haircut, the haircut order, the equity table, 30 and 50 percent for the rest),
lines 263 and 480 (held-to-maturity unrealised losses reduce the effective CBC), 276 (the monetisation schedule); the
U11 research addendum L1.6 (the exact order of para 74) and L4 (the Federal Reserve margins, the central bank's haircut
is 1 minus the margin). Contract `engine-liquidity-api` 1.2.0, module cbc.

The LCR haircut is taken from `regulatory.lcr_nsfr.HQLA_HAIRCUTS`, never a second table (decision Q1); the central
bank's
haircut is the caller's (1 minus its margin), with the 50 percent fallback only where the framework gives none. Money is
USD thousands, haircuts and shares are decimals in `[0, 1]`, days are integers counted from 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from riskvalidation.liquidity._checks import check_nonnegative_stock, check_unit_share
from riskvalidation.regulatory.lcr_nsfr import HQLA_HAIRCUTS

# Table 2.1 of the note (para 74, third bullet): the haircut of a marketable equity not eligible for a central bank,
# by the issuer's rating.
ECB_EQUITY_HAIRCUTS: Mapping[str, float] = MappingProxyType(
    {"AAA-A-": 0.20, "BBB+-BBB-": 0.35, "lower-or-unrated": 0.50})
ECB_NON_EQUITY_HQLA_HAIRCUT = 0.30   # a non-equity marketable asset that is HQLA under the LCR Delegated Regulation
FALLBACK_HAIRCUT = 0.50              # every other case where no haircut can be determined
KINDS = ("cash", "equity", "non-equity")
RULE_LABELS = ("nominal-value", "central-bank-framework", "lcr-haircut-hqla", "fifty-percent-fallback",
               "ecb-equity-table", "ecb-non-equity-hqla")


@dataclass(frozen=True)
class HtmResult:
    """The CBC of a held-to-maturity book on the carrying and on the fair-value basis."""

    unrealised_loss: float
    cbc_without_loss: float
    cbc_with_loss: float
    cbc_removed: float


def haircut_rule(kind: str, *, central_bank_eligible: bool = False, central_bank_haircut: float | None = None,
                 hqla_level: str | None = None, equity_rating: str | None = None) -> tuple[float, str]:
    """The haircut of an asset and the rule that gave it, in the order of the note (para 74).

    Cash, reserves and committed facilities are at nominal value (haircut 0, para 73). A central-bank-eligible asset
    takes the central bank's haircut when it is given; when it cannot be determined, the LCR haircut if the asset is
    HQLA (`hqla_level` a key of `HQLA_HAIRCUTS`) and 0.50 otherwise. A marketable asset not eligible takes the equity
    table for equities (AAA to A- 0.20, BBB+ to BBB- 0.35, lower or unrated 0.50) and 0.30 if HQLA under the LCR
    Delegated Regulation, 0.50 otherwise, for the rest. The result is `(haircut, rule label)`, the label one of
    `RULE_LABELS`."""
    if kind not in KINDS:
        raise ValueError(f"kind: must be one of {KINDS}, got {kind!r}")
    if hqla_level is not None and hqla_level not in HQLA_HAIRCUTS:
        raise ValueError(f"hqla_level: must be None or one of {tuple(HQLA_HAIRCUTS)}, got {hqla_level!r}")
    if kind == "cash":
        return 0.0, "nominal-value"
    if central_bank_eligible:
        if central_bank_haircut is not None:
            return float(check_unit_share(central_bank_haircut, "central_bank_haircut")[0]), "central-bank-framework"
        if hqla_level is not None:
            return HQLA_HAIRCUTS[hqla_level], "lcr-haircut-hqla"
        return FALLBACK_HAIRCUT, "fifty-percent-fallback"
    if kind == "equity":
        if equity_rating not in ECB_EQUITY_HAIRCUTS:
            raise ValueError(f"equity_rating: must be one of {tuple(ECB_EQUITY_HAIRCUTS)}, got {equity_rating!r}")
        return ECB_EQUITY_HAIRCUTS[equity_rating], "ecb-equity-table"
    if hqla_level is not None:
        return ECB_NON_EQUITY_HQLA_HAIRCUT, "ecb-non-equity-hqla"
    return FALLBACK_HAIRCUT, "fifty-percent-fallback"


def liquidity_value(market_value: np.ndarray | float, haircut: np.ndarray | float,
                    encumbered: np.ndarray | float = 0.0) -> np.ndarray | float:
    """`(market value - encumbered) x (1 - haircut)`: only unencumbered assets count (LiST para 63) and the liquidity
    value is the market value less a haircut (para 74). `0 <= encumbered <= market value`, `haircut` in `[0, 1]`;
    monotone decreasing in the haircut. Golden: `liquidity_value(100, 0.2) = 80` and `(100, 0.1, 20) = 72`."""
    scalar = np.ndim(market_value) == 0 and np.ndim(haircut) == 0 and np.ndim(encumbered) == 0
    mv = check_nonnegative_stock(market_value, "market_value")
    enc = check_nonnegative_stock(encumbered, "encumbered")
    hc = check_unit_share(haircut, "haircut")
    if np.any(enc > mv):
        raise ValueError(f"encumbered: must not exceed market_value, got {float(np.max(enc))} against "
                         f"{float(np.min(mv))}")
    out = (mv - enc) * (1.0 - hc)
    return float(out[0]) if scalar else out


def _schedule(points: Sequence[tuple[int, float]], horizon: int) -> np.ndarray:
    """The monetisation share on each day 1..H from a step schedule `(day, cumulative share)`."""
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon: must be a positive integer, got {horizon!r}")
    days, shares = [], []
    for point in points:
        try:
            day, share = point
        except (TypeError, ValueError) as exc:
            raise ValueError(f"points: each point must be (day, cumulative share), got {point!r}") from exc
        if day != int(day) or int(day) < 1:
            raise ValueError(f"points: days must be integers from 1, got {day!r}")
        days.append(int(day))
        shares.append(float(check_unit_share(share, "points")[0]))
    if any(b <= a for a, b in zip(days, days[1:], strict=False)):
        raise ValueError(f"points: days must be strictly increasing, got {days}")
    if any(b < a for a, b in zip(shares, shares[1:], strict=False)):
        raise ValueError(f"points: cumulative shares must be non-decreasing, got {shares}")
    out = np.zeros(int(horizon))
    for day, share in zip(days, shares, strict=True):
        if day <= horizon:
            out[day - 1:] = share
    return out


def monetisation_capacity(value: float, points: Sequence[tuple[int, float]], horizon: int) -> np.ndarray:
    """The amount usable on each day: `value x share(d)`, the share of day `d` that of the last point with `day <= d`
    (zero before the first point). `points` is the step schedule `(day, cumulative share)` (the named parameter
    `monetisation_schedule`); shares are non-decreasing and at most 1. Golden: `100` with `[(1, 0.5), (2, 0.8),
    (3, 1.0)]` over 4 days gives `[50, 80, 100, 100]`."""
    v = float(check_nonnegative_stock(value, "value")[0])
    return v * _schedule(points, horizon)


def cbc_stack(classes: Mapping[str, tuple[float, float, float]], points: Sequence[tuple[int, float]],
              horizon: int) -> dict[str, np.ndarray]:
    """The CBC stack of the ladder: for each class id `(market value used, encumbered, haircut)`, its liquidity value
    times the monetisation share of each day. The sum over classes is `CBC(t)`. Golden: `{"A": (100, 20, 0.1), "B":
    (50, 0, 0.5)}` with `[(1, 1.0)]` over 2 days gives `A = [72, 72]` and `B = [25, 25]`."""
    if not classes:
        raise ValueError("classes: must not be empty, got {}")
    share = _schedule(points, horizon)
    out: dict[str, np.ndarray] = {}
    for key, (market_value, encumbered, haircut) in classes.items():
        out[key] = float(liquidity_value(market_value, haircut, encumbered)) * share
    return out


def cbc_with_htm_loss(carrying: float, fair_value: float, encumbered: float, haircut: float) -> HtmResult:
    """The CBC of a held-to-maturity book without and with its unrealised loss (dossier 05 lines 263 and 480).

    `unrealised_loss = fair_value - carrying`; `cbc_without_loss = liquidity_value(carrying, haircut, encumbered)`;
    `cbc_with_loss = liquidity_value(fair_value, haircut, encumbered)`; `cbc_removed = cbc_without_loss -
    cbc_with_loss`, non-negative whenever `fair_value <= carrying`. The encumbered amount is the same on both bases;
    the loss is a stated assumption pro rata by type, never a reported by-type loss (A3)."""
    without = float(liquidity_value(carrying, haircut, encumbered))
    with_loss = float(liquidity_value(fair_value, haircut, encumbered))
    return HtmResult(unrealised_loss=float(fair_value) - float(carrying), cbc_without_loss=without,
                     cbc_with_loss=with_loss, cbc_removed=without - with_loss)


__all__ = ["ECB_EQUITY_HAIRCUTS", "ECB_NON_EQUITY_HQLA_HAIRCUT", "FALLBACK_HAIRCUT", "HtmResult", "KINDS",
           "RULE_LABELS", "cbc_stack", "cbc_with_htm_loss", "haircut_rule", "liquidity_value",
           "monetisation_capacity"]
