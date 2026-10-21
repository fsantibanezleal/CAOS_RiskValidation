"""Scenario families (idiosyncratic, market-wide, combined) and phases (acute 1 to 5 days, protracted up to 6 to 12
months), composed from the parameter sets of `runoff` and the haircuts of `cbc`; the fast-run calibration, the anchor
and
the two sensitivity grids.

Sources: dossier 05 E.2 line 244 (BCBS Principle 10: short-term and protracted, institution-specific and market-wide
scenarios, individually and in combination), E.5 lines 276 to 278 (rungs 2 to 4: segments, scenario engine, run-speed
calibration to Table A.1, sensitivity to the uninsured share and to unrealised securities losses), 279; the U11 research
addendum L1 (the LiST shocks are idiosyncratic only, annex slide 5; collateral calls have no numeric parameter, N paras
123
to 126). Contract `engine-liquidity-api` 2.0.0, module scenarios.

`Scenario`, `ListScenario` and `FastRun` are frozen dataclasses. A scenario input the dossiers do not settle is a
convention, labelled: the fast-run calibration rule (the dossiers give no method, UNVERIFIED), collateral calls (a
scenario input the caller supplies, not modelled here, UNVERIFIED as a parameter) and the mapping of a variant to a
family and phase (the named parameter `variant_scenarios`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from riskvalidation.liquidity import ladder as _ladder
from riskvalidation.liquidity import runoff as _runoff
from riskvalidation.liquidity._checks import check_increasing_grid, check_nonnegative_stock, check_unit_share

FAMILIES = ("idiosyncratic", "market-wide", "combined")
PHASES = ("acute", "protracted")
FAST_RUN_RULES = ("maximum-implied-daily-rate", "median-implied-daily-rate", "mean-implied-daily-rate")
FAST_RUN_CLASS = "concentrated-uninsured"  # the one segment class a fast-run parameter set holds


@dataclass(frozen=True)
class ListScenario:
    """The ECB LiST 2019 parameters of one severity: open-maturity daily rates by class, contractual roll-off and inflow
    rates, committed-facility drawdown rates and the relevant days of the horizon. The LiST shocks are idiosyncratic
    only (annex slide 5). A class is read by its id (`open_maturity["other-retail"]`) or as an attribute with
    underscores (`stable_retail`)."""

    severity: str
    open_maturity: Mapping[str, float]
    contractual_outflow: Mapping[str, float]
    contractual_inflow: Mapping[str, float]
    drawdown: Mapping[str, float]
    relevant_days: int = _runoff.LIST_RELEVANT_DAYS

    def __post_init__(self) -> None:
        for name in ("open_maturity", "contractual_outflow", "contractual_inflow", "drawdown"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __reduce__(self) -> tuple[Any, ...]:  # a mappingproxy cannot be pickled or deep-copied
        return (type(self), (self.severity, dict(self.open_maturity), dict(self.contractual_outflow),
                             dict(self.contractual_inflow), dict(self.drawdown), self.relevant_days))

    def __hash__(self) -> int:
        return hash((self.severity, self.relevant_days, tuple(self.open_maturity.items())))

    def class_rate(self, liquidity_class: str) -> float:
        """The daily rate of an open-maturity class; a class the set does not hold raises, never mapped silently."""
        if liquidity_class not in self.open_maturity:
            raise ValueError(f"liquidity class {liquidity_class!r}: not held by the LiST {self.severity} "
                             f"parameter set, known classes are {tuple(self.open_maturity)}")
        return self.open_maturity[liquidity_class]

    def __getattr__(self, name: str) -> float:
        if not name.startswith("_"):
            key = name.replace("_", "-")
            held = self.__dict__.get("open_maturity", {})
            if key in held:
                return held[key]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


@dataclass(frozen=True)
class FastRun:
    """The fast-run parameter set: the daily rate of the concentrated uninsured segment (decimal per relevant day)
    and the
    implied rates it came from. It holds one class, `FAST_RUN_CLASS`."""

    rule: str
    daily_rate: float
    implied_rates: tuple[float, ...]
    rows: tuple[tuple[float, int], ...]

    @property
    def open_maturity(self) -> Mapping[str, float]:
        return MappingProxyType({FAST_RUN_CLASS: self.daily_rate})

    def class_rate(self, liquidity_class: str) -> float:
        if liquidity_class != FAST_RUN_CLASS:
            raise ValueError(f"liquidity class {liquidity_class!r}: not held by the fast-run parameter set, "
                             f"which holds only {FAST_RUN_CLASS!r}")
        return self.daily_rate


@dataclass(frozen=True, eq=False)
class ScenarioSegment:
    """One segment of a scenario: its id, its LiST class, its opening balance and its daily-rate path `(H,)`."""

    segment_id: str
    liquidity_class: str
    opening: float
    daily_rates: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        rates = np.array(self.daily_rates, dtype=float)
        rates.setflags(write=False)
        object.__setattr__(self, "daily_rates", rates)


@dataclass(frozen=True, eq=False)
class Scenario:
    """A composed scenario: the family, the phase, the horizon and per segment the daily-rate path."""

    family: str
    phase: str
    horizon: int
    segments: tuple[ScenarioSegment, ...]

    @property
    def opening(self) -> np.ndarray:
        """`(S,)` the opening balances, in segment order."""
        return np.array([s.opening for s in self.segments], dtype=float)

    def rates(self) -> np.ndarray:
        """`(S, H)` the daily-rate paths."""
        return np.vstack([s.daily_rates for s in self.segments])

    def outflows(self) -> np.ndarray:
        """`(S, H)` the outflows by segment and day (`ladder.behavioural_outflows`)."""
        return _ladder.behavioural_outflows(self.opening, self.rates())


def list_scenario(severity: str) -> ListScenario:
    """The ECB LiST 2019 parameter set of a severity (`"baseline"`, `"adverse"` or `"extreme"`): exactly
    `runoff.LIST_PARAMETERS` (no stable-retail proxy for uninsured corporate balances: the fastest sight class runs at
    its
    own rate). An unknown severity raises. Golden: Adverse has `stable_retail = 0.00103258` and
    `non_operational_non_financial_corporates = 0.00688541`."""
    if severity not in _runoff.LIST_PARAMETERS:
        raise ValueError(f"severity: must be one of {tuple(_runoff.LIST_PARAMETERS)}, got {severity!r}")
    p = _runoff.LIST_PARAMETERS[severity]
    return ListScenario(severity=severity, open_maturity=dict(p["open_maturity"]),
                        contractual_outflow=dict(p["contractual_outflow"]),
                        contractual_inflow=dict(p["contractual_inflow"]), drawdown=dict(p["drawdown"]))


def compose_scenario(family: str, phase: str, parameter_set: ListScenario | FastRun,
                     segments: Mapping[str, tuple[str, float]], horizon: int) -> Scenario:
    """Compose a scenario: every segment's rate path is that of its class in the parameter set, constant over the
    horizon.

    `segments` maps a segment id to `(LiST class, opening balance)`. A class the set does not hold raises; a
    financial-customer sight class has rate 1.0 and empties the stock on day 1 (LiST para 111); the opening balances are
    not altered. The mapping of a variant to a family and phase is the named parameter `variant_scenarios`. Golden:
    `{"retail": ("other-retail", 100.0)}`, Adverse, horizon 2 gives the rates `[0.00373218, 0.00373218]` and the
    outflows `[0.373218, 0.371825...]`."""
    if family not in FAMILIES:
        raise ValueError(f"family: must be one of {FAMILIES}, got {family!r}")
    if phase not in PHASES:
        raise ValueError(f"phase: must be one of {PHASES}, got {phase!r}")
    if not isinstance(parameter_set, (ListScenario, FastRun)):
        raise ValueError(f"parameter_set: must be a ListScenario or a FastRun, got {type(parameter_set).__name__}")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon: must be a positive integer, got {horizon!r}")
    if not segments:
        raise ValueError("segments: must not be empty, got {}")
    out = []
    for segment_id, (liquidity_class, opening) in segments.items():
        rate = parameter_set.class_rate(liquidity_class)
        bal = float(check_nonnegative_stock(opening, f"opening of segment {segment_id!r}")[0])
        out.append(ScenarioSegment(segment_id=str(segment_id), liquidity_class=liquidity_class, opening=bal,
                                   daily_rates=np.full(int(horizon), rate)))
    return Scenario(family=family, phase=phase, horizon=int(horizon), segments=tuple(out))


def _row(row: Any) -> tuple[float, int, int | None]:
    """`(share, days, year or None)` of a Table A.1 row given as a pair or as an `ObservedRun`."""
    if isinstance(row, _runoff.ObservedRun):
        return row.outflow_share, row.days, row.year
    try:
        share, days = row
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rows: each row must be (outflow share, days), got {row!r}") from exc
    return share, days, None


def fast_run_scenario(rows: Sequence[tuple[float, int]], rule: str = "maximum-implied-daily-rate", *,
                      scored: bool = False) -> FastRun:
    """The daily rate of the concentrated uninsured segment from observed runs (Table A.1 `(outflow share, days)` pairs
    the caller selects): each row's rate is `daily_rate_from_cumulative`; the fitted rate is the rule's statistic
    (`maximum-implied-daily-rate`, `median-implied-daily-rate` or `mean-implied-daily-rate`, the named parameter
    `fast_run_calibration_rule`). The dossiers give no calibration method (UNVERIFIED), so the rule is a convention.

    Calibration on the 2023 rows is refused with a message when the caller marks the rows `scored` (decision Q4: the
    scored variant is calibrated on pre-2023 rows only); a row is recognised as a 2023 row by its `ObservedRun` year or
    by its `(share, days)` pair equal to a 2023 row of `TABLE_A1`. Plain pairs are matched by exact float equality
    (`(0.850000001, 2)` passes as a pre-2023 row), so callers should pass `ObservedRun` rows. Golden: Icesave, Northern
    Rock, WaMu and IndyMac give `[0.002971, 0.054258, 0.010481, 0.005938]` and `0.054258` (maximum), `0.0082095`
    (median), `0.018412` (mean)."""
    if rule not in FAST_RUN_RULES:
        raise ValueError(f"rule: must be one of {FAST_RUN_RULES}, got {rule!r}")
    if len(rows) == 0:
        raise ValueError("rows: must not be empty, got []")
    a2023 = {(r.outflow_share, r.days) for r in _runoff.TABLE_A1 if r.year == 2023}
    pairs: list[tuple[float, int]] = []
    for row in rows:
        share, days, year = _row(row)
        if days != int(days) or int(days) < 1:
            raise ValueError(f"rows: days must be positive integers, got {days!r}")
        share = float(check_unit_share(share, "rows")[0])
        if scored and (year == 2023 or (year is None and (share, int(days)) in a2023)):
            raise ValueError(f"rows: the 2023 rows of Table A.1 cannot calibrate a scored variant (they are scored "
                             f"against, decision Q4), got the row ({share}, {int(days)})")
        pairs.append((share, int(days)))
    implied = tuple(float(_runoff.daily_rate_from_cumulative(s, d)) for s, d in pairs)
    stat = {"maximum-implied-daily-rate": np.max, "median-implied-daily-rate": np.median,
            "mean-implied-daily-rate": np.mean}[rule]
    return FastRun(rule=rule, daily_rate=float(stat(implied)), implied_rates=implied, rows=tuple(pairs))


def anchor_from_observation(outflow_share: float, days: int) -> dict[str, float | int | bool]:
    """The anchor row of decision Q4: an observed outflow and the daily rate it implies. `forecast_claim` is always
    `False` (the observation is never a forecast); the rate round-trips through `cumulative_outflow_constant`. Golden:
    `(0.85, 2)` gives `implied_daily_rate = 0.612702`."""
    share = float(check_unit_share(outflow_share, "outflow_share")[0])
    if days != int(days) or int(days) < 1:
        raise ValueError(f"days: must be a positive integer, got {days!r}")
    return {"outflow_share": share, "days": int(days),
            "implied_daily_rate": float(_runoff.daily_rate_from_cumulative(share, int(days))),
            "forecast_claim": False}


def uninsured_share_grid(insured: float, uninsured: float, shares: Sequence[float]) -> list[tuple[float, float]]:
    """One `(insured, uninsured)` pair per share of the grid (dossier 05 line 278), the total `insured + uninsured`
    preserved and `uninsured / total = share`; shares lie in `[0, 1]` and increase. `insured` and `uninsured` are the
    call report's DEPINS and DEPUNA. Golden: `(40, 60)` with `[0.5, 0.9]` gives `[(50, 50), (10, 90)]`."""
    ins = float(check_nonnegative_stock(insured, "insured")[0])
    un = float(check_nonnegative_stock(uninsured, "uninsured")[0])
    total = ins + un
    if total <= 0.0:
        raise ValueError(f"insured and uninsured: the total must be positive, got {total}")
    grid = check_increasing_grid(check_unit_share(shares, "shares"), "shares")
    out = []
    for share in grid:
        uninsured_part = total * float(share)
        out.append((total - uninsured_part, uninsured_part))
    return out


def htm_loss_grid(reported_loss: float, multiples: Sequence[float]) -> list[float]:
    """The held-to-maturity loss at each multiple of the reported loss (`SCHF - SCHA`, negative): `reported_loss x
    multiple` (dossier 05 line 278). A multiple of 0 removes the loss and 1 is the reported loss; multiples are
    non-negative and increase. Golden: `-15` with `[0, 1, 2]` gives `[0, -15, -30]`."""
    try:
        loss = float(reported_loss)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reported_loss: must be numeric, got {reported_loss!r}") from exc
    if not np.isfinite(loss):
        raise ValueError(f"reported_loss: must be finite, got {reported_loss!r}")
    grid = check_increasing_grid(check_nonnegative_stock(multiples, "multiples"), "multiples", minimum=0.0)
    return [loss * float(m) for m in grid]


__all__ = ["FAMILIES", "FAST_RUN_CLASS", "FAST_RUN_RULES", "FastRun", "ListScenario", "PHASES", "Scenario",
           "ScenarioSegment", "anchor_from_observation", "compose_scenario", "fast_run_scenario", "htm_loss_grid",
           "list_scenario", "uninsured_share_grid"]
