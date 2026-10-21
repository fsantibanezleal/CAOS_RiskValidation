"""The depositor generator with known truth: a systemic two-state Markov chain over days and, per segment, a withdrawal
hazard by the state of the day.

Source: dossier 08 G.2 line 698 (a depositor generator with known truth) and G.4 lines 739 to 742 (the generator's
verification); the generator contract of `generators._base` (a frozen configuration, one dataset per seed, an exact
truth,
a verifier). Contract `engine-liquidity-api` 2.0.0, module deposits; placement in the engine, option (a), no
concentration parameter (decision Q10).

The mechanism: the state is calm (0) on day 1; per day it moves calm to stress with probability `p_enter` and stress to
calm with `p_exit`, one chain shared by all depositors; a depositor of a segment still deposited on day `d` withdraws
with
hazard `h_calm` or `h_stress` by the state of day `d`; balances are equal within a segment, so the cumulative outflow
share is the share of depositors who have withdrawn. The stress-state transition law has no dossier basis: it is this
contract's convention (UNVERIFIED). Because the chain is shared by all depositors, `p_enter_stress` and
`p_exit_stress` are fields of the configuration, not of a segment (BL-082, `engine-liquidity-api` 2.0.0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from riskvalidation.generators._base import Check, GeneratorBase, jsonable, mean_check
from riskvalidation.liquidity._checks import check_unit_share, check_weights_sum_to_one
from riskvalidation.liquidity.runoff import SEGMENTS


@dataclass(frozen=True)
class DepositorSegment:
    segment: str                 # one of SEGMENTS
    share_of_deposits: float     # share of the call-report deposits, in [0, 1]
    calm_daily_hazard: float     # in [0, 1]
    stress_daily_hazard: float   # in [0, 1]

    def __post_init__(self) -> None:
        if self.segment not in SEGMENTS:
            raise ValueError(f"segment: must be one of {SEGMENTS}, got {self.segment!r}")
        for name in ("share_of_deposits", "calm_daily_hazard", "stress_daily_hazard"):
            check_unit_share(getattr(self, name), name)


@dataclass(frozen=True)
class DepositorConfig:
    segments: tuple[DepositorSegment, ...]
    n_depositors: int            # per segment, positive
    horizon_days: int            # positive
    p_enter_stress: float        # per day calm to stress, in [0, 1]; one chain shared by every depositor
    p_exit_stress: float         # per day stress to calm, in [0, 1]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        object.__setattr__(self, "segments", segments)
        if not segments:
            raise ValueError("segments: must not be empty, got ()")
        ids = [s.segment for s in segments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"segment: each segment id may appear once, got {ids}")
        for name in ("p_enter_stress", "p_exit_stress"):
            check_unit_share(getattr(self, name), name)
        for name in ("n_depositors", "horizon_days"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name}: must be a positive integer, got {value!r}")
        check_weights_sum_to_one(np.array([s.share_of_deposits for s in segments]), "share_of_deposits")


class DepositorGenerator(GeneratorBase):
    """`draw` returns `{"state": (H,) int, "withdrawn_day": {segment: (n,) int, 0 = not within the horizon},
    "outflow_rate": {segment: (H,) cumulative share}}`; `truth` returns `{"outflow_rate": {segment: (H,) exact expected
    cumulative outflow share by horizon}}`; `verify` returns the `Check` list of `generators._base`."""

    name = "depositors"

    def __init__(self, config: DepositorConfig) -> None:
        self.spec = config

    def config(self) -> dict[str, Any]:
        """The generator's name, the package version and every parameter, JSON-safe."""
        from riskvalidation import __version__

        return {"generator": self.name, "riskvalidation": __version__, "parameters": jsonable(asdict(self.spec))}

    def _chain(self, rng: np.random.Generator) -> np.ndarray:
        """The calm (0) or stress (1) state of each day, calm on day 1."""
        h = self.spec.horizon_days
        cfg = self.spec
        u = rng.random(h - 1) if h > 1 else np.empty(0)
        state = np.zeros(h, dtype=np.int64)
        for d in range(1, h):
            state[d] = (1 if u[d - 1] < cfg.p_enter_stress else 0) if state[d - 1] == 0 else (
                0 if u[d - 1] < cfg.p_exit_stress else 1)
        return state

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        cfg = self.spec
        state = self._chain(rng)
        withdrawn: dict[str, np.ndarray] = {}
        outflow: dict[str, np.ndarray] = {}
        for seg in cfg.segments:
            hazard = np.where(state == 0, seg.calm_daily_hazard, seg.stress_daily_hazard)
            cumulative = 1.0 - np.cumprod(1.0 - hazard)
            u = rng.random(cfg.n_depositors)
            idx = np.searchsorted(cumulative, u, side="right")  # first day with P(withdrawn by then) above u
            day = np.where(idx >= cfg.horizon_days, 0, idx + 1).astype(np.int64)
            counts = np.bincount(day[day > 0], minlength=cfg.horizon_days + 1)[1:]
            withdrawn[seg.segment] = day
            outflow[seg.segment] = np.cumsum(counts) / cfg.n_depositors
        return {"state": state, "withdrawn_day": withdrawn, "outflow_rate": outflow}

    def truth(self) -> dict[str, Any]:
        """The exact expected cumulative outflow share by horizon, by the transfer recursion `u_1 = (1 - h_calm, 0)`,
        `u_{d+1} = (u_d P) elementwise (1 - h_calm, 1 - h_stress)`, `truth(n) = 1 - sum(u_n)`."""
        out: dict[str, np.ndarray] = {}
        cfg = self.spec
        p = np.array([[1.0 - cfg.p_enter_stress, cfg.p_enter_stress], [cfg.p_exit_stress, 1.0 - cfg.p_exit_stress]])
        for seg in cfg.segments:
            keep = np.array([1.0 - seg.calm_daily_hazard, 1.0 - seg.stress_daily_hazard])
            u = np.array([keep[0], 0.0])
            path = np.empty(self.spec.horizon_days)
            path[0] = 1.0 - u.sum()
            for d in range(1, self.spec.horizon_days):
                u = (u @ p) * keep
                path[d] = 1.0 - u.sum()
            out[seg.segment] = path
        return {"outflow_rate": out}

    def verify(self, n_draws: int = 20_000, seed: int = 20261005) -> list[Check]:
        """A large seeded sample on its own stream against the exact truth: the mean cumulative outflow share of each
        segment at the first day, the middle day and the last day of the horizon, with the standard error of a mean."""
        if not isinstance(n_draws, (int, np.integer)) or n_draws < 2:
            raise ValueError(f"n_draws: must be an integer of at least 2, got {n_draws!r}")
        rng = np.random.default_rng(seed)
        h = self.spec.horizon_days
        samples = {seg.segment: np.empty((n_draws, h)) for seg in self.spec.segments}
        for i in range(n_draws):
            for key, rate in self.draw(rng)["outflow_rate"].items():
                samples[key][i] = rate
        truth = self.truth()["outflow_rate"]
        horizons = sorted({1, math.ceil(h / 2), h})
        return [mean_check(f"{key}@{d}", float(truth[key][d - 1]), samples[key][:, d - 1])
                for key in samples for d in horizons]


__all__ = ["DepositorConfig", "DepositorGenerator", "DepositorSegment"]
