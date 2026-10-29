"""The e-process record and the readings every sequential monitor shares.

An *e-process* `E_t` is a non-negative process that starts at 1 and whose expected value under the null hypothesis
does not exceed 1 at any stopping time (a test supermartingale). Ville's inequality bounds the chance that it ever
reaches `1 / alpha_level` by `alpha_level`: the monitor that rejects the first time `E_t >= 1 / alpha_level` has size
at most `alpha_level` however often, and for however long, the analyst looks (anytime-valid size), where a fixed-window
test repeated every day does not. The statistics are kept as `log E_t` so a long series cannot overflow.

Sources: Ville (1939), Etude critique de la notion de collectif, Gauthier-Villars; Ramdas, Grunwald, Vovk and Shafer
(2023), Game-theoretic statistics and safe anytime-valid inference, Statistical Science 38(4):576-601, arXiv 2210.01948;
Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991.

A reading is evidence, not a verdict on compliance: `to_test_result` maps the record to the engine's `TestResult`
(red when the monitor rejected, green otherwise, `not_evaluated` for an empty series) without registering a test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from riskvalidation.core.result import Light, TestResult
from riskvalidation.validation.sequential._core import VERSION_LABEL, check_alpha_level, frozen

_MAX_EXP = 700.0  # exp(700) is near the largest power of e a double holds


@dataclass(frozen=True, eq=False)
class EProcessResult:
    """One monitored quantity: the path `log E_t`, `t = 1..n`, and its readings.

    `log_e` is the natural log of the e-process (dimensionless, one value per time step); `alpha_level` the size of the
    monitor (default 0.05, threshold 20); `h0` the null; `reference` the source of the construction; `extras` carries
    the parameters used (the alternatives, their weights) and any component detail. Times are 1-based counts of
    observations (days, vintages or years, as the caller's series are).
    """

    monitor_id: str
    log_e: np.ndarray
    alpha_level: float
    h0: str
    reference: str
    notes: str = ""
    inputs_hash: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_alpha_level(self.alpha_level)
        object.__setattr__(self, "log_e", frozen(self.log_e))
        if self.log_e.ndim != 1:
            raise ValueError("log_e must be one-dimensional")

    @property
    def n(self) -> int:
        """The number of observations the path covers."""
        return int(self.log_e.size)

    @property
    def threshold(self) -> float:
        """`1 / alpha_level`: the level at which the monitor rejects (20 for 0.05)."""
        return 1.0 / self.alpha_level

    @property
    def max_log_e(self) -> float:
        """The supremum of `log E_t` over the path (NaN for an empty path)."""
        return float(self.log_e.max()) if self.n else float("nan")

    @property
    def rejection_time(self) -> int | None:
        """The first `t` (1-based) with `E_t >= 1 / alpha_level`, or `None`."""
        return detection_time(self.log_e, self.alpha_level)

    @property
    def rejected(self) -> bool:
        """Whether the monitor crossed its threshold at any time of the path."""
        return self.rejection_time is not None

    @property
    def anytime_p_value(self) -> float | None:
        """`min(1, 1 / sup_t E_t)`, a p-value valid at any stopping time (Ville); `None` for an empty path."""
        return anytime_p_value(self.log_e)

    @property
    def e_process(self) -> np.ndarray:
        """`E_t` itself (clipped at `exp(700)`), for plotting; use `log_e` for arithmetic."""
        return np.exp(np.minimum(self.log_e, _MAX_EXP))

    def to_test_result(self, model_id: str | None = None, segment: str | None = None) -> TestResult:
        """The engine's result record: red when the monitor rejected, green otherwise, `not_evaluated` when empty.

        statistic = `sup E_t` (clipped at `exp(700)`), p_value = the anytime-valid p-value, metric = the final `E_n`.
        The monitor ids are registered in the test catalogue (`sequential.var_hits`, `sequential.var_hit_clustering`,
        `sequential.pd_grades`, `sequential.battery`).
        """
        kw: dict[str, Any] = {
            "test_id": self.monitor_id, "h0": self.h0, "reference": self.reference, "policy_version": VERSION_LABEL,
            "alternative": "the null fails; the e-process grows without bound in expectation",
            "alpha_red": self.alpha_level, "model_id": model_id, "segment": segment, "n": self.n,
            "inputs_hash": self.inputs_hash,
            "extras": {**dict(self.extras), "threshold": self.threshold, "rejection_time": self.rejection_time},
        }
        if self.n == 0:
            return TestResult(light=Light.NOT_EVALUATED, statistic=float("nan"), notes="not evaluated: no observation",
                              **kw)
        return TestResult(
            light=Light.RED if self.rejected else Light.GREEN, statistic=math.exp(min(self.max_log_e, _MAX_EXP)),
            p_value=self.anytime_p_value, metric=math.exp(min(float(self.log_e[-1]), _MAX_EXP)), notes=self.notes,
            **kw)


@dataclass(frozen=True, eq=False)
class GradeEProcessResult(EProcessResult):
    """An `EProcessResult` for a rating scale: `log_e` is the combined path and `grade_log_e` the path of each grade.

    `grade_log_e` has shape (steps, grades); `grades` labels its columns; `combine` is `"product"` or `"average"`.
    """

    grade_log_e: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    grades: tuple[str, ...] = ()
    combine: str = "product"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "grade_log_e", frozen(self.grade_log_e))

    def grade_rejection_times(self) -> dict[str, int | None]:
        """For each grade, the first `t` its own e-process crossed the threshold (a grade-level reading, not
        corrected for looking at several grades)."""
        return {g: detection_time(self.grade_log_e[:, j], self.alpha_level) for j, g in enumerate(self.grades)}


def detection_time(log_e: np.ndarray, alpha_level: float = 0.05) -> int | None:
    """The first time `t` (1-based) with `log E_t >= log(1 / alpha_level)`, or `None` when the path never crosses.

    `log_e` is a path of natural logs (dimensionless); `alpha_level` the size of the monitor. The detection delay of a
    planted defect is this time minus the time the defect starts.
    """
    a = check_alpha_level(alpha_level)
    path = np.asarray(log_e, dtype=float)
    if path.ndim != 1:
        raise ValueError("log_e must be one-dimensional")
    hit = np.flatnonzero(path >= -math.log(a))
    return int(hit[0]) + 1 if hit.size else None


def anytime_p_value(log_e: np.ndarray) -> float | None:
    """`min(1, exp(-sup_t log E_t))`: the smallest level at which the monitor would have rejected by now, valid at any
    stopping time (Ville's inequality). `None` for an empty path."""
    path = np.asarray(log_e, dtype=float)
    if path.size == 0:
        return None
    return float(min(1.0, math.exp(-max(0.0, float(path.max())))))


__all__ = [
    "EProcessResult",
    "GradeEProcessResult",
    "anytime_p_value",
    "detection_time",
]
