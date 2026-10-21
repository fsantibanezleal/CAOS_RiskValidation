"""Credit-loss generators of the IFRS 9 and LGD back-tests (U16-E2, BL-088).

- ``StageTransitions``: one-year stage transitions of three IFRS 9 stages, counts drawn row by row from a multinomial
  whose probabilities are the predicted matrix plus a stated departure (``shift``; zero is the null).
- ``LgdCalibration``: realised LGDs drawn as Beta variables whose mean is the predicted LGD plus a bias, in
  ``n_periods`` default years of ``n_per_period`` workouts each (a bias in the ``biased_periods`` only).
- ``DownturnLgd``: the realised LGDs of the defaults of a downturn period, Beta with mean ``estimate + shortfall``.

The Beta draws have a concentration ``concentration`` (the sum of both shape parameters): 2 gives the wide, bimodal
spread of workout LGDs (a standard deviation near 0.28), kept inside [0, 1]; the mean gap is exact by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from riskvalidation.generators._base import (
    Check,
    GeneratorBase,
    frozen_array,
    mean_check,
    variance_check,
)


def _int_at_least(name: str, value: int, minimum: int) -> int:
    if int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, got {value}")
    return int(value)


def _beta(rng: np.random.Generator, mean: np.ndarray, concentration: float) -> np.ndarray:
    return rng.beta(mean * concentration, (1.0 - mean) * concentration)


@dataclass(frozen=True)
class StageTransitions(GeneratorBase):
    """``row_n[i]`` accounts start the year in stage ``i + 1``; their destinations are multinomial with the row
    ``matrix[i] + shift[i]`` (``shift`` rows add to zero; ``None`` is the null, the predicted matrix itself)."""

    name: ClassVar[str] = "stage_transitions"
    row_n: tuple[int, int, int]
    matrix: tuple
    shift: tuple | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_n", tuple(_int_at_least("row_n", n, 1) for n in self.row_n))
        if len(self.row_n) != 3:
            raise ValueError("row_n needs one count per stage (3)")
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (3, 3) or np.any(m < 0.0) or not np.allclose(m.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError("matrix must be a row-stochastic (3, 3) matrix")
        object.__setattr__(self, "matrix", frozen_array(m, ndim=2))
        if self.shift is not None:
            s = np.asarray(self.shift, dtype=float)
            if s.shape != (3, 3) or not np.allclose(s.sum(axis=1), 0.0, atol=1e-12):
                raise ValueError("shift must be (3, 3) with rows adding to zero")
            if np.any(m + s < 0.0) or np.any(m + s > 1.0):
                raise ValueError("matrix + shift must stay a probability")
            object.__setattr__(self, "shift", frozen_array(s, ndim=2))

    @property
    def predicted(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @property
    def true_matrix(self) -> np.ndarray:
        return self.predicted + (0.0 if self.shift is None else np.asarray(self.shift, dtype=float))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        t = self.true_matrix
        counts = np.array([rng.multinomial(n, t[i]) for i, n in enumerate(self.row_n)], dtype=float)
        return {"predicted_matrix": self.predicted, "observed_counts": counts}

    def truth(self) -> dict[str, Any]:
        p, t = self.predicted, self.true_matrix
        n = np.asarray(self.row_n, dtype=float)[:, None]
        pos = p > 0.0
        noncentrality = float(np.sum(n * (t - p) ** 2 / np.where(pos, p, 1.0), where=pos))
        return {"null_holds": self.shift is None or not np.any(np.asarray(self.shift)), "degrees_of_freedom": 6,
                "noncentrality": noncentrality, "smallest_expected_count": float(np.min(n * p)),
                "true_matrix": t.tolist()}

    def verify(self, n_draws: int = 4_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 421]))
        counts = np.array([self.draw(rng)["observed_counts"] for _ in range(n_draws)])
        share = counts / np.asarray(self.row_n, dtype=float)[None, :, None]
        t = self.true_matrix
        return [mean_check(f"transition {i + 1} to {j + 1}", t[i, j], share[:, i, j])
                for i in range(3) for j in range(3)]


@dataclass(frozen=True)
class LgdCalibration(GeneratorBase):
    """Workouts in ``n_periods`` default years, ``n_per_period`` each: predicted LGD uniform on ``predicted_range``,
    realised LGD Beta with mean ``predicted + bias`` where the year is in ``biased_periods`` (``None``: every year)."""

    name: ClassVar[str] = "lgd_calibration"
    n_per_period: int
    n_periods: int = 1
    bias: float = 0.0
    biased_periods: tuple[int, ...] | None = None
    predicted_range: tuple[float, float] = (0.15, 0.65)
    concentration: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_per_period", _int_at_least("n_per_period", self.n_per_period, 2))
        object.__setattr__(self, "n_periods", _int_at_least("n_periods", self.n_periods, 1))
        lo, hi = (float(x) for x in self.predicted_range)
        if not 0.0 < lo < hi < 1.0:
            raise ValueError("predicted_range must satisfy 0 < low < high < 1")
        object.__setattr__(self, "predicted_range", (lo, hi))
        if not float(self.concentration) > 0.0:
            raise ValueError("concentration must be positive")
        if self.biased_periods is not None:
            object.__setattr__(self, "biased_periods", tuple(int(k) for k in self.biased_periods))
            if any(not 0 <= k < self.n_periods for k in self.biased_periods):
                raise ValueError("biased_periods must index the periods")
        if lo + min(self.bias, 0.0) <= 0.0 or hi + max(self.bias, 0.0) >= 1.0:
            raise ValueError("predicted_range plus the bias must stay inside (0, 1)")

    def _biased(self) -> np.ndarray:
        periods = range(self.n_periods) if self.biased_periods is None else self.biased_periods
        flags = np.zeros(self.n_periods, dtype=bool)
        flags[list(periods)] = True
        return flags

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """The predicted LGDs, then the realised ones; ``period`` is the default year of each workout."""
        n = self.n_per_period * self.n_periods
        period = np.repeat(np.arange(self.n_periods, dtype=np.int64), self.n_per_period)
        predicted = rng.uniform(*self.predicted_range, size=n)
        mean = predicted + self.bias * self._biased()[period]
        return {"realised": _beta(rng, mean, self.concentration), "predicted": predicted, "period": period}

    def gap_variance(self, biased: bool) -> float:
        """Var(realised - predicted) for a workout of a biased (or unbiased) year, exact: E[m (1 - m)] / (c + 1)."""
        lo, hi = self.predicted_range
        shift = self.bias if biased else 0.0
        a, b = lo + shift, hi + shift
        e_m, e_m2 = (a + b) / 2.0, (a * a + a * b + b * b) / 3.0
        return float((e_m - e_m2) / (self.concentration + 1.0))

    def truth(self) -> dict[str, Any]:
        sd = math.sqrt(self.gap_variance(True))
        return {"null_holds": self.bias == 0.0, "mean_gap_by_period": (self.bias * self._biased()).tolist(),
                "sd_of_gap": sd, "t_noncentrality_one_period": self.bias / sd * math.sqrt(self.n_per_period)}

    def verify(self, n_draws: int = 2_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 422]))
        gaps = []
        for _ in range(n_draws):
            d = self.draw(rng)
            gaps.append((d["realised"] - d["predicted"]).reshape(self.n_periods, self.n_per_period))
        g = np.concatenate(gaps, axis=1)
        flags = self._biased()
        out = []
        for k in range(self.n_periods):
            out.append(mean_check(f"mean gap, period {k}", self.bias * flags[k], g[k]))
        out.append(variance_check("variance of the gap, a biased year" if flags.any() else "variance of the gap",
                                  self.gap_variance(bool(flags.any())), g[int(np.argmax(flags))]))
        return out


@dataclass(frozen=True)
class DownturnLgd(GeneratorBase):
    """``n`` downturn workouts: realised LGD Beta with mean ``estimate + shortfall`` against the estimate
    ``estimate`` (``shortfall = 0`` is the boundary of the one-sided null)."""

    name: ClassVar[str] = "downturn_lgd"
    n: int
    estimate: float = 0.45
    shortfall: float = 0.0
    concentration: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _int_at_least("n", self.n, 2))
        if not 0.0 < float(self.estimate) + float(self.shortfall) < 1.0 or not 0.0 < float(self.estimate) < 1.0:
            raise ValueError("estimate and estimate + shortfall must lie in (0, 1)")
        if not float(self.concentration) > 0.0:
            raise ValueError("concentration must be positive")

    @property
    def true_mean(self) -> float:
        return float(self.estimate + self.shortfall)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        mean = np.full(self.n, self.true_mean)
        return {"estimate": float(self.estimate), "realised_downturn": _beta(rng, mean, self.concentration)}

    def truth(self) -> dict[str, Any]:
        m = self.true_mean
        sd = math.sqrt(m * (1.0 - m) / (self.concentration + 1.0))
        return {"null_holds": self.shortfall <= 0.0, "true_mean": m, "sd": sd,
                "t_noncentrality": self.shortfall / sd * math.sqrt(self.n)}

    def verify(self, n_draws: int = 4_000, seed: int = 20261021) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 423]))
        x = np.concatenate([self.draw(rng)["realised_downturn"] for _ in range(n_draws)])
        tr = self.truth()
        return [mean_check("mean realised downturn LGD", tr["true_mean"], x),
                variance_check("variance of the realised downturn LGD", tr["sd"] ** 2, x)]


__all__ = ["DownturnLgd", "LgdCalibration", "StageTransitions"]
