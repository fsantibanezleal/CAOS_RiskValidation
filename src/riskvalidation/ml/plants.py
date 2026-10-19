"""Planted-defect generators for the size and power of the ML validation tests (dossier 07 D pattern 6, l.611-616:
synthetic data with a known ground truth, a verifier and a difficulty knob; the Method ladder of the product SDD l.335).

Each plant is a frozen ``GeneratorBase`` configuration with ``config()``, ``draw(rng)`` (one dataset keyed by the names
of the arguments it feeds), an exact ``truth()`` and a ``verify()`` whose large draw reproduces the truth within Monte
Carlo error. The ``seed`` of a plant is the default seed of ``sample()`` and, where the plant has a training seed
(``SeedUnstablePlant``), part of every fit's random stream. Built on ``generators.LogisticPortfolio`` (covariates,
shift, misspecification) and on the construction of ``generators.BinormalScores`` (a leak at the feature level). The
design constants of each plant (the portfolio, the prevalence, the separation, the Beta law of the PDs) are stated in
its docstring and in ``docs/ml/08_size-and-power.md``; they are the plants' own, not regulatory values. Contract:
``engine-ml-api`` 1.0.0, module ``riskvalidation/ml/plants.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import integrate, optimize, special, stats

from riskvalidation.generators._base import Check, GeneratorBase, frozen_array, mean_check
from riskvalidation.generators.scores import LogisticPortfolio
from riskvalidation.ml import conformal, robustness
from riskvalidation.validation.discrimination import auc_with_variance

PORTFOLIO = {"intercept": -1.5, "coefficients": (0.8, 0.5, 0.3), "correlation": 0.3}  # about 20 percent defaults
SHIFT_PORTFOLIO = {
    "intercept": 0.0,
    "coefficients": (1.6, 1.0, 0.6),
    "correlation": 0.3,
}  # the shift plant: AUC about 0.85
SHIFTED_SHARE = 0.25  # the share of evaluation rows that come from the shifted region
BREAK_AT, BREAK_SLOPE = 1.5, 2.0  # beyond x_0 = 1.5 the true logit loses 2 per unit: its slope 1.6 reverses to -0.4
PREVALENCE = 0.3  # the share of defaulters of the noise and seed plants
SEPARATION = 1.5  # the standardised separation of the signal column of the seed plant
ALPHA = 0.10  # the miscoverage of the conformal check in the C20 shift kind (a convention of the contract)
BETA_A, BETA_B = 2.0, 5.0  # the law of the PDs of the exchangeability plant
CAL_FACTOR = 10  # the calibration window of the exchangeability plant is this many times the test window
DEFAULT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def _positive_int(n: Any, name: str, minimum: int) -> int:
    if isinstance(n, bool) or int(n) != n or n < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, got {n}")
    return int(n)


def _nonneg(x: Any, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a nonnegative number, got {x}")
    return v


class _Plant(GeneratorBase):
    """The shared part of the plants: the default draw of ``sample()``."""

    seed: int

    def sample(self) -> dict[str, Any]:
        """One dataset on the plant's own seed."""
        return self.draw(np.random.default_rng(self.seed))  # type: ignore[attr-defined]


def _portfolio(n: int, **kw: Any) -> LogisticPortfolio:
    return LogisticPortfolio(n=n, **PORTFOLIO, **kw)


@dataclass(frozen=True)
class LeakPlant(_Plant):
    """A logistic portfolio of ``n`` obligors with three features and a leaking fourth column ``y * leak_strength + N(0,
    1)`` (the construction of ``BinormalScores.leak`` at the feature level). ``leak_strength`` is the separation in
    noise standard deviations, so the single-feature AUC of the leak is exactly ``Phi(leak_strength / sqrt 2)``: 0.5 at
    strength 0 (the null) and above 0.95 from 2.3. ``truth()`` names the leaking column."""

    name: ClassVar[str] = "ml_leak_plant"
    n: int
    leak_strength: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _positive_int(self.n, "n", 20))
        object.__setattr__(self, "leak_strength", _nonneg(self.leak_strength, "leak_strength"))

    @property
    def _p(self) -> int:
        return len(PORTFOLIO["coefficients"])

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        d = _portfolio(self.n).draw(rng)
        y = d["defaults"]
        leak = self.leak_strength * y + rng.standard_normal(self.n)
        return {"X": np.column_stack([d["x"], leak]), "y": y, "defaults": y, "leak_column": self._p}

    def truth(self) -> dict[str, Any]:
        return {
            "leak_column": self._p,
            "leak_strength": self.leak_strength,
            "n_features": self._p + 1,
            "leak_auc": float(special.ndtr(self.leak_strength / math.sqrt(2.0))),
        }

    def verify(self, n_draws: int = 500, seed: int = 20261005) -> list[Check]:
        """The single-feature AUC of the leak column against its exact value, and the default rate."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1401]))
        aucs, rates = [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            aucs.append(auc_with_variance(d["X"][:, d["leak_column"]], d["y"])[0])
            rates.append(d["y"].mean())
        return [
            mean_check("single-feature AUC of the leak", self.truth()["leak_auc"], np.array(aucs)),
            mean_check("default rate", _portfolio(self.n).truth()["mean_pd_true"], np.array(rates)),
        ]


@dataclass(frozen=True)
class CovariateShiftPlant(_Plant):
    """A development sample and an evaluation sample of ``n`` obligors each. A quarter of the evaluation rows
    (``SHIFTED_SHARE``, placed at random) come from a shifted region: their covariates in ``features`` are shifted by
    ``shift`` standard deviations (shift 0 is the null: no row is shifted). The true logit follows the linear model
    inside the development support and its slope in the first covariate reverses beyond ``BREAK_AT`` (a regime the
    development sample barely contains), so the shifted rows are the ones the model ranks badly: the worst subsample by
    shift score is the shifted quarter, and its AUC collapses. ``predict`` is the linear model, valid where the
    development sample lives. The portfolio is the stronger one of ``SHIFT_PORTFOLIO`` (about 50 percent defaults, AUC
    about 0.85); ``draw`` also returns the boolean mask ``shifted`` of the evaluation rows."""

    name: ClassVar[str] = "ml_covariate_shift_plant"
    n: int
    shift: float
    features: Any
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _positive_int(self.n, "n", 20))
        if not math.isfinite(float(self.shift)):
            raise ValueError("shift must be finite")
        object.__setattr__(self, "shift", float(self.shift))
        feats = tuple(int(f) for f in self.features)
        p = len(SHIFT_PORTFOLIO["coefficients"])
        if not feats or any(not 0 <= f < p for f in feats) or len(set(feats)) != len(feats):
            raise ValueError(f"features must be distinct column indices in [0, {p}), got {list(self.features)}")
        object.__setattr__(self, "features", frozen_array(feats, dtype=int))

    def _shift_vector(self) -> np.ndarray:
        v = np.zeros(len(SHIFT_PORTFOLIO["coefficients"]))
        v[list(self.features)] = self.shift
        return v

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The linear model: ``expit(intercept + X @ coefficients)``, valid inside the development support."""
        beta = np.asarray(SHIFT_PORTFOLIO["coefficients"])
        return special.expit(SHIFT_PORTFOLIO["intercept"] + np.asarray(X, dtype=float) @ beta)

    def _labels(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        beta = np.asarray(SHIFT_PORTFOLIO["coefficients"])
        eta = SHIFT_PORTFOLIO["intercept"] + x @ beta - BREAK_SLOPE * np.clip(x[:, 0] - BREAK_AT, 0.0, None)
        return (rng.random(len(x)) < special.expit(eta)).astype(np.int64)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        x_dev = LogisticPortfolio(n=self.n, **SHIFT_PORTFOLIO).draw(rng)["x"]
        x_eval = LogisticPortfolio(n=self.n, **SHIFT_PORTFOLIO).draw(rng)["x"]
        shifted = np.zeros(self.n, dtype=bool)
        shifted[rng.permutation(self.n)[: round(SHIFTED_SHARE * self.n)]] = True
        if self.shift != 0.0:
            x_eval[shifted] += self._shift_vector()
        else:
            shifted[:] = False
        return {
            "X_dev": x_dev,
            "y_dev": self._labels(x_dev, rng),
            "X_eval": x_eval,
            "y_eval": self._labels(x_eval, rng),
            "shifted": shifted,
        }

    def truth(self) -> dict[str, Any]:
        share = SHIFTED_SHARE if self.shift != 0.0 else 0.0
        return {
            "shifted_features": [int(f) for f in self.features],
            "shift": self.shift,
            "null": self.shift == 0.0,
            "shifted_share": share,
            "eval_mean": [float(share * m) for m in self._shift_vector()],
            "break_at": BREAK_AT,
            "break_slope": BREAK_SLOPE,
        }

    def verify(self, n_draws: int = 500, seed: int = 20261005) -> list[Check]:
        """The mean of every evaluation covariate against the planted shift (development covariates are mean zero), the
        share of shifted rows and the mean of the rows that were shifted."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1402]))
        dev, ev, share, hit = [], [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            dev.append(d["X_dev"].mean(axis=0))
            ev.append(d["X_eval"].mean(axis=0))
            share.append(d["shifted"].mean())
            if d["shifted"].any():
                hit.append(d["X_eval"][d["shifted"]].mean(axis=0) - d["X_eval"][~d["shifted"]].mean(axis=0))
        dev_a, ev_a = np.array(dev), np.array(ev)
        t = self.truth()
        out = [
            mean_check(f"evaluation mean of feature {j}", t["eval_mean"][j], ev_a[:, j]) for j in range(ev_a.shape[1])
        ]
        out.append(mean_check("development mean of feature 0", 0.0, dev_a[:, 0]))
        if hit:
            out.append(
                mean_check(
                    "shifted minus other rows, first shifted feature", self.shift, np.array(hit)[:, self.features[0]]
                )
            )
        return out


@dataclass(frozen=True)
class NoiseSensitivityPlant(_Plant):
    """Defaulters (prevalence 0.3) and survivors separated by the signal column ``x_0 = sensitivity * y + N(0, 1)`` and
    a pure-noise column ``x_1``; the model ``predict`` is monotone in ``x_0``. With the noise of ``robustness_curve``
    (scale ``eps`` times the column standard deviation) the AUC at scale eps is exactly
    ``Phi(sensitivity / sqrt(2 (1 + eps^2 (1 + sensitivity^2 pi (1 - pi)))))``: ``truth()`` gives that curve and its
    delta against scale 0. ``sensitivity`` is the standardised separation of the fragile column: the larger it is, the
    larger the share of the column's variance that is signal and the more AUC a given eps destroys (sensitivity 0: no
    signal to lose, AUC 0.5 at every scale). The contract's "AUC loss per unit eps" is therefore an output
    (``truth()``), not an input."""

    name: ClassVar[str] = "ml_noise_sensitivity_plant"
    n: int
    sensitivity: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _positive_int(self.n, "n", 20))
        object.__setattr__(self, "sensitivity", _nonneg(self.sensitivity, "sensitivity"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return special.expit(np.asarray(X, dtype=float)[:, 0] - self.sensitivity / 2.0)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        y = (rng.random(self.n) < PREVALENCE).astype(np.int64)
        x0 = self.sensitivity * y + rng.standard_normal(self.n)
        return {"X": np.column_stack([x0, rng.standard_normal(self.n)]), "y": y}

    def auc_at(self, eps: float) -> float:
        var_col = 1.0 + self.sensitivity**2 * PREVALENCE * (1.0 - PREVALENCE)
        return float(special.ndtr(self.sensitivity / math.sqrt(2.0 * (1.0 + eps * eps * var_col))))

    def truth(self, grid: Sequence[float] = DEFAULT_GRID) -> dict[str, Any]:
        auc = [self.auc_at(float(e)) for e in grid]
        return {
            "grid": [float(e) for e in grid],
            "auc": auc,
            "auc_delta": [a - auc[0] for a in auc],
            "sensitivity": self.sensitivity,
            "prevalence": PREVALENCE,
        }

    def verify(self, n_draws: int = 200, seed: int = 20261005) -> list[Check]:
        """The AUC at each scale of the default grid against the exact curve (noise as in ``robustness_curve``)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1403]))
        curves = []
        for i in range(n_draws):
            d = self.draw(rng)
            curves.append(robustness.robustness_curve(self.predict, d["X"], d["y"], DEFAULT_GRID, seed=i).auc)
        arr = np.array(curves)
        truth = self.truth()["auc"]
        return [mean_check(f"AUC at eps {e}", truth[j], arr[:, j]) for j, e in enumerate(DEFAULT_GRID)]


@dataclass(frozen=True)
class SeedUnstablePlant(_Plant):
    """The same defaulters and survivors as the noise plant (prevalence 0.3, separation 1.5 in ``x_0``) and a pure-noise
    column ``x_1``. The model of training seed ``s`` is ``expit(x_0 + b_s x_1)`` with ``b_s = instability * xi_s`` and
    ``xi_s`` a standard normal drawn from the stream ``(plant seed, s)``: ``instability`` is the standard deviation of
    the injected seed effect on the score. The population AUC of seed s is ``Phi(1.5 / sqrt(2 (1 + b_s^2)))``, so
    ``truth()`` gives the mean and the standard deviation of the AUC across seeds by Gauss-Hermite quadrature over
    ``xi`` (instability 0: no dispersion)."""

    name: ClassVar[str] = "ml_seed_unstable_plant"
    n: int
    instability: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _positive_int(self.n, "n", 20))
        object.__setattr__(self, "instability", _nonneg(self.instability, "instability"))

    def _xi(self, seed: int) -> float:
        return float(np.random.default_rng([self.seed, int(seed)]).standard_normal())

    def train(self, seed: int):
        """The predict callable fitted with training seed ``seed`` (a pure function of the plant and the seed)."""
        b = self.instability * self._xi(seed)

        def predict(X: np.ndarray) -> np.ndarray:
            x = np.asarray(X, dtype=float)
            return special.expit(x[:, 0] + b * x[:, 1])

        return predict

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        y = (rng.random(self.n) < PREVALENCE).astype(np.int64)
        return {
            "X": np.column_stack([SEPARATION * y + rng.standard_normal(self.n), rng.standard_normal(self.n)]),
            "y": y,
        }

    def _population_auc(self, xi: np.ndarray) -> np.ndarray:
        return special.ndtr(SEPARATION / np.sqrt(2.0 * (1.0 + (self.instability * xi) ** 2)))

    def truth(self) -> dict[str, Any]:
        nodes, weights = np.polynomial.hermite_e.hermegauss(128)
        w = weights / math.sqrt(2.0 * math.pi)
        f = self._population_auc(nodes)
        mean = float(w @ f)
        sd = 0.0 if self.instability == 0.0 else math.sqrt(max(float(w @ (f * f)) - mean * mean, 0.0))
        return {"mean_auc": mean, "sd_auc_across_seeds": sd, "instability": self.instability, "separation": SEPARATION}

    def verify(self, n_draws: int = 8, seed: int = 20261005, n_rows: int = 50_000) -> list[Check]:
        """On one large dataset, the AUC of ``n_draws`` training seeds against the exact population AUC of each seed."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1404]))
        big = SeedUnstablePlant(n_rows, self.instability, self.seed).draw(rng)
        out = []
        for s in range(n_draws):
            auc, var, _, _ = auc_with_variance(self.train(s)(big["X"]), big["y"])
            exact = float(self._population_auc(np.array([self._xi(s)]))[0])
            out.append(Check(f"AUC of seed {s}", exact, auc, math.sqrt(var)))
        return out


@dataclass(frozen=True)
class ExchangeabilityBreakPlant(_Plant):
    """A calibration window of ``10 n`` obligors and a test window of ``n``. The PD of each obligor is Beta(2, 5); the
    labels of the calibration window are Bernoulli(PD) and those of the test window Bernoulli(``expit(logit PD +
    drift)``), so ``drift`` is the change of the label rate after the calibration window in log-odds (drift 0 is
    exchangeability: the null). ``truth()`` gives, for alpha 0.10, the population quantile of the scores, the coverage
    of the split conformal set at it in the calibration window (1 - alpha) and in the test window, and the coverage loss
    between them (exact quadrature)."""

    name: ClassVar[str] = "ml_exchangeability_break_plant"
    n: int
    drift: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", _positive_int(self.n, "n", 20))
        if not math.isfinite(float(self.drift)):
            raise ValueError("drift must be finite")
        object.__setattr__(self, "drift", float(self.drift))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        n_cal = CAL_FACTOR * self.n
        p_cal, p_test = rng.beta(BETA_A, BETA_B, n_cal), rng.beta(BETA_A, BETA_B, self.n)
        y_cal = (rng.random(n_cal) < p_cal).astype(np.int64)
        p_label = special.expit(special.logit(p_test) + self.drift)
        y_test = (rng.random(self.n) < p_label).astype(np.int64)
        return {"p_cal": p_cal, "y_cal": y_cal, "p_test": p_test, "y_test": y_test}

    @staticmethod
    def _calibration_coverage(q: float) -> float:
        """P(score <= q) in the calibration window: labels 1 have score 1 - p, labels 0 have score p."""
        a, b = BETA_A, BETA_B
        mean = a / (a + b)
        e_p_above = mean * (1.0 - special.betainc(a + 1.0, b, 1.0 - q))  # E[p; p >= 1 - q]
        cdf, e_p_below = special.betainc(a, b, q), mean * special.betainc(a + 1.0, b, q)  # P(p <= q), E[p; p <= q]
        return float(e_p_above + (cdf - e_p_below))

    def _test_coverage(self, q: float) -> float:
        def pdf(p: float) -> float:
            return float(stats.beta.pdf(p, BETA_A, BETA_B))

        def relabelled(p: float) -> float:
            return float(special.expit(special.logit(p) + self.drift))

        hi_start, lo_end = 1.0 - q, q
        # label 1 is covered when 1 - p <= q, label 0 when p <= q; the test label is 1 with probability relabelled(p)
        one = integrate.quad(lambda p: pdf(p) * relabelled(p), max(hi_start, 0.0), 1.0, limit=200)[0] if q > 0 else 0.0
        zero = (
            integrate.quad(lambda p: pdf(p) * (1.0 - relabelled(p)), 0.0, min(lo_end, 1.0), limit=200)[0]
            if q > 0
            else 0.0
        )
        return float(one + zero)

    def truth(self, alpha: float = ALPHA) -> dict[str, Any]:
        q = float(
            optimize.brentq(lambda t: self._calibration_coverage(t) - (1.0 - alpha), 1e-9, 1.0 - 1e-9, xtol=1e-14)
        )
        cov_test = self._test_coverage(q)
        return {
            "alpha": alpha,
            "qhat": q,
            "coverage_cal": 1.0 - alpha,
            "coverage_test": cov_test,
            "coverage_loss": (1.0 - alpha) - cov_test,
            "drift": self.drift,
        }

    def verify(self, n_draws: int = 500, seed: int = 20261005) -> list[Check]:
        """At the population quantile, the empirical coverage of the calibration window against 1 - alpha and of the
        test window against the exact test coverage."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1405]))
        t = self.truth()
        cal, test = [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            for p, y, out in ((d["p_cal"], d["y_cal"], cal), (d["p_test"], d["y_test"], test)):
                sets = conformal.conformal_sets(p, t["qhat"])
                out.append(float(sets[np.arange(len(y)), y].mean()))
        return [
            mean_check("calibration-window coverage at the population quantile", t["coverage_cal"], np.array(cal)),
            mean_check("test-window coverage at the population quantile", t["coverage_test"], np.array(test)),
        ]


__all__ = [
    "CovariateShiftPlant",
    "ExchangeabilityBreakPlant",
    "LeakPlant",
    "NoiseSensitivityPlant",
    "SeedUnstablePlant",
]
