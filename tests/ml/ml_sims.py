"""The size and power runs of the `ml.` validation tests (RV-1437): each test is fed, through the engine pipeline it
belongs to, by a generator with a null (no defect) and a planted defect, and `harness.run.simulate` counts how often the
light is not green over `N_REP` seeded repetitions.

The five planted-defect generators of `riskvalidation.ml.plants` serve the tests they name; the other tests are fed by
the small array generators of this module (an attribution table, an effect range, an exact linear Shapley table,
binormal train and test scores, a calibrated and a miscalibrated slice, a NaN rate, two forecasts). They are test
fixtures of the size and power study, never part of the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import special

from riskvalidation.generators import GeneratorBase
from riskvalidation.harness.run import Simulation, light_not_green, simulate
from riskvalidation.ml import (  # noqa: F401  (the sub-modules the pipelines use)
    cbpe,
    challenger,
    conformal,
    dataquality,
    effects,
    explanations,
    generalisation,
    leakage,
    robustness,
    seeds,
)
from riskvalidation.ml.plants import (
    CovariateShiftPlant,
    ExchangeabilityBreakPlant,
    LeakPlant,
    NoiseSensitivityPlant,
    SeedUnstablePlant,
)
from riskvalidation.validation import ml as v
from riskvalidation.validation.discrimination import auc_with_variance

N_REP = 4000  # no fewer than 4,000 repetitions (contract engine-ml-api 1.0.0, SDD l.335)
SEED = 20261019
SHIFT_PERMUTATIONS = 99  # R of the size study (the engine default is 199)
SHIFT_N = 2000  # development rows of the shift plant; the evaluation set is its first SHIFT_N // 2 rows (docs/ml/08)


def study(test, generator, *, seed: int = SEED, n_rep: int = N_REP, test_kwargs=None) -> Simulation:
    """The rate at which the light is not green (an alarm), with its Monte Carlo error."""
    return simulate(
        test, generator, n_rep=n_rep, seed=seed, alphas=(), rules=[light_not_green("alarm")], test_kwargs=test_kwargs
    )


# -------------------------------------------------------------------------------------------- the pipelines


def run_robustness(plant: NoiseSensitivityPlant, grid, tolerance: float, preset_eps: float):
    def run(X, y, seed):
        curve = robustness.robustness_curve(plant.predict, X, y, grid, seed)
        return v.robustness("m", curve, tolerance, preset_eps)

    return run


def run_shift(plant: CovariateShiftPlant, share: float, shift_permutations: int = SHIFT_PERMUTATIONS):
    def run(X_dev, y_dev, X_eval, y_eval, seed):
        X_eval, y_eval = X_eval[: len(X_dev) // 2], y_eval[: len(X_dev) // 2]  # dev >= 2 x eval
        score = robustness.shift_score_rank(X_dev, X_eval, seed)
        res = robustness.worst_subsample_performance(plant.predict, X_eval, y_eval, score, share)
        ref = robustness.shift_development_reference(
            plant.predict, X_dev, y_dev, X_eval, y_eval, score, share, shift_permutations, seed
        )
        return v.shift_resilience("m", res, ref)

    return run


def run_seeds(plant: SeedUnstablePlant, n_seeds: int):
    def run(X, y):
        return v.seed_stability("m", seeds.seed_auc_dispersion(plant.train, X, y, list(range(n_seeds))))

    return run


def run_leak(X, y):
    half = len(y) // 2
    res = leakage.resemblance_auc(X[:half], X[half:], seed=1)
    return v.leakage_screen("m", 0, leakage.single_feature_auc(X, y), 0.0, res)


def run_conformal(n_alpha: float = 0.10, n_cal: int | None = None):
    """The pipeline of the conformal coverage test; `n_cal` keeps only the first `n_cal` calibration points (the plant
    draws ten times the test window), so the three split ratios of the size study share one generator."""

    def run(p_cal, y_cal, p_test, y_test):
        if n_cal is not None:
            p_cal, y_cal = p_cal[:n_cal], y_cal[:n_cal]
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        q = conformal.split_conformal_quantile(scores, n_alpha)
        cov = conformal.conformal_coverage(conformal.conformal_sets(p_test, q), y_test, n_alpha, n_cal=len(p_cal))
        return v.conformal_coverage("m", cov)

    return run


# -------------------------------------------------------------------------------------------- the fixtures


@dataclass(frozen=True)
class RefitAttributions(GeneratorBase):
    """Per-obligor attributions of `refits` refits: a common table plus refit noise of scale `noise`."""

    name: ClassVar[str] = "refit_attributions"
    n: int = 200
    p: int = 8
    refits: int = 4
    noise: float = 0.1

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        scale = 1.0 - 0.05 * np.arange(self.p)
        base = rng.standard_normal((self.n, self.p)) * scale
        return {"attributions_by_refit": base + self.noise * rng.standard_normal((self.refits, self.n, self.p))}

    def truth(self) -> dict[str, Any]:
        return {"noise": self.noise}


def run_reason_codes(attributions_by_refit):
    return v.reason_code_stability("m", explanations.reason_code_overlap(attributions_by_refit))


@dataclass(frozen=True)
class ImportanceVsEffect(GeneratorBase):
    """Mean absolute SHAP per feature and the range of the ALE curve: related (`related`) or independent."""

    name: ClassVar[str] = "importance_vs_effect"
    p: int = 10
    related: bool = True

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        imp = rng.lognormal(0.0, 0.7, self.p)
        eff = imp * rng.lognormal(0.0, 0.2, self.p) if self.related else rng.lognormal(0.0, 0.7, self.p)
        return {"attribution_importance": imp, "effect_range": eff}

    def truth(self) -> dict[str, Any]:
        return {"related": self.related}


def run_effect(attribution_importance, effect_range):
    rho = explanations.rank_agreement(attribution_importance, effect_range)
    return v.effect_agreement("m", rho, len(attribution_importance))


@dataclass(frozen=True)
class LinearShapley(GeneratorBase):
    """The exact Shapley values of a linear model, with an explainer that is `tamper` off in its first column."""

    name: ClassVar[str] = "linear_shapley"
    n: int = 200
    tamper: float = 0.0

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        x = rng.standard_normal((self.n, 5))
        w, b = rng.standard_normal(5), float(rng.standard_normal())
        phi = (x - x.mean(axis=0)) * w
        phi[:, 0] += self.tamper
        return {"prediction": b + x @ w, "base_value": b + float(x.mean(axis=0) @ w), "attributions": phi}

    def truth(self) -> dict[str, Any]:
        return {"gap": abs(self.tamper)}


def run_additivity(prediction, base_value, attributions):
    return v.additivity("m", explanations.shapley_additivity_gap(prediction, base_value, attributions))


@dataclass(frozen=True)
class TrainTestScores(GeneratorBase):
    """Binormal scores on a training and a test sample: the model separates by `sep_train` there and `sep_test` here."""

    name: ClassVar[str] = "train_test_scores"
    n: int = 1000
    sep_train: float = 1.0
    sep_test: float = 1.0

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tag, sep in (("train", self.sep_train), ("test", self.sep_test)):
            y = (rng.random(self.n) < 0.3).astype(int)
            out[f"risk_{tag}"] = sep * y + rng.standard_normal(self.n)
            out[f"y_{tag}"] = y
        return out

    def truth(self) -> dict[str, Any]:
        auc = lambda d: float(special.ndtr(d / math.sqrt(2.0)))  # noqa: E731
        return {
            "auc_train": auc(self.sep_train),
            "auc_test": auc(self.sep_test),
            "gap": auc(self.sep_train) - auc(self.sep_test),
        }


def run_overfit(risk_train, y_train, risk_test, y_test):
    gap = generalisation.overfit_gap(
        auc_with_variance(risk_train, y_train)[0], auc_with_variance(risk_test, y_test)[0], 0.05
    )
    return v.overfit("m", gap)


@dataclass(frozen=True)
class SliceErrors(GeneratorBase):
    """Brier terms of `slices` slices of PDs ~ Beta(1, 4); in slice 0 the model gives 1 - p when `defect` is set."""

    name: ClassVar[str] = "slice_errors"
    n: int = 3000
    slices: int = 6
    defect: bool = False

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        p = rng.beta(1.0, 4.0, self.n)
        y = (rng.random(self.n) < p).astype(int)
        bins = np.arange(self.n) % self.slices
        q = np.where((bins == 0) & self.defect, 1.0 - p, p)
        return {"error": (q - y) ** 2, "bins": bins}

    def truth(self) -> dict[str, Any]:
        return {"defect": self.defect}


def run_weak_spot(error, bins):
    return v.weak_spot("m", generalisation.weak_spots(error, bins, 2.0), 2.0, 6)


@dataclass(frozen=True)
class MissingCells(GeneratorBase):
    """A feature table with each cell missing independently with probability `rate`."""

    name: ClassVar[str] = "missing_cells"
    n: int = 1000
    p: int = 10
    rate: float = 0.002

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        x = rng.standard_normal((self.n, self.p))
        x[rng.random((self.n, self.p)) < self.rate] = np.nan
        return {"X": x}

    def truth(self) -> dict[str, Any]:
        return {"rate": self.rate}


def run_dq(X):
    return v.data_quality("m", [dataquality.dq_completeness(X)])


@dataclass(frozen=True)
class TwoForecasts(GeneratorBase):
    """A champion and a challenger forecasting the same labels: the challenger adds `extra` noise to the true PD."""

    name: ClassVar[str] = "two_forecasts"
    n: int = 1500
    extra: float = 0.0

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        p = rng.beta(2.0, 5.0, self.n)
        y = (rng.random(self.n) < p).astype(int)
        champ = np.clip(p + 0.08 * rng.standard_normal(self.n), 0.01, 0.99)
        chall = np.clip(p + math.hypot(0.08, self.extra) * rng.standard_normal(self.n), 0.01, 0.99)
        return {"p_champion": champ, "p_challenger": chall, "y": y}

    def truth(self) -> dict[str, Any]:
        return {"extra": self.extra}


def run_challenger(p_champion, p_challenger, y, seed):
    return v.challenger_brier("m", challenger.paired_brier_bootstrap(p_champion, p_challenger, y, 200, seed))


# -------------------------------------------------------------------------------------------- the conditions


def conditions() -> dict[str, tuple[Any, Any, Any, Any, dict[str, Any]]]:
    """name: (test callable, null generator, defect generator, feed or None, description)."""
    grid = (0.0, 0.25, 0.5, 1.0)
    noise_null, noise_def = NoiseSensitivityPlant(2000, 0.25, 1), NoiseSensitivityPlant(2000, 2.5, 1)
    shift_null, shift_def = CovariateShiftPlant(SHIFT_N, 0.0, [0], 1), CovariateShiftPlant(SHIFT_N, 3.0, [0], 1)
    seed_null, seed_def = SeedUnstablePlant(4000, 0.2, 1), SeedUnstablePlant(4000, 1.0, 1)
    return {
        "ml.robustness": (
            run_robustness(noise_null, grid, 0.02, 0.5),
            noise_null,
            noise_def,
            run_robustness(noise_def, grid, 0.02, 0.5),
            {"null": "sensitivity 0.25", "defect": "sensitivity 2.5"},
        ),
        "ml.shift_resilience": (
            run_shift(shift_null, 0.25),
            shift_null,
            shift_def,
            run_shift(shift_def, 0.25),
            {"null": "shift 0", "defect": "a quarter of the rows shifted 3 sd on feature 0"},
        ),
        "ml.seed_stability": (
            run_seeds(seed_null, 10),
            seed_null,
            seed_def,
            run_seeds(seed_def, 10),
            {"null": "instability 0.2", "defect": "instability 1.0"},
        ),
        "ml.reason_code_stability": (
            run_reason_codes,
            RefitAttributions(noise=0.1),
            RefitAttributions(noise=1.5),
            None,
            {"null": "refit noise 0.1", "defect": "refit noise 1.5"},
        ),
        "ml.effect_agreement": (
            run_effect,
            ImportanceVsEffect(related=True),
            ImportanceVsEffect(related=False),
            None,
            {"null": "effect range follows importance", "defect": "independent"},
        ),
        "ml.additivity": (
            run_additivity,
            LinearShapley(tamper=0.0),
            LinearShapley(tamper=1e-6),
            None,
            {"null": "exact Shapley values", "defect": "first column off by 1e-6"},
        ),
        "ml.overfit": (
            run_overfit,
            TrainTestScores(sep_train=1.0, sep_test=1.0),
            TrainTestScores(sep_train=2.0, sep_test=1.0),
            None,
            {"null": "equal separation", "defect": "train separation 2, test 1"},
        ),
        "ml.weak_spot": (
            run_weak_spot,
            SliceErrors(defect=False),
            SliceErrors(defect=True),
            None,
            {"null": "calibrated slices", "defect": "slice 0 predicts 1 - p"},
        ),
        "ml.conformal_coverage": (
            run_conformal(),
            ExchangeabilityBreakPlant(2000, 0.0, 1),
            ExchangeabilityBreakPlant(2000, 0.5, 1),
            None,
            {"null": "drift 0", "defect": "log-odds drift 0.5"},
        ),
        "ml.conformal_coverage (n_cal 2000, m 2000)": (
            run_conformal(n_cal=2000),
            ExchangeabilityBreakPlant(2000, 0.0, 1),
            ExchangeabilityBreakPlant(2000, 0.5, 1),
            None,
            {"null": "drift 0, n_cal = m = 2000", "defect": "log-odds drift 0.5, n_cal = m = 2000"},
        ),
        "ml.conformal_coverage (n_cal 1000, m 4000)": (
            run_conformal(n_cal=1000),
            ExchangeabilityBreakPlant(4000, 0.0, 1),
            ExchangeabilityBreakPlant(4000, 0.5, 1),
            None,
            {"null": "drift 0, n_cal 1000, m 4000", "defect": "log-odds drift 0.5, n_cal 1000, m 4000"},
        ),
        "ml.leakage_screen": (
            run_leak,
            LeakPlant(1000, 0.0, 1),
            LeakPlant(1000, 4.0, 1),
            None,
            {"null": "leak strength 0", "defect": "leak strength 4"},
        ),
        "ml.data_quality": (
            run_dq,
            MissingCells(rate=0.002),
            MissingCells(rate=0.03),
            None,
            {"null": "NaN rate 0.2 percent", "defect": "NaN rate 3 percent"},
        ),
        "ml.challenger_brier": (
            run_challenger,
            TwoForecasts(extra=0.0),
            TwoForecasts(extra=0.15),
            None,
            {"null": "equal noise", "defect": "challenger noise 0.15 above"},
        ),
    }
