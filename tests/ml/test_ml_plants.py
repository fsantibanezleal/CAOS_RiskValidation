"""RV-1428 and RV-1429: the planted-defect generators of `riskvalidation.ml.plants` (contract engine-ml-api 1.0.0):
a known truth, a verifier, and the checks the contract names for each."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special, stats

from riskvalidation.generators import GeneratorBase
from riskvalidation.ml.conformal import conformal_coverage, conformal_sets, split_conformal_quantile
from riskvalidation.ml.leakage import single_feature_auc
from riskvalidation.ml.plants import (
    CovariateShiftPlant,
    ExchangeabilityBreakPlant,
    LeakPlant,
    NoiseSensitivityPlant,
    SeedUnstablePlant,
)
from riskvalidation.ml.robustness import robustness_curve, shift_score_rank
from riskvalidation.ml.seeds import seed_auc_dispersion
from riskvalidation.validation.discrimination import auc_with_variance

ALL = (LeakPlant(500, 1.0, 1), CovariateShiftPlant(500, 1.0, [0], 1), NoiseSensitivityPlant(500, 1.0, 1),
       SeedUnstablePlant(500, 0.5, 1), ExchangeabilityBreakPlant(500, 0.5, 1))


@pytest.mark.parametrize("plant", ALL, ids=lambda p: type(p).__name__)
def test_every_plant_is_a_generator_with_config_truth_draw_and_verify(plant):
    assert isinstance(plant, GeneratorBase)
    json.dumps(plant.config())  # the configuration is JSON-safe
    json.dumps(plant.truth(), default=lambda o: np.asarray(o).tolist())
    a = plant.draw(np.random.default_rng(3))
    b = plant.draw(np.random.default_rng(3))
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
    checks = plant.verify(n_draws=300)
    assert checks and all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    with pytest.raises(AttributeError):
        plant.n = 3  # frozen


def test_leak_plant_single_feature_auc_of_the_leak_rises_with_strength_and_matches_truth():
    aucs = []
    for s in (0.0, 1.0, 2.0, 4.0):
        plant = LeakPlant(4000, s, 5)
        t = plant.truth()
        assert t["leak_column"] == 3 and t["leak_strength"] == s
        assert t["leak_auc"] == pytest.approx(float(special.ndtr(s / math.sqrt(2.0))), abs=1e-15)
        d = plant.draw(np.random.default_rng(6))
        x, y = d["X"], d["y"]
        assert x.shape == (4000, 4) and d["leak_column"] == 3
        res = single_feature_auc(x, y)
        a = auc_with_variance(x[:, 3], y)
        aucs.append(a[0])
        assert abs(a[0] - t["leak_auc"]) <= 4.0 * math.sqrt(a[1]) + 1e-12
        if s >= 4.0:
            assert res.feature == 3 and res.suspect is True  # the screen names the leaking column
        if s == 0.0:
            assert res.suspect is False
    assert aucs == sorted(aucs) and aucs[-1] > aucs[0] + 0.4
    assert LeakPlant(100, 1.0, 1).verify(n_draws=50)  # verify() recomputes the truth and passes
    with pytest.raises(ValueError, match="leak_strength"):
        LeakPlant(100, -1.0, 1)


def test_covariate_shift_plant_truth_and_the_shift_score_finds_the_shift():
    plant = CovariateShiftPlant(3000, 1.5, [0, 2], 7)
    t = plant.truth()
    assert t["shifted_features"] == [0, 2] and t["shift"] == 1.5 and t["null"] is False
    assert CovariateShiftPlant(3000, 0.0, [0], 7).truth()["null"] is True
    d = plant.draw(np.random.default_rng(8))
    assert d["X_dev"].shape == d["X_eval"].shape == (3000, 3)
    means = d["X_eval"].mean(axis=0) - d["X_dev"].mean(axis=0)
    assert means[0] > 1.2 and means[2] > 1.2 and abs(means[1]) < 0.15
    score = shift_score_rank(d["X_dev"], d["X_eval"], seed=9)
    null = CovariateShiftPlant(3000, 0.0, [0, 2], 7).draw(np.random.default_rng(8))
    null_score = shift_score_rank(null["X_dev"], null["X_eval"], seed=9)
    assert float(score.mean()) > 0.7 and abs(float(null_score.mean()) - 0.5) < 0.05
    pred = plant.predict(d["X_eval"])
    assert pred.shape == (3000,) and np.all((pred > 0.0) & (pred < 1.0))
    with pytest.raises(ValueError, match="features"):
        CovariateShiftPlant(100, 1.0, [5], 1)


def test_noise_sensitivity_plant_recovers_the_true_degradation_curve():
    plant = NoiseSensitivityPlant(4000, 2.5, 3)
    t = plant.truth()
    grid = t["grid"]
    assert grid[0] == 0.0 and len(grid) == len(t["auc"]) == len(t["auc_delta"])
    assert t["auc_delta"][0] == 0.0 and all(a > b for a, b in zip(t["auc"], t["auc"][1:], strict=False))
    reps = 120
    curves = []
    for r in range(reps):
        d = plant.draw(np.random.default_rng([4, r]))
        curves.append(robustness_curve(plant.predict, d["X"], d["y"], grid, seed=r).auc)
    curves = np.array(curves)
    se = curves.std(axis=0, ddof=1) / math.sqrt(reps)
    assert np.all(np.abs(curves.mean(axis=0) - np.array(t["auc"])) <= 4.0 * se + 1e-4), (curves.mean(axis=0), t["auc"])
    null = NoiseSensitivityPlant(1000, 0.0, 3).truth()
    assert all(a == 0.5 for a in null["auc"])  # no sensitivity: no signal to lose
    with pytest.raises(ValueError, match="sensitivity"):
        NoiseSensitivityPlant(100, -1.0, 1)


def test_seed_unstable_plant_dispersion_matches_truth():
    plant = SeedUnstablePlant(20_000, 0.8, 5)
    t = plant.truth()
    d = plant.draw(np.random.default_rng(1))
    seeds = list(range(60))
    disp = seed_auc_dispersion(plant.train, d["X"], d["y"], seeds)
    assert disp.n_seeds == 60
    # the population AUC of seed k is Phi(delta / sqrt(2 (1 + (instability xi_k)^2))); the sample AUC adds a smooth
    # sampling deviation, so the standard deviation is compared with the exact one within 25 percent and the mean with 0.01
    assert disp.sd == pytest.approx(t["sd_auc_across_seeds"], rel=0.25)
    assert disp.mean == pytest.approx(t["mean_auc"], abs=0.01)
    x = d["X"][:50]
    np.testing.assert_array_equal(plant.train(7)(x), plant.train(7)(x))  # a seed fixes the fit
    assert not np.array_equal(plant.train(7)(x), plant.train(8)(x))
    zero = SeedUnstablePlant(5000, 0.0, 5)
    assert zero.truth()["sd_auc_across_seeds"] == 0.0
    dz = zero.draw(np.random.default_rng(2))
    assert seed_auc_dispersion(zero.train, dz["X"], dz["y"], [1, 2, 3]).sd == 0.0  # identity: no instability
    with pytest.raises(ValueError, match="instability"):
        SeedUnstablePlant(100, -0.1, 1)


def test_exchangeability_break_plant_truth_and_coverage():
    alpha = 0.10
    null = ExchangeabilityBreakPlant(2000, 0.0, 4)
    assert null.truth()["coverage_loss"] == pytest.approx(0.0, abs=1e-12)
    plant = ExchangeabilityBreakPlant(2000, 1.0, 4)
    t = plant.truth()
    assert t["alpha"] == alpha and t["coverage_loss"] > 0.02 and t["coverage_test"] < t["coverage_cal"]
    # the exact coverage at the population quantile against a large draw
    d = plant.draw(np.random.default_rng(5))
    assert len(d["p_cal"]) > len(d["p_test"]) == 2000
    scores = np.where(d["y_cal"] == 1, 1.0 - d["p_cal"], d["p_cal"])
    q = split_conformal_quantile(scores, alpha)
    cov = conformal_coverage(conformal_sets(d["p_test"], q), d["y_test"], alpha)
    se = math.sqrt(cov.coverage * (1.0 - cov.coverage) / cov.n)
    assert abs(cov.coverage - t["coverage_test"]) <= 5.0 * se + 0.01
    p = stats.binomtest(round(cov.coverage * cov.n), cov.n, 1.0 - alpha).pvalue
    assert cov.p_value == pytest.approx(p, abs=1e-12)
    with pytest.raises(ValueError, match="n"):
        ExchangeabilityBreakPlant(5, 0.0, 1)
