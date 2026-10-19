"""RV-1469 to RV-1471: the planted-bias generators, each with an exact truth() and a verify() that a large draw
reproduces within Monte Carlo error."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.fairness.conditional import fit_decision_model, protected_effect
from riskvalidation.fairness.metrics import group_selection_rates
from riskvalidation.fairness.plants import LabelNoiseBiasPlant, ProxyFeaturePlant, ThresholdBiasPlant
from riskvalidation.generators import GeneratorBase


def test_threshold_bias_plant_truth_and_verify():
    plant = ThresholdBiasPlant(n=4000, gap=0.5, seed=11)
    assert isinstance(plant, GeneratorBase)
    tr = plant.truth()
    assert 0.0 < tr["selection_rate_protected"] < tr["selection_rate_reference"] < 1.0
    assert tr["selection_rate_reference"] == pytest.approx(0.7, abs=1e-12)  # the stated base rate
    assert tr["selection_rate_gap"] == pytest.approx(tr["selection_rate_reference"] - tr["selection_rate_protected"],
                                                     abs=1e-15)
    assert tr["selection_rate_gap"] > 0.05
    checks = plant.verify(n_draws=300)
    assert checks and all(c.ok() for c in checks), [c.to_dict() for c in checks]
    # one large draw: the observed rates by group reproduce the truth within 4 standard errors
    big = ThresholdBiasPlant(n=400_000, gap=0.5, seed=1).draw(np.random.default_rng(5))
    rates = group_selection_rates(big["decision"], big["group"])
    for key, field in (("0", "selection_rate_reference"), ("1", "selection_rate_protected")):
        p = tr[field]
        se = math.sqrt(p * (1 - p) / rates[key].n)
        assert abs(rates[key].rate - p) < 4 * se
    assert set(big) >= {"decision", "group", "y", "pd", "x"}
    assert set(np.unique(big["decision"])) == {0, 1} and set(np.unique(big["group"])) == {0, 1}
    assert set(np.unique(big["y"])) == {0, 1}


def test_threshold_bias_plant_null_and_determinism():
    null = ThresholdBiasPlant(n=500, gap=0.0, seed=3)
    tr = null.truth()
    assert tr["selection_rate_gap"] == 0.0 and tr["null"] is True
    assert ThresholdBiasPlant(n=500, gap=0.4, seed=3).truth()["null"] is False
    a = null.draw(np.random.default_rng(9))
    b = null.draw(np.random.default_rng(9))
    for key in ("decision", "group", "y"):
        assert np.array_equal(a[key], b[key])
    assert len(a["decision"]) == 500
    cfg = null.config()
    assert cfg["generator"] == "threshold_bias_plant" and cfg["parameters"]["gap"] == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        null.gap = 1.0  # type: ignore[misc]
    for bad in (dict(n=1, gap=0.1, seed=1), dict(n=100, gap=-0.1, seed=1), dict(n=100, gap=float("nan"), seed=1)):
        with pytest.raises(ValueError):
            ThresholdBiasPlant(**bad)
    # draw_default uses the plant's own seed
    assert np.array_equal(null.draw_default()["decision"], null.draw_default()["decision"])


def test_label_noise_plant_truth_and_verify():
    plant = LabelNoiseBiasPlant(n=4000, noise=0.15, seed=5)
    assert plant.truth()["flip_rate"] == 0.15
    assert plant.truth()["flipped_group"] == 1
    checks = plant.verify(n_draws=200)
    assert checks and all(c.ok() for c in checks), [c.to_dict() for c in checks]
    d = LabelNoiseBiasPlant(n=200_000, noise=0.15, seed=1).draw(np.random.default_rng(2))
    flipped = d["y"] != d["y_true"]
    assert not flipped[d["group"] == 0].any()  # only the protected group's labels are noisy
    assert abs(flipped[d["group"] == 1].mean() - 0.15) < 4 * math.sqrt(0.15 * 0.85 / (d["group"] == 1).sum())
    assert LabelNoiseBiasPlant(n=100, noise=0.0, seed=1).truth()["flip_rate"] == 0.0
    for bad in (-0.1, 0.6, float("nan")):
        with pytest.raises(ValueError):
            LabelNoiseBiasPlant(n=100, noise=bad, seed=1)


def test_proxy_plant_truth_and_verify():
    plant = ProxyFeaturePlant(n=3000, proxy_strength=0.7, seed=2)
    tr = plant.truth()
    assert tr["proxy"] == "proxy" and tr["proxy_strength"] == 0.7
    assert tr["protected_effect_given_proxy"] == 0.0
    checks = plant.verify(n_draws=200)
    assert checks and all(c.ok() for c in checks), [c.to_dict() for c in checks]
    d = ProxyFeaturePlant(n=200_000, proxy_strength=0.7, seed=1).draw(np.random.default_rng(8))
    corr = float(np.corrcoef(d["proxy"], d["group"])[0, 1])
    assert corr == pytest.approx(0.7, abs=0.01)
    assert d["controls_with_proxy"].shape[1] == d["controls"].shape[1] + 1
    assert np.array_equal(d["controls_with_proxy"][:, tr["proxy_index"]], d["proxy"])
    for bad in (-0.1, 1.0, 1.2):
        with pytest.raises(ValueError):
            ProxyFeaturePlant(n=100, proxy_strength=bad, seed=1)


def test_proxy_plant_controlled_disparity_falls_with_the_proxy():
    d = ProxyFeaturePlant(n=60_000, proxy_strength=0.7, seed=1).draw(np.random.default_rng(21))
    protected = d["group"].astype(float)[:, None]
    without = protected_effect(fit_decision_model(d["decision"], protected, d["controls"]), 1)
    with_proxy = protected_effect(fit_decision_model(d["decision"], protected, d["controls_with_proxy"]), 1)
    assert without.lo > 0.0  # the disparity is visible while the proxy is not controlled for
    assert without.coefficient > 0.5

    def z_of(e):  # the Wald z of an effect, from its interval
        return e.coefficient / ((e.hi - e.lo) / (2 * 1.959963984540054))

    assert abs(z_of(with_proxy)) < 3.0  # and is within three standard errors of zero once it is controlled for
    assert abs(with_proxy.coefficient) < abs(without.coefficient) / 5
    # no proxy transmission: the proxy strength 0 makes the uncontrolled disparity vanish too
    d0 = ProxyFeaturePlant(n=60_000, proxy_strength=0.0, seed=1).draw(np.random.default_rng(22))
    zero = protected_effect(fit_decision_model(d0["decision"], d0["group"].astype(float)[:, None], d0["controls"]), 1)
    assert abs(z_of(zero)) < 3.0
