"""RV-1478: the size of each `fair.` test under its null and its power against a planted bias, measured with
`harness.run.simulate` at 4,000 repetitions (SDD l.335), with the Monte Carlo error (Morris, White and Crowther 2019).

The tests with a p-value are read at p < 0.05. The tests that carry a metric and a policy light (parity, equalized odds,
equal opportunity, calibration, AUC, situation testing) and the rule tests (adverse impact, conditional disparity,
replication) are read by their own light: a rejection is a light that is not green, which is the amber rule of the
contract. Set FAIRNESS_SIZE_POWER_OUT to a path to write the measured rates as JSON (the figures of docs/fairness)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from riskvalidation.fairness.conditional import fit_decision_model, protected_effect
from riskvalidation.fairness.metrics import group_selection_rates
from riskvalidation.fairness.plants import LabelNoiseBiasPlant, ProxyFeaturePlant, ThresholdBiasPlant
from riskvalidation.fairness.replication import replicate_effect
from riskvalidation.fairness.situation import situation_test
from riskvalidation.generators import GeneratorBase
from riskvalidation.harness.run import light_not_green, simulate
from riskvalidation.validation import fairness as fv

N_REP = 4000
OUT = os.environ.get("FAIRNESS_SIZE_POWER_OUT")
_RECORD: dict[str, Any] = {}


def record(name: str, sim) -> float:
    """The not-green rate (or the p < 0.05 rate) of a simulation, stored for the docs, with its Monte Carlo error."""
    rate = sim.rates["p<0.05"] if name.endswith("[p]") else sim.rates["light not green"]
    r = rate.rate
    _RECORD[name] = {"rate": r, "se": rate.se, "n_rep": sim.n_rep, "undefined": rate.undefined}
    if OUT:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(_RECORD, fh, indent=1, sort_keys=True)
    return r


def se_of(rate: float) -> float:
    return math.sqrt(max(rate * (1 - rate), 1e-12) / N_REP)


# ----- adapters: a draw of a plant to the arguments of one test


def z_adapter(decision, group):
    rates = group_selection_rates(decision, group)
    return fv.two_proportion_z("sp", rates["0"].favourable, rates["0"].n, rates["1"].favourable, rates["1"].n)


def fisher_adapter(decision, group):
    rates = group_selection_rates(decision, group)
    return fv.fisher_exact("sp", rates["0"].favourable, rates["0"].n, rates["1"].favourable, rates["1"].n)


def parity_adapter(decision, group):
    return fv.demographic_parity("sp", group_selection_rates(decision, group), "0")


def adverse_adapter(decision, group):
    return fv.adverse_impact("sp", group_selection_rates(decision, group), "0")


def eo_adapter(y, decision, group):
    return fv.equalized_odds("sp", y, decision, group)


def eopp_adapter(y, decision, group):
    return fv.equal_opportunity("sp", y, decision, group)


def cal_adapter(pd, defaults, group):
    return fv.calibration_within_groups("sp", pd, defaults, group)


def auc_adapter(pd, defaults, group):
    return fv.auc_by_group("sp", pd, defaults, group)


def go(test, plant, seed, rules=()):
    return simulate(test, plant, n_rep=N_REP, seed=seed, rules=tuple(rules) or (light_not_green(),), alphas=(0.05,))


# ---------------------------------------------------------------------------------------------------------------


def test_size_and_power_two_proportion_z_and_fisher():
    null, alt = ThresholdBiasPlant(n=2000, gap=0.0, seed=1), ThresholdBiasPlant(n=2000, gap=0.3, seed=1)
    z_size = record("two_proportion_z size [p]", go(z_adapter, null, 101))
    z_power = record("two_proportion_z power [p]", go(z_adapter, alt, 102))
    se = se_of(0.05)
    assert abs(z_size - 0.05) <= 3.3 * se, (z_size, se)  # the pooled z-test holds its 5 percent level
    assert z_power >= 0.85
    f_size = record("fisher_exact size [p]", go(fisher_adapter, null, 103))
    f_power = record("fisher_exact power [p]", go(fisher_adapter, alt, 104))
    assert f_size <= 0.05 + 3.1 * se  # exact: conservative, never above the level beyond Monte Carlo error
    assert f_size < z_size + 3 * se and f_power >= 0.80
    # the planted gap is a real selection-rate gap of the stated size
    assert alt.truth()["selection_rate_gap"] > 0.08


def test_size_and_power_parity_and_adverse_impact():
    null = ThresholdBiasPlant(n=2000, gap=0.0, seed=1)
    p_size = record("demographic_parity size", go(parity_adapter, null, 111))
    p_power = record("demographic_parity power", go(parity_adapter, ThresholdBiasPlant(n=2000, gap=0.5, seed=1), 112))
    assert p_size < 0.05 and p_power > 0.95
    a_size = record("adverse_impact size", go(adverse_adapter, null, 113))
    a_power = record("adverse_impact power", go(adverse_adapter, ThresholdBiasPlant(n=2000, gap=0.8, seed=1), 114))
    assert a_size < 0.01 and a_power > 0.95


def test_size_and_power_outcome_conditioned_tests():
    null = LabelNoiseBiasPlant(n=8000, noise=0.0, seed=1)
    alt = LabelNoiseBiasPlant(n=8000, noise=0.25, seed=1)
    for name, adapter, seed in (("equalized_odds", eo_adapter, 121), ("calibration_within_groups", cal_adapter, 125),
                                ("auc_by_group", auc_adapter, 127)):
        size = record(f"{name} size", go(adapter, null, seed))
        power = record(f"{name} power, label noise 0.25", go(adapter, alt, seed + 1))
        assert size < 0.05, (name, size)
        assert power > 0.9, (name, power)
    # equal opportunity reads the true positive rate only, which label noise of 0.25 moves by about 0.06 (the flipped
    # defaults enter the positive class): the power is weak there, and strong against the threshold bias that moves it
    size = record("equal_opportunity size", go(eopp_adapter, null, 123))
    weak = record("equal_opportunity power, label noise 0.25", go(eopp_adapter, alt, 124))
    strong = record("equal_opportunity power, threshold bias 0.5",
                    go(eopp_adapter, ThresholdBiasPlant(n=8000, gap=0.5, seed=1), 129))
    null_threshold = record("equal_opportunity size, threshold plant at gap 0",
                            go(eopp_adapter, ThresholdBiasPlant(n=8000, gap=0.0, seed=1), 130))
    assert size < 0.05 and null_threshold < 0.05
    assert weak > 0.3 and strong > 0.9, (weak, strong)
    for name, adapter, seed in (("equalized_odds", eo_adapter, 131),):
        assert record(f"{name} power, threshold bias 0.5",
                      go(adapter, ThresholdBiasPlant(n=8000, gap=0.5, seed=1), seed)) > 0.9


@dataclass(frozen=True)
class PairedYears(GeneratorBase):
    """Two independent years of a proxy plant; `reverse` recodes the protected indicator in the second year (a planted
    instability of the estimate between year t and year t+1). Test-only."""

    name: ClassVar[str] = "paired_years"
    n: int = 3000
    proxy_strength: float = 0.7
    reverse: bool = False

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        plant = ProxyFeaturePlant(n=self.n, proxy_strength=self.proxy_strength, seed=0)
        a, b = plant.draw(rng), plant.draw(rng)
        if self.reverse:
            b = {**b, "group": 1 - b["group"]}
        return {"fit": a, "confirm": b}

    def truth(self) -> dict[str, Any]:
        return {"reverse": self.reverse, "proxy_strength": self.proxy_strength}


def _effect_of(draw, controls_key):
    protected = draw["group"].astype(float)[:, None]
    return protected_effect(fit_decision_model(draw["decision"], protected, draw[controls_key]), 1)


def cond_adapter(decision, group, controls):
    model = fit_decision_model(decision, group.astype(float)[:, None], controls)
    return fv.conditional_disparity("sp", protected_effect(model, 1), ["x1", "x2", "x3"], ["credit_score"])


def cond_proxy_adapter(decision, group, controls_with_proxy):
    model = fit_decision_model(decision, group.astype(float)[:, None], controls_with_proxy)
    return fv.conditional_disparity("sp", protected_effect(model, 1), ["x1", "x2", "x3", "proxy"], ["credit_score"])


def rate_spread_adapter(decision, group, controls):
    # the same construction on a continuous outcome: the linear model of a rate spread driven by the proxy
    from riskvalidation.fairness.conditional import fit_rate_spread_model

    spread = controls[:, 0] * 0.3 + (decision == 0) * 0.5
    model = fit_rate_spread_model(spread, group.astype(float)[:, None], controls)
    return fv.rate_spread_disparity("sp", protected_effect(model, 1), ["x1", "x2", "x3"], ["credit_score"])


def replication_adapter(fit, confirm):
    rep = replicate_effect(_effect_of(fit, "controls"), _effect_of(confirm, "controls"))
    return fv.replication("sp", rep)


def test_size_and_power_conditional_disparity_and_replication():
    null = ProxyFeaturePlant(n=3000, proxy_strength=0.0, seed=1)
    alt = ProxyFeaturePlant(n=3000, proxy_strength=0.7, seed=1)
    se = se_of(0.05)
    size = record("conditional_disparity size", go(cond_adapter, null, 131))
    assert abs(size - 0.05) <= 3.3 * se + 0.005, size  # a 95 percent interval excludes a true zero 5% of the time
    power = record("conditional_disparity power", go(cond_adapter, alt, 132))
    assert power > 0.99
    controlled = record("conditional_disparity size with the proxy controlled", go(cond_proxy_adapter, alt, 133))
    assert abs(controlled - 0.05) <= 3.3 * se + 0.005, controlled  # the disparity falls once the proxy is a control
    rs_size = record("rate_spread_disparity size", go(rate_spread_adapter, null, 134))
    rs_power = record("rate_spread_disparity power", go(rate_spread_adapter, alt, 135))
    assert rs_size <= 0.05 + 3.3 * se + 0.005 and rs_power > 0.99
    stable = record("replication false alarm, stable effect", go(replication_adapter, PairedYears(), 136))
    assert stable < 0.01
    flipped = record("replication power, reversed effect", go(replication_adapter, PairedYears(reverse=True), 137))
    assert flipped > 0.99
    # with no true effect the sign is a coin flip: the rule has no size to speak of, and the record says so
    coin = record("replication amber rate, no effect", go(replication_adapter, PairedYears(proxy_strength=0.0), 138))
    assert abs(coin - 0.5) <= 4 * se_of(0.5)


@dataclass(frozen=True)
class SituationPlant(GeneratorBase):
    """Two continuous attributes; the negative decision (0) is likelier in the protected group by `bias`. Test-only."""

    name: ClassVar[str] = "situation_plant"
    n: int = 400
    bias: float = 0.0

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        group = (rng.random(self.n) < 0.4).astype(int)
        x = rng.uniform(0.0, 10.0, (self.n, 2))
        p_negative = 0.25 + np.where(group == 1, self.bias, 0.0)
        decision = (rng.random(self.n) >= p_negative).astype(int)
        return {"X": x, "group": group, "decision": decision}

    def truth(self) -> dict[str, Any]:
        return {"bias": self.bias}


def situation_adapter(X, group, decision):
    return fv.situation_testing("sp", situation_test(X, group, decision, k=30))


def test_size_and_power_situation_testing():
    # the contract's amber cut is a share of 0.10 of the complainants significant; the rule is also read at 0.15 and
    # 0.20 to show how its size depends on the cut (the complainants share neighbours, so the share is more
    # variable than a binomial count)
    from riskvalidation.harness.run import metric_above

    rules = (light_not_green(), metric_above(0.15, "share>0.15"), metric_above(0.20, "share>0.20"))
    null = go(situation_adapter, SituationPlant(bias=0.0), 141, rules)
    alt = go(situation_adapter, SituationPlant(bias=0.4), 142, rules)
    size = record("situation_testing size", null)
    power = record("situation_testing power", alt)
    for label in ("share>0.15", "share>0.20"):
        _RECORD[f"situation_testing size at {label}"] = {"rate": null.rates[label].rate, "se": null.rates[label].se,
                                                       "n_rep": N_REP, "undefined": null.rates[label].undefined}
        _RECORD[f"situation_testing power at {label}"] = {"rate": alt.rates[label].rate, "se": alt.rates[label].se,
                                                        "n_rep": N_REP, "undefined": alt.rates[label].undefined}
    if OUT:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(_RECORD, fh, indent=1, sort_keys=True)
    # measured, not tuned: at n = 400 and k = 30 the amber cut of 0.10 raises a false alarm on about one null draw in
    # five (docs/fairness/04); the cut is a proposed convention for the main session's approval
    assert size < 0.25, size
    assert power > 0.95, power
    assert null.rates["share>0.20"].rate < size
