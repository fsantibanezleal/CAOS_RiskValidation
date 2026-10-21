"""RV-U16-E2: the generators of the nine size studies keep the generator contract (frozen configuration, JSON-safe,
deterministic draw, exact truth held to a large draw on their own stream) and the registry of the studies is
complete."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from riskvalidation.core.registry import catalogue
from riskvalidation.engines.vasicek import lhp_quantile
from riskvalidation.generators.credit import DownturnLgd, LgdCalibration, StageTransitions
from riskvalidation.generators.exceedances import EsResiduals, LhpLossRates, PfeSeries
from riskvalidation.generators.montecarlo import (
    CreditRiskPlusSample,
    McQuantile,
    lhp_quantile_closed_form,
    sample_quantile_with_se,
)
from riskvalidation.generators.size_studies import N_REP, STAGE_MATRIX, STAGE_SHIFT, STUDIES, Study

GENERATORS = [
    PfeSeries(500, 0.975, forecast_factor=0.9),
    EsResiduals(800, 0.05, es_factor=0.95),
    LhpLossRates(40, true_rho=0.3),
    StageTransitions((4000, 1500, 500), STAGE_MATRIX, STAGE_SHIFT),
    LgdCalibration(60, n_periods=5, bias=0.1, biased_periods=(4,)),
    DownturnLgd(150, shortfall=0.05),
    McQuantile(5000, mc_rho=0.13),
    CreditRiskPlusSample(500, mc_sigma2=0.6),
]


def _equal(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: type(g).__name__)
def test_generator_contract(gen):
    cfg = gen.config()
    assert cfg["generator"] == gen.name and "riskvalidation" in cfg
    json.dumps(cfg)
    json.dumps(gen.truth(), default=str)
    a = gen.draw(np.random.default_rng(np.random.SeedSequence(7)))
    b = gen.draw(np.random.default_rng(np.random.SeedSequence(7)))
    c = gen.draw(np.random.default_rng(np.random.SeedSequence(8)))
    assert a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    assert any(not _equal(a[k], c[k]) for k in a)  # another stream, another dataset
    with pytest.raises(AttributeError):
        gen.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: type(g).__name__)
def test_generator_truth_holds_on_a_large_draw(gen):
    checks = gen.verify(n_draws=100) if isinstance(gen, CreditRiskPlusSample) else gen.verify(n_draws=1_000)
    assert checks and all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]


def test_generators_refuse_bad_parameters():
    for bad in (
        lambda: PfeSeries(1, 0.95),
        lambda: PfeSeries(100, 1.0),
        lambda: PfeSeries(100, 0.95, forecast_factor=0.0),
        lambda: EsResiduals(100, 0.0),
        lambda: LhpLossRates(0),
        lambda: LhpLossRates(10, true_rho=1.0),
        lambda: StageTransitions((10, 10), STAGE_MATRIX),
        lambda: StageTransitions((10, 10, 10), ((0.5, 0.6, 0.0),) * 3),
        lambda: StageTransitions((10, 10, 10), STAGE_MATRIX, ((0.1, 0.0, 0.0),) * 3),
        lambda: StageTransitions((10, 10, 10), STAGE_MATRIX, ((-0.95, 0.95, 0.0),) * 3),
        lambda: LgdCalibration(1),
        lambda: LgdCalibration(10, n_periods=2, biased_periods=(2,)),
        lambda: LgdCalibration(10, bias=0.5),
        lambda: DownturnLgd(1),
        lambda: DownturnLgd(10, estimate=0.9, shortfall=0.2),
        lambda: McQuantile(10),
        lambda: McQuantile(1000, se_method="bootstrap"),
        lambda: CreditRiskPlusSample(5),
        lambda: CreditRiskPlusSample(100, obligors=(0, 0)),
        lambda: CreditRiskPlusSample(100, sigma2=0.0),
    ):
        with pytest.raises(ValueError):
            bad()


def test_the_closed_form_anchor_is_the_engines_and_the_null_flags_are_right():
    """The anchor of the Monte Carlo study is the closed form (Vasicek 2002), and equals the engine's rung C1."""
    for q, pd, rho in ((0.99, 0.02, 0.12), (0.999, 0.003, 0.06), (0.95, 0.1, 0.2)):
        assert lhp_quantile_closed_form(q, pd, rho) == pytest.approx(float(lhp_quantile(q, pd, rho)), rel=1e-12)
    assert McQuantile(1000).truth()["null_holds"] and not McQuantile(1000, mc_rho=0.13).truth()["null_holds"]
    assert PfeSeries(100).truth()["null_holds"] and not PfeSeries(100, forecast_factor=0.9).truth()["null_holds"]
    assert EsResiduals(100).truth()["null_holds"] and not EsResiduals(100, es_factor=0.9).truth()["null_holds"]
    assert LhpLossRates(10).truth()["null_holds"] and not LhpLossRates(10, true_rho=0.3).truth()["null_holds"]
    assert StageTransitions((10, 10, 10), STAGE_MATRIX).truth()["null_holds"]
    assert LgdCalibration(10).truth()["null_holds"] and DownturnLgd(10).truth()["null_holds"]
    assert CreditRiskPlusSample(100).truth()["null_holds"]
    assert not CreditRiskPlusSample(100, mc_sigma2=0.6).truth()["null_holds"]


def test_the_standard_error_of_the_sample_quantile():
    """The asymptotic standard error is the standard deviation of the estimates; Woodruff's is close to it."""
    gen = McQuantile(20000)
    rng = np.random.default_rng(np.random.SeedSequence(31))
    est = np.array([gen.draw(rng)["estimate"] for _ in range(400)])
    assert gen.asymptotic_se == pytest.approx(float(est.std(ddof=1)), rel=0.12)
    x = np.sort(np.random.default_rng(1).standard_normal(20000))
    e, se = sample_quantile_with_se(x, 0.99)
    assert e == pytest.approx(2.326, abs=0.12) and 0.02 < se < 0.035
    assert math.isfinite(se)


def test_the_recursion_pmf_has_the_closed_form_moments_and_covers_the_sample():
    """The anchor of the CreditRisk+ study is the recursion's pmf, which has the closed-form mean and variance."""
    gen = CreditRiskPlusSample(2000)
    k = np.arange(gen.grid + 1)
    f = gen.pmf
    assert f.min() >= 0.0 and f.sum() == pytest.approx(1.0, abs=1e-9)
    tr = gen.truth()
    assert float(f @ k) == pytest.approx(tr["mean_units"], rel=1e-6)
    assert float(f @ k**2 - (f @ k) ** 2) == pytest.approx(tr["recursion_variance_units"], rel=1e-6)
    assert gen.draw(np.random.default_rng(3))["mc_units"].max() < gen.grid


def test_the_registry_of_the_studies_covers_the_nine_and_their_seeds_are_distinct():
    nine = {"ccr.pfe_exceedance_backtest", "ifrs9.stage_migration_backtest", "lgd_ead.lgd_calibration_by_segment",
            "lgd_ead.lgd_calibration_over_time", "lgd_ead.downturn_coverage", "market.mcneil_frey_residual_test",
            "portfolio.tail_backtest", "portfolio.mc_convergence", "portfolio.crp_recursion_vs_mc"}
    assert set(STUDIES) == nine
    import riskvalidation.validation  # noqa: F401  (registers every test)
    from test_size import SEEDS  # the seeds of the null scenarios of the older families

    registered = {s.test_id: s for s in catalogue()}
    seeds = []
    for tid, studies in STUDIES.items():
        assert registered[tid].decision == "p_value"
        for st in studies:
            assert isinstance(st, Study) and st.test_id == tid and st.n_rep == N_REP >= 4000
            assert st.null.truth()["null_holds"] and not st.alternative.truth()["null_holds"], (tid, st.label)
            assert st.alternative_text
            seeds += [st.seed, st.power_seed]
    assert len(set(seeds)) == len(seeds) and not set(seeds) & set(SEEDS.values())
