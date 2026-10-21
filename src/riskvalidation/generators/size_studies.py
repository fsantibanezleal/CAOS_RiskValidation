"""The size and power studies of the nine p-value tests that had none (U16-E2, BL-088): each is a named, importable
configuration of a null scenario and of one alternative, with its seed and the keyword arguments of the test.

A study is data: ``Study.null`` is a generator at the boundary of the test's null with its assumptions met,
``Study.alternative`` the same generator with one stated departure, ``Study.test_kwargs`` the fixed arguments of the
test. ``STUDIES[test_id]`` lists the studies of a test (the tail back-test has two levels). The `sizepower` suites
under ``tests/`` run them with `riskvalidation.harness.simulate`; Contraste's C22 reads the same names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from riskvalidation.generators._base import GeneratorBase
from riskvalidation.generators.credit import DownturnLgd, LgdCalibration, StageTransitions
from riskvalidation.generators.exceedances import EsResiduals, LhpLossRates, PfeSeries
from riskvalidation.generators.montecarlo import CreditRiskPlusSample, McQuantile

N_REP = 4000


@dataclass(frozen=True)
class Study:
    """One size and power study: a test, its null scenario, its alternative, the seeds and the fixed arguments."""

    test_id: str
    label: str
    null: GeneratorBase
    alternative: GeneratorBase
    alternative_text: str
    seed: int
    power_seed: int
    test_kwargs: dict[str, Any] = field(default_factory=dict)
    n_rep: int = N_REP


# the predicted matrix of the stage migration study: stage 1 stays 90%, stage 2 78%, stage 3 90%
STAGE_MATRIX = ((0.90, 0.08, 0.02), (0.12, 0.78, 0.10), (0.02, 0.08, 0.90))
# the departure: 1.5 points of the stage 1 accounts move from staying to stage 2 (a stage 2 migration of 9.5%, not 8%)
STAGE_SHIFT = ((-0.015, 0.015, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
STAGE_ROWS = (4000, 1500, 500)  # the smallest expected count is 10 (stage 3 to stage 1): the chi-square assumption

STUDIES: dict[str, tuple[Study, ...]] = {
    "ccr.pfe_exceedance_backtest": (Study(
        test_id="ccr.pfe_exceedance_backtest", label="PFE at 95%, 1000 dates",
        null=PfeSeries(1000, 0.95), alternative=PfeSeries(1000, 0.95, forecast_factor=0.90),
        alternative_text="the PFE forecast is 90% of the true 95% quantile (exceedance frequency 6.94%, not 5%)",
        seed=5201, power_seed=5301),),
    "ifrs9.stage_migration_backtest": (Study(
        test_id="ifrs9.stage_migration_backtest", label="three stages, 4000, 1500 and 500 accounts",
        null=StageTransitions(STAGE_ROWS, STAGE_MATRIX),
        alternative=StageTransitions(STAGE_ROWS, STAGE_MATRIX, STAGE_SHIFT),
        alternative_text="the stage 1 to stage 2 migration is 9.5%, not the predicted 8% (stage 1 staying 88.5%)",
        seed=5202, power_seed=5302),),
    "lgd_ead.lgd_calibration_by_segment": (Study(
        test_id="lgd_ead.lgd_calibration_by_segment", label="one segment, 250 workouts",
        null=LgdCalibration(250), alternative=LgdCalibration(250, bias=0.05),
        alternative_text="the realised LGD is 5 points above the predicted one on average",
        seed=5203, power_seed=5303),),
    "lgd_ead.lgd_calibration_over_time": (Study(
        test_id="lgd_ead.lgd_calibration_over_time", label="five default years, 60 workouts each",
        null=LgdCalibration(60, n_periods=5), alternative=LgdCalibration(60, n_periods=5, bias=0.10,
                                                                         biased_periods=(4,)),
        alternative_text="in the last of five years the realised LGD is 10 points above the predicted one",
        seed=5204, power_seed=5304),),
    "lgd_ead.downturn_coverage": (Study(
        test_id="lgd_ead.downturn_coverage", label="150 downturn workouts, estimate 0.45",
        null=DownturnLgd(150), alternative=DownturnLgd(150, shortfall=0.05),
        alternative_text="the mean realised downturn LGD is 5 points above the 0.45 estimate",
        seed=5205, power_seed=5305),),
    "market.mcneil_frey_residual_test": (Study(
        test_id="market.mcneil_frey_residual_test", label="2500 days at 5%, 125 exceedances expected",
        null=EsResiduals(2500, 0.05), alternative=EsResiduals(2500, 0.05, es_factor=0.95),
        alternative_text="the ES forecast is 95% of the true 5% expected shortfall of a normal loss "
                         "(mean residual 0.103)",
        seed=5206, power_seed=5306, test_kwargs={"n_boot": 999}),),
    "portfolio.tail_backtest": (
        Study(test_id="portfolio.tail_backtest", label="60 years at the 95% level",
              null=LhpLossRates(60), alternative=LhpLossRates(60, true_rho=0.30),
              alternative_text="the true asset correlation is 0.30, the fit says 0.12 (the 95% quantile is exceeded "
                               "9.12% of the years, not 5%)",
              seed=5207, power_seed=5307, test_kwargs={"primary_level": 0.95}),
        Study(test_id="portfolio.tail_backtest", label="250 years at the 99% level",
              null=LhpLossRates(250), alternative=LhpLossRates(250, true_rho=0.30),
              alternative_text="the true asset correlation is 0.30, the fit says 0.12 (the 99% quantile is exceeded "
                               "4.29% of the years, not 1%)",
              seed=5208, power_seed=5308, test_kwargs={"primary_level": 0.99})),
    "portfolio.mc_convergence": (Study(
        test_id="portfolio.mc_convergence", label="20000 scenarios, the 99% quantile of the LHP loss rate",
        null=McQuantile(20000), alternative=McQuantile(20000, mc_rho=0.13),
        alternative_text="the run simulates correlation 0.13, the anchor is the closed form at 0.12 "
                         "(quantile 0.0964 against 0.0917)",
        seed=5209, power_seed=5309),),
    "portfolio.crp_recursion_vs_mc": (Study(
        test_id="portfolio.crp_recursion_vs_mc", label="2000 scenarios, 400 obligors in five exposure bands",
        null=CreditRiskPlusSample(2000), alternative=CreditRiskPlusSample(2000, mc_sigma2=0.6),
        alternative_text="the sample has sector variance 0.6, the recursion's pmf 0.5 (loss variance 172 units "
                         "against 150)",
        seed=5210, power_seed=5310),),
}


def study(test_id: str, index: int = 0) -> Study:
    """The ``index``-th study of a test."""
    return STUDIES[test_id][index]


__all__ = ["N_REP", "STAGE_MATRIX", "STAGE_SHIFT", "STAGE_ROWS", "STUDIES", "Study", "study"]
