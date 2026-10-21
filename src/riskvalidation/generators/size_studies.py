"""The size and power studies of the nine p-value tests that had none (U16-E2, BL-088): each is a named, importable
configuration of a null scenario and of one alternative, with its seed and the keyword arguments of the test.

A study is data: ``Study.null`` is a generator at the boundary of the test's null with its assumptions met,
``Study.alternative`` the same generator with one stated departure, ``Study.test_kwargs`` the fixed arguments of the
test. ``STUDIES[test_id]`` lists the studies of a test (the tail back-test has two levels). The `sizepower` suites
under ``tests/`` run them with `riskvalidation.harness.simulate`; Contraste's C22 reads the same names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from riskvalidation.fairness.metrics import group_selection_rates
from riskvalidation.fairness.plants import ThresholdBiasPlant
from riskvalidation.generators._base import GeneratorBase
from riskvalidation.generators.credit import DownturnLgd, LgdCalibration, StageTransitions
from riskvalidation.generators.exceedances import EsResiduals, LhpLossRates, PfeSeries
from riskvalidation.generators.hits import EsLosses, PitSeries, VarHits
from riskvalidation.generators.montecarlo import CreditRiskPlusSample, McQuantile
from riskvalidation.generators.pipelines import SHIFT_N, run_conformal, run_shift
from riskvalidation.harness.run import Rule, Simulation, light_not_green, simulate
from riskvalidation.ml.plants import CovariateShiftPlant, ExchangeabilityBreakPlant

# how a study reads a rejection from a result: p < alpha at 5% and 1% ("p"), p <= 0.05 for a Monte Carlo p-value that
# can equal 0.05 ("p_le"), the light is red ("red", the back-tests' red level 5%), the light is not green ("alarm")
RULES = ("p", "p_le", "red", "alarm")

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
    null_text: str = ""
    feed: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    null_runner: Callable[..., Any] | None = None  # a pipeline that runs the test on a draw (default: the test itself)
    alt_runner: Callable[..., Any] | None = None  # the pipeline of the alternative (default: null_runner)
    rule: str = "p"
    power_n_rep: int | None = None  # repetitions of the power run (default: n_rep)

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule is one of {RULES}, got {self.rule!r}")


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


def _two_group_feed(draw: dict[str, Any]) -> dict[str, Any]:
    """The two-by-two table of favourable decisions of groups 0 and 1, as the fairness z and Fisher tests read it."""
    r = group_selection_rates(draw["decision"], draw["group"])
    return {"model_id": "sp", "x1": r["0"].favourable, "n1": r["0"].n, "x2": r["1"].favourable, "n2": r["1"].n}


_GAP_TEXT = "the protected group's decision threshold is lower by 0.3 in log-odds (a selection-rate gap above 8 points)"
_CLUSTER_TEXT = "a hit is followed by a hit with probability {p}, not {q} (clustered exceedances at the same hit rate)"
_SHIFT_TEXT = "a quarter of the rows shifted 3 sd on feature 0"
_SHIFT_NULL, _SHIFT_ALT = CovariateShiftPlant(SHIFT_N, 0.0, [0], 1), CovariateShiftPlant(SHIFT_N, 3.0, [0], 1)
_ML_SEED = 20261019  # the seed of the ML size study (tests/ml/ml_sims.py); the defect run uses the next one

# The twelve p-value tests of the non-credit families that were measured in suites with their scenario written inline
# (the fairness z and Fisher, the market back-tests, the two ML tests): the same scenarios, now named here. n_rep is the
# repetitions of the suite that measures the size (the market suites are CI tests on one shared stream per suite).
_MORE: dict[str, tuple[Study, ...]] = {
    "fair.two_proportion_z": (Study(
        test_id="fair.two_proportion_z", label="2000 applicants, two groups",
        null=ThresholdBiasPlant(n=2000, gap=0.0, seed=1), alternative=ThresholdBiasPlant(n=2000, gap=0.3, seed=1),
        alternative_text=_GAP_TEXT, null_text="no threshold gap", seed=101, power_seed=102, feed=_two_group_feed),),
    "fair.fisher_exact": (Study(
        test_id="fair.fisher_exact", label="2000 applicants, two groups",
        null=ThresholdBiasPlant(n=2000, gap=0.0, seed=1), alternative=ThresholdBiasPlant(n=2000, gap=0.3, seed=1),
        alternative_text=_GAP_TEXT, null_text="no threshold gap", seed=103, power_seed=104, feed=_two_group_feed),),
    "market.kupiec_pof": (Study(
        test_id="market.kupiec_pof", label="1500 days at 5%",
        null=VarHits(1500, 0.05), alternative=VarHits(1500, 0.05, rate_factor=1.3),
        alternative_text="the VaR is exceeded 1.3 times too often (a hit rate of 6.5%, not 5%)",
        null_text="independent hits at 5%", seed=101, power_seed=5311, n_rep=700, power_n_rep=2000, rule="red"),),
    "market.christoffersen_independence": (Study(
        test_id="market.christoffersen_independence", label="1500 days at 5%",
        null=VarHits(1500, 0.05), alternative=VarHits(1500, 0.05, cluster=0.15),
        alternative_text=_CLUSTER_TEXT.format(p="15%", q="5%"), null_text="independent hits at 5%",
        seed=101, power_seed=5312, n_rep=700, power_n_rep=2000, rule="red"),),
    "market.christoffersen_conditional_coverage": (Study(
        test_id="market.christoffersen_conditional_coverage", label="1500 days at 5%",
        null=VarHits(1500, 0.05), alternative=VarHits(1500, 0.05, cluster=0.15),
        alternative_text=_CLUSTER_TEXT.format(p="15%", q="5%"), null_text="independent hits at 5%",
        seed=101, power_seed=5313, n_rep=700, power_n_rep=2000, rule="red"),),
    "market.duration_test": (Study(
        test_id="market.duration_test", label="400 days at 5%",
        null=VarHits(400, 0.05, with_var=False), alternative=VarHits(400, 0.05, cluster=0.40, with_var=False),
        alternative_text=_CLUSTER_TEXT.format(p="40%", q="5%"), null_text="independent hits at 5%",
        seed=102, power_seed=5314, n_rep=240, power_n_rep=1000, rule="p_le", test_kwargs={"n_mc": 99}),),
    "market.acerbi_szekely": (Study(
        test_id="market.acerbi_szekely", label="250 days at 2.5%",
        null=EsLosses(250, 0.025), alternative=EsLosses(250, 0.025, scale=1.25),
        alternative_text="the realised losses are 1.25 times as dispersed as the VaR and ES forecasts say",
        null_text="normal losses with their true VaR and ES", seed=103, power_seed=5315, n_rep=300,
        power_n_rep=1000, rule="p_le", test_kwargs={"n_sim": 300}),),
    "market.du_escanciano_unconditional": (Study(
        test_id="market.du_escanciano_unconditional", label="2000 days at 10%",
        null=PitSeries(2000, 0.1), alternative=PitSeries(2000, 0.1, tail_shape=1.05),
        alternative_text="the lower tail has too much mass (u = r^1.05: 11.2% of the PITs below 0.10, not 10%)",
        null_text="uniform PITs", seed=103, power_seed=5316, n_rep=600, power_n_rep=2000),),
    "market.du_escanciano_conditional": (Study(
        test_id="market.du_escanciano_conditional", label="2000 days at 10%",
        null=PitSeries(2000, 0.1), alternative=PitSeries(2000, 0.1, cluster=0.20),
        alternative_text="the tail PITs (below 0.10) cluster: one follows another with probability 20%, not 10%",
        null_text="uniform PITs", seed=103, power_seed=5317, n_rep=600, power_n_rep=2000,
        test_kwargs={"lags": 5}),),
    "market.dynamic_quantile_test": tuple(
        Study(test_id="market.dynamic_quantile_test", label=label, null=VarHits(t, a),
              alternative=VarHits(t, a, cluster=c),
              alternative_text=_CLUSTER_TEXT.format(p=f"{100 * c:g}%", q=f"{100 * a:g}%"),
              null_text="independent hits", seed=seed, power_seed=power_seed, n_rep=n, power_n_rep=1000, rule="p_le",
              test_kwargs={"lags": 4, "n_mc": 99})
        for label, t, a, c, seed, power_seed, n in (
            ("1500 days at 5%", 1500, 0.05, 0.15, 101, 5318, 700),
            ("250 days at 1%", 250, 0.01, 0.10, 104, 5319, 2000),
            ("250 days at 2.5%", 250, 0.025, 0.15, 104, 5320, 2000),
            ("750 days at 1%", 750, 0.01, 0.10, 104, 5321, 2000))),
    "ml.shift_resilience": (Study(
        test_id="ml.shift_resilience", label="2000 development rows, a quarter worst-subsample share",
        null=_SHIFT_NULL, alternative=_SHIFT_ALT, alternative_text=_SHIFT_TEXT, null_text="shift 0",
        seed=_ML_SEED, power_seed=_ML_SEED + 1, null_runner=run_shift(_SHIFT_NULL, 0.25),
        alt_runner=run_shift(_SHIFT_ALT, 0.25), rule="alarm"),),
    "ml.conformal_coverage": (Study(
        test_id="ml.conformal_coverage", label="2000 test rows, 90% coverage",
        null=ExchangeabilityBreakPlant(2000, 0.0, 1), alternative=ExchangeabilityBreakPlant(2000, 0.5, 1),
        alternative_text="log-odds drift 0.5 between calibration and test", null_text="drift 0",
        seed=_ML_SEED, power_seed=_ML_SEED + 1, null_runner=run_conformal(), rule="alarm"),),
}
STUDIES.update(_MORE)


def _rules(rule: str) -> tuple[tuple[float, ...], list[Rule]]:
    """The harness arguments (alphas, rules) of a study's rejection rule."""
    if rule == "p":
        return (0.05, 0.01), []
    if rule == "p_le":
        return (), [Rule("p<=0.05", lambda r: (r.p_value <= 0.05) if r.p_value is not None and r.p_value == r.p_value
                         else None)]
    if rule == "red":
        return (), [Rule("red", lambda r: None if r.light.value in ("not_evaluated", "error")
                         else r.light.value == "red")]
    return (), [light_not_green("alarm")]


def rejection_label(study: Study) -> str:
    """The label of the rate of the study's own rule: `p<0.05` for the p rule, else the rule's label."""
    return {"p": "p<0.05", "p_le": "p<=0.05", "red": "red", "alarm": "alarm"}[study.rule]


def simulate_study(study: Study, which: str = "null", *, n_rep: int | None = None,
                   seed: int | None = None) -> Simulation:
    """Run the null or the alternative of a study with `harness.simulate`: its generator, seed, feed, fixed arguments
    and rejection rule. The one entry point of the suites and of Contraste's C22, so that a scenario is never
    rebuilt."""
    if which not in ("null", "alternative"):
        raise ValueError("which is 'null' or 'alternative'")
    null = which == "null"
    runner = study.null_runner if null else (study.alt_runner or study.null_runner)
    alphas, rules = _rules(study.rule)
    return simulate(runner or study.test_id, study.null if null else study.alternative,
                    n_rep=n_rep or (study.n_rep if null else (study.power_n_rep or study.n_rep)),
                    seed=seed if seed is not None else (study.seed if null else study.power_seed), feed=study.feed,
                    alphas=alphas, rules=rules, test_kwargs=study.test_kwargs)


def study(test_id: str, index: int = 0) -> Study:
    """The ``index``-th study of a test."""
    return STUDIES[test_id][index]


__all__ = ["N_REP", "RULES", "STAGE_MATRIX", "STAGE_SHIFT", "STAGE_ROWS", "STUDIES", "Study", "rejection_label",
           "simulate_study", "study"]
