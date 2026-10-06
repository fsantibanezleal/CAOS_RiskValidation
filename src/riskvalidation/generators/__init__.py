"""Known-truth generators: data-generating mechanisms whose parameters are known, so the size and power of a test
and the error of an estimator can be measured (see :mod:`riskvalidation.harness`).

Every generator is a frozen configuration with ``config()`` (the configuration written next to the data),
``draw(rng)`` (one dataset, keyed by the names of the test arguments it feeds), ``truth()`` (exact) and ``verify()``
(a large draw on the generator's own stream reproduces the truth within Monte Carlo error).
"""

from riskvalidation.generators._base import Check, GeneratorBase, bvn_equal, jsonable
from riskvalidation.generators.counts import (
    WP14,
    DefaultCounts,
    default_correlation,
    feed_cell,
    feed_grades,
    feed_portfolio,
    feed_series,
    joint_default_probability,
)
from riskvalidation.generators.drift import YURDAKUL_NARANJO, TwoSample, population_psi
from riskvalidation.generators.framingham import DEMLER, NestedLda, lda_auc
from riskvalidation.generators.ratings import (
    GradeFrequencies,
    Migrations,
    break_monotonicity,
    concentrate,
    decaying_matrix,
    monotone_cells,
)
from riskvalidation.generators.scores import (
    HANLEY_MCNEIL,
    BinormalScores,
    LogisticPortfolio,
    binormal_auc,
    binormal_auc_variance,
)

__all__ = [
    "DEMLER",
    "HANLEY_MCNEIL",
    "WP14",
    "YURDAKUL_NARANJO",
    "BinormalScores",
    "Check",
    "DefaultCounts",
    "GeneratorBase",
    "GradeFrequencies",
    "LogisticPortfolio",
    "Migrations",
    "NestedLda",
    "TwoSample",
    "binormal_auc",
    "binormal_auc_variance",
    "break_monotonicity",
    "bvn_equal",
    "concentrate",
    "decaying_matrix",
    "default_correlation",
    "feed_cell",
    "feed_grades",
    "feed_portfolio",
    "feed_series",
    "joint_default_probability",
    "jsonable",
    "lda_auc",
    "monotone_cells",
    "population_psi",
]
