"""Population and characteristic stability: has the population the model scores drifted from development?

The PSI bands of 0.10 and 0.25 are an industry convention "used without reference to statistical type I or
type II error rates"; Yurdakul and Naranjo (2020) show that (1/n + 1/m)^-1 PSI is approximately chi-square with
B - 1 degrees of freedom for B fixed bins, which gives a benchmark that holds the type I error at the chosen
level. Both are reported; the light follows the benchmark's p-value.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult
from riskvalidation.validation._common import as_float_array

YURDAKUL_NARANJO = (
    "Yurdakul and Naranjo (2020), Statistical properties of the population stability index, Journal of Risk "
    "Model Validation 14(4), doi:10.21314/JRMV.2020.227"
)
CONVENTIONAL_BANDS = ((0.10, "little change"), (0.25, "moderate change"))


def quantile_edges(reference: Sequence[float], bins: int = 10) -> np.ndarray:
    """Bin edges at the reference sample's quantiles, fixed before the current sample is seen.

    Edges are made unique; the outer edges are open (-inf, +inf) so every current value falls in a bin.
    """
    x = as_float_array(reference, "reference")
    if bins < 2:
        raise ValueError("at least two bins are required")
    inner = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_counts(sample: Sequence[float], edges: np.ndarray) -> np.ndarray:
    x = as_float_array(sample, "sample")
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1)


def _psi_core(expected: Sequence[float], actual: Sequence[float]) -> dict:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.ndim != 1 or e.size < 2:
        raise ValueError("expected and actual counts must be vectors over the same bins (at least two)")
    if np.any(e < 0) or np.any(a < 0):
        raise ValueError("counts must be non-negative")
    n, m = float(e.sum()), float(a.sum())
    if n <= 0 or m <= 0:
        raise ValueError("both samples must be non-empty")
    # an empty bin would make the logarithm infinite; it is floored at half an observation of its sample
    pe = np.maximum(e, 0.5) / n
    pa = np.maximum(a, 0.5) / m
    contrib = (pa - pe) * np.log(pa / pe)
    psi = float(contrib.sum())
    b = e.size
    scale = 1.0 / n + 1.0 / m
    band = next((label for cut, label in CONVENTIONAL_BANDS if psi < cut), "significant change")
    return {
        "psi": psi,
        "n": int(n),
        "m": int(m),
        "bins": b,
        "p_value": float(stats.chi2.sf(psi / scale, b - 1)),
        "benchmark_05": float(scale * stats.chi2.ppf(0.95, b - 1)),
        "band": band,
        "contributions": contrib.tolist(),
        "empty_bins_floored": int(np.sum((e == 0) | (a == 0))),
    }


def psi_benchmark(n: int, m: int, bins: int = 10, alpha: float = 0.05) -> float:
    """Yurdakul-Naranjo critical PSI: (1/n + 1/m) chi2_{1-alpha, B-1}."""
    return float((1.0 / n + 1.0 / m) * stats.chi2.ppf(1.0 - alpha, bins - 1))


def _psi_result(test_id: str, expected, actual, policy, model_id, segment) -> TestResult:
    core = _psi_core(expected, actual)
    return make_result(
        test_id,
        statistic=core["psi"],
        metric=core["psi"],
        p_value=core["p_value"],
        policy=policy,
        alternative="the binned distribution changed",
        n=core["m"],
        model_id=model_id,
        segment=segment,
        inputs=(np.asarray(expected, float), np.asarray(actual, float)),
        notes=f"conventional band: {core['band']} (0.10/0.25 rule of thumb, no error control)",
        extras={k: core[k] for k in ("n", "m", "bins", "benchmark_05", "band", "contributions",
                                     "empty_bins_floored")},
    )


@register(
    "stability.psi",
    family="stability",
    name="Population stability index",
    h0="the current population has the same binned distribution as the development population",
    reference=YURDAKUL_NARANJO,
    decision="p_value",
)
def stability_psi(
    expected_counts: Sequence[float],
    actual_counts: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """PSI = sum (a_i - e_i) ln(a_i / e_i) on fixed bins, with the chi-square benchmark p-value."""
    return _psi_result("stability.psi", expected_counts, actual_counts, policy, model_id, segment)


@register(
    "stability.csi",
    family="stability",
    name="Characteristic stability index",
    h0="the current distribution of one model input equals its development distribution",
    reference=f"{YURDAKUL_NARANJO}; the PSI formula applied to one characteristic (scorecard practice)",
    decision="p_value",
)
def stability_csi(
    expected_counts: Sequence[float],
    actual_counts: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The PSI of one characteristic; pass the characteristic name as ``segment``."""
    return _psi_result("stability.csi", expected_counts, actual_counts, policy, model_id, segment)


@register(
    "stability.chi2",
    family="stability",
    name="Chi-square test of homogeneity",
    h0="development and current samples share one binned distribution",
    reference="Pearson chi-square test of homogeneity on a 2 x B contingency table (SciPy chi2_contingency)",
    decision="p_value",
)
def stability_chi2(
    expected_counts: Sequence[float],
    actual_counts: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    table = np.vstack([np.asarray(expected_counts, float), np.asarray(actual_counts, float)])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        raise ValueError("at least two non-empty bins are required")
    res = stats.chi2_contingency(table, correction=False)
    return make_result(
        "stability.chi2",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        policy=policy,
        alternative="the distributions differ",
        n=int(table[1].sum()),
        model_id=model_id,
        segment=segment,
        inputs=(table,),
        extras={"dof": int(res.dof)},
    )


@register(
    "stability.ks",
    family="stability",
    name="Two-sample Kolmogorov-Smirnov test",
    h0="development and current samples come from the same continuous distribution",
    reference="Two-sample Kolmogorov-Smirnov test (SciPy ks_2samp)",
    decision="p_value",
)
def stability_ks(
    development: Sequence[float],
    current: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    x = as_float_array(development, "development")
    y = as_float_array(current, "current")
    res = stats.ks_2samp(x, y)
    return make_result(
        "stability.ks",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        policy=policy,
        alternative="the distributions differ",
        n=int(y.size),
        model_id=model_id,
        segment=segment,
        inputs=(x, y),
    )


@register(
    "stability.wasserstein",
    family="stability",
    name="Wasserstein-1 distance",
    h0="descriptive: area between the two empirical distribution functions, in the variable's units",
    reference="Wasserstein-1 distance W1(F, G) = integral |F(x) - G(x)| dx (SciPy wasserstein_distance)",
    decision="metric",
)
def stability_wasserstein(
    development: Sequence[float],
    current: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    x = as_float_array(development, "development")
    y = as_float_array(current, "current")
    return make_result(
        "stability.wasserstein",
        metric=float(stats.wasserstein_distance(x, y)),
        policy=policy,
        n=int(y.size),
        model_id=model_id,
        segment=segment,
        inputs=(x, y),
        notes="in the units of the variable; no default threshold exists in any regulatory source",
    )


__all__ = [
    "bin_counts",
    "psi_benchmark",
    "quantile_edges",
    "stability_chi2",
    "stability_csi",
    "stability_ks",
    "stability_psi",
    "stability_wasserstein",
]
