"""Situation testing with k-nearest-neighbour comparators, ceteris paribus against mutatis mutandis (dossier 07 A.3
l.158-177), transcribed from the companion paper, Alvarez and Ruggieri, "Counterfactual Situation Testing: From Single
to Multidimensional Discrimination", arXiv:2502.01267v2 = JAIR 82 (doi:10.1613/jair.1.17935), Section 4 and equations
3 to 10, and the main paper, arXiv:2405.13693v4 (doi:10.1111/coin.70238), for the comparators (contract
`engine-fairness-api` 1.0.0, `categorical` keyword of `situation_test` 1.1.0; research addendum 7.2 to 7.6).

Setting: the group indicator is 1 for the protected group (the control search space `D_c`) and 0 for the others (the
test search space `D_t`); the decision 0 is the negative one. For every complainant `c` of the protected group the
control group is the k nearest protected rows to `x_c` (the complainant excluded) and the test group is the k nearest
rows of the other group to `x_c` (ceteris paribus) or to the counterfactual `x_c^CF` (mutatis mutandis); neither the
group nor the decision enters the distance. The statistic is delta p = p_c - p_t, the share of negative decisions in
the control group minus that in the test group, with the denominator k (the "CST without centers" version). The claim
is statistically significant at level alpha when the lower bound of the one-sided Wald interval exceeds the accepted
deviation tau.

The distance normalization is a named choice. `range` is the paper's text (equation 5: the continuous term is divided by
max - min over the search space) and is the default. `standard` is the reference code's default (a scikit-learn
StandardScaler, the absolute z-score difference, not bounded by 1). The published counts of the loan-application
experiment were produced under `standard`, so they are claimed only there. Nothing here reads the reference code's
report columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.fairness._checks import binary, finite_matrix, same_rows, unit_interval_open

NORMALIZATIONS = ("range", "standard")
COMPARATORS = ("ceteris-paribus", "mutatis-mutandis")


@dataclass(frozen=True)
class SituationResult:
    """The counts of one situation test over the complainants of the protected group."""

    n_complainants: int
    n_cases: int
    n_significant: int
    k: int
    tau: float
    alpha: float

    def __post_init__(self) -> None:
        if not 0 <= self.n_significant <= self.n_cases <= self.n_complainants:
            raise ValueError("n_significant must not exceed n_cases, and n_cases must not exceed n_complainants")


def _normalization(normalization: str) -> str:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be 'range' or 'standard', got {normalization!r}")
    return normalization


def _attributes(x: object, name: str, p: int | None = None) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array of attributes")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    if p is not None and a.shape[0] != p:
        raise ValueError(f"{name} has {a.shape[0]} attributes but {p} were expected")
    return a


def _scale_and_mask(categorical: object, scale: object, p: int) -> tuple[np.ndarray, np.ndarray]:
    cat = np.asarray(categorical)
    if cat.shape != (p,):
        raise ValueError(f"categorical must have one flag per attribute ({p}), got shape {cat.shape}")
    cat = cat.astype(bool)
    sc = np.array(scale, dtype=float)
    if sc.shape != (p,):
        raise ValueError(f"scale must have one value per attribute ({p}), got shape {sc.shape}")
    for j in np.flatnonzero(~cat):
        if not (math.isfinite(sc[j]) and sc[j] > 0.0):
            raise ValueError(f"scale of continuous attribute {j} must be positive and finite, got {sc[j]!r}")
    return cat, np.where(cat, 1.0, sc)


def _distances(center: np.ndarray, pool: np.ndarray, cat: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Gower distance (equations 3 to 5) of `center` to every row of `pool`: the mean over the attributes of the
    overlap distance (categorical) or |difference| / scale (continuous)."""
    per = np.where(cat, (pool != center).astype(float), np.abs(pool - center) / scale)
    return per.sum(axis=1) / pool.shape[1]


def gower_distance(
    x1: np.ndarray, x2: np.ndarray, categorical: np.ndarray, scale: np.ndarray, normalization: str = "range"
) -> float:
    """Mean over the non-protected attributes of the overlap distance (categorical) or the Manhattan distance over
    the attribute scale (continuous). Under `range`, `scale` holds max - min of each continuous attribute over the
    search space, and the distance lies in [0, 1] for attributes inside that range. Under `standard`, `scale` holds the
    population standard deviation (ddof 0) of the attribute: the continuous term is the absolute z-score difference and
    is not bounded by 1. The arithmetic is the same; the normalization states what `scale` holds and any other value
    raises `ValueError`. Protected attributes never enter (the caller leaves them out); a non-positive scale of a
    continuous attribute raises `ValueError` (the scale of a categorical attribute is not read)."""
    _normalization(normalization)
    a = _attributes(x1, "x1")
    b = _attributes(x2, "x2", a.shape[0])
    cat, sc = _scale_and_mask(categorical, scale, a.shape[0])
    return float(_distances(a, b[None, :], cat, sc)[0])


def _select(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances in ascending order, ties broken by row order."""
    kth = np.partition(d, k - 1)[k - 1]
    below = np.flatnonzero(d < kth)
    ties = np.flatnonzero(d == kth)[: k - below.size]
    chosen = np.concatenate([below, ties])
    return chosen[np.lexsort((chosen, d[chosen]))]


def k_nearest(
    center: np.ndarray,
    pool: np.ndarray,
    k: int,
    categorical: np.ndarray,
    scale: np.ndarray,
    normalization: str = "range",
) -> np.ndarray:
    """Indices of the k nearest rows of `pool` to `center` by ascending distance (the normalization of
    `gower_distance`), ties broken by row order; neither the protected attribute nor the decision is used (equations 6
    and 7). The center itself is not removed: pass a pool without it."""
    _normalization(normalization)
    c = _attributes(center, "center")
    m = finite_matrix(pool, "pool", rows_of="pool")
    if m.shape[1] != c.shape[0]:
        raise ValueError(f"center has {c.shape[0]} attributes but pool has {m.shape[1]} columns")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= m.shape[0]:
        raise ValueError(f"k must be an integer in [1, {m.shape[0]}] (the rows of pool), got {k!r}")
    cat, sc = _scale_and_mask(categorical, scale, c.shape[0])
    return _select(_distances(c, m, cat, sc), int(k)).astype(np.int64)


def situation_statistic(
    decision_control: np.ndarray, decision_test: np.ndarray, negative: int = 0
) -> tuple[float, float, float]:
    """(delta_p, p_c, p_t): the share of `negative` decisions in the control group and in the test group (the
    denominator is the group size k, the centers excluded) and their difference p_c - p_t, in [-1, 1]
    (equations 8 and 9)."""
    if negative not in (0, 1):
        raise ValueError(f"negative must be 0 or 1, got {negative!r}")
    dc = binary(decision_control, "decision_control")
    dt = binary(decision_test, "decision_test")
    if dc.shape != dt.shape:
        raise ValueError(
            f"decision_test has {dt.shape[0]} rows but decision_control has {dc.shape[0]}: both groups have k rows"
        )
    p_c = float(np.mean(dc == negative))
    p_t = float(np.mean(dt == negative))
    return p_c - p_t, p_c, p_t


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha))


def _wald_lower(delta: float, p_c: float, p_t: float, k: int, z: float) -> float:
    return delta - z * math.sqrt((p_c * (1.0 - p_c) + p_t * (1.0 - p_t)) / k)


def wald_interval_one_sided(delta: float, p_c: float, p_t: float, k: int, alpha: float = 0.05) -> float:
    """The lower bound delta - z_alpha sqrt((p_c (1 - p_c) + p_t (1 - p_t)) / k) of the one-sided interval
    [lower, +inf) (equation 10), with z_alpha = Phi^-1(1 - alpha) the exact normal quantile and the plus sign of the
    numerator (footnote 7: the EAAMO paper's minus sign is a typo). The claim is significant when the lower bound
    exceeds tau."""
    for name, v in (("delta", delta), ("p_c", p_c), ("p_t", p_t)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    for name, v in (("p_c", p_c), ("p_t", p_t)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {v!r}")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    a = unit_interval_open(alpha, "alpha")
    return float(_wald_lower(delta, p_c, p_t, int(k), _z(a)))


def bonferroni_alpha(alpha: float, q: int) -> float:
    """The level alpha / q for q protected attributes (multiple discrimination, Definition 4.6)."""
    a = unit_interval_open(alpha, "alpha")
    if not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError(f"q must be an integer of at least 1, got {q!r}")
    return a / int(q)


def _scales(x: np.ndarray, protected: np.ndarray, others: np.ndarray, normalization: str, cat: np.ndarray):
    """The scale of each continuous attribute; a categorical attribute's scale is not read (overlap distance), so it
    is set to 1 and never raises for being constant."""
    if normalization == "range":
        ctrl = np.ptp(x[protected], axis=0)
        test = np.ptp(x[others], axis=0)
    else:
        ctrl = test = x.std(axis=0)
    ctrl, test = np.where(cat, 1.0, ctrl), np.where(cat, 1.0, test)
    for name, sc in (("control", ctrl), ("test", test)):
        bad = np.flatnonzero(~(sc > 0.0))
        if bad.size:
            raise ValueError(
                f"attribute {int(bad[0])} has a zero scale over the {name} search space (a constant column): "
                "remove it or choose another normalization"
            )
    return ctrl, test


def situation_test(
    X: np.ndarray,
    group: np.ndarray,
    decision: np.ndarray,
    k: int,
    tau: float = 0.0,
    alpha: float = 0.05,
    comparator: str = "ceteris-paribus",
    counterfactual: np.ndarray | None = None,
    normalization: str = "range",
    categorical: np.ndarray | None = None,
) -> SituationResult:
    """Runs the test for every complainant of the protected group (group 1), regardless of the complainant's own
    decision. `X` holds the non-protected attributes, `group` is 1 for the protected group and 0 for the others,
    `decision` is 1 favourable and 0 negative. `categorical` (1.1.0) is a boolean mask over the columns of `X`: a
    categorical column contributes the overlap measure (1 if the values differ, 0 otherwise), every other column the
    normalised Manhattan distance, as in equations 3 to 5 of the paper, and the distance is their mean over the columns;
    under `normalization="standard"` the categorical columns still use overlap and the standardisation applies to the
    other columns only. `None` (the 1.0.0 behaviour) treats every column as continuous.

    comparator 'ceteris-paribus': the test group is built around the complainant. 'mutatis-mutandis': around the row of
    `counterfactual` aligned with the complainant (the algorithm supplies none; it must have the shape of `X`). Under
    `range` the scale of each attribute is max - min over the search space the group is drawn from (control: the
    protected rows; test: the other rows); under `standard` it is the population standard deviation over all of `X`,
    shared by the counterfactual (the paper's one space), not the reference code's separate scaler for it.

    n_cases counts the complainants with delta p > tau; n_significant those whose lower bound of equation 10 exceeds
    tau. Raises `ValueError` naming the field for unknown comparators and normalizations, 'mutatis-mutandis' without a
    counterfactual, malformed inputs (a `categorical` mask that is not one flag per column of `X` included), k beyond
    the search spaces and a zero scale of a continuous column."""
    if comparator not in COMPARATORS:
        raise ValueError(f"comparator must be one of {COMPARATORS}, got {comparator!r}")
    _normalization(normalization)
    x = finite_matrix(X, "X", rows_of="the decision")  # rows are checked against group below
    g = binary(group, "group")
    d = binary(decision, "decision")
    same_rows(g, "group", x, "X")
    same_rows(g, "group", d, "decision")
    if g.min() == g.max():
        raise ValueError("group must contain both the protected group (1) and the others (0)")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    a = unit_interval_open(alpha, "alpha")
    if not (math.isfinite(tau) and -1.0 <= tau <= 1.0):
        raise ValueError(f"tau must lie in [-1, 1], got {tau!r}")
    if comparator == "mutatis-mutandis":
        if counterfactual is None:
            raise ValueError("counterfactual is required for the 'mutatis-mutandis' comparator")
        cf = np.array(counterfactual, dtype=float)
        if cf.shape != x.shape or not np.all(np.isfinite(cf)):
            raise ValueError(
                f"counterfactual must be a finite array of the shape of X {x.shape}, aligned row by row, got {cf.shape}"
            )
    else:
        cf = None

    prot_idx, other_idx = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    if k > prot_idx.size - 1:
        raise ValueError(f"k = {k} exceeds the {prot_idx.size - 1} other complainants of the control search space")
    if k > other_idx.size:
        raise ValueError(f"k = {k} exceeds the {other_idx.size} rows of the test search space")
    if categorical is None:
        cat = np.zeros(x.shape[1], dtype=bool)
    else:
        cat = np.asarray(categorical)
        if cat.shape != (x.shape[1],):
            raise ValueError(f"categorical must have one flag per column of X ({x.shape[1]}), got shape {cat.shape}")
        cat = cat.astype(bool)
    ctrl_scale, test_scale = _scales(x, prot_idx, other_idx, normalization, cat)
    xp, xo = x[prot_idx], x[other_idx]
    neg_p, neg_o = d[prot_idx] == 0, d[other_idx] == 0
    z = _z(a)
    cases = significant = 0
    for i in range(prot_idx.size):
        d_ctrl = _distances(xp[i], xp, cat, ctrl_scale)
        d_ctrl[i] = np.inf  # the complainant is the search center: excluded from its own control group
        centre = xp[i] if cf is None else cf[prot_idx[i]]
        d_test = _distances(centre, xo, cat, test_scale)
        p_c = float(neg_p[_select(d_ctrl, k)].sum()) / k
        p_t = float(neg_o[_select(d_test, k)].sum()) / k
        delta = p_c - p_t
        if delta > tau:
            cases += 1
        if _wald_lower(delta, p_c, p_t, k, z) > tau:
            significant += 1
    return SituationResult(int(prot_idx.size), cases, significant, k, float(tau), float(a))


__all__ = [
    "SituationResult",
    "bonferroni_alpha",
    "gower_distance",
    "k_nearest",
    "situation_statistic",
    "situation_test",
    "wald_interval_one_sided",
]
