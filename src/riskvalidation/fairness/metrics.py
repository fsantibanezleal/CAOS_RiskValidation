"""Group fairness metrics: selection rates, demographic parity, equalized odds, the four-fifths flag, calibration and
AUC within groups (dossier 06 C.7 l.1029-1040; dossier 04 J.3 l.897-905; contract `engine-fairness-api` 1.0.0).

Every comparison of two rates is made on the counts, in exact integer arithmetic, and rounded once when it becomes a
float: a rate difference or ratio that sits exactly on a convention (a tolerance, the four-fifths line) is then decided
on the exact value and not on a float that missed it by one unit in the last place (60/100 over 75/100 is
0.7999999999999999 in IEEE doubles and exactly 4/5). The four-fifths line is an employment benchmark (29 CFR 1607.4(D))
and never a credit-law threshold. The outcome-conditioned metrics need an outcome and never run on a dataset that has
none (HMDA has no outcome). Groups are named by `str(label)` and ordered by their label (numeric labels numerically).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from riskvalidation.fairness._checks import (
    InsufficientData,
    binary,
    group_labels,
    outcome,
    same_rows,
)
from riskvalidation.validation.calibration import pd_ece
from riskvalidation.validation.discrimination import disc_auc

FOUR_FIFTHS_THRESHOLD = 0.8
EMPLOYMENT_BENCHMARK = "employment benchmark"
HIGHEST_RATE_GROUP = "highest-rate-group"


@dataclass(frozen=True)
class SelectionRate:
    """One group's size, favourable count and selection rate (rate == favourable / n)."""

    group: str
    n: int
    favourable: int
    rate: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if not 0 <= self.favourable <= self.n:
            raise ValueError(f"favourable must lie in [0, n], got {self.favourable} of {self.n}")
        if self.rate != self.favourable / self.n:
            raise ValueError(f"rate must equal favourable / n = {self.favourable / self.n}, got {self.rate}")


def group_selection_rates(decision: np.ndarray, group: np.ndarray) -> dict[str, SelectionRate]:
    """The selection rate of each group present (a group with no rows is absent), ordered by label."""
    d = binary(decision, "decision")
    g = group_labels(group, d.shape[0], "group")
    labels, inverse = np.unique(g, return_inverse=True)
    n = np.bincount(inverse, minlength=labels.size)
    fav = np.bincount(inverse, weights=d, minlength=labels.size).astype(np.int64)
    return {
        str(lab): SelectionRate(str(lab), int(n[i]), int(fav[i]), int(fav[i]) / int(n[i]))
        for i, lab in enumerate(labels)
    }


def _need(rates: Mapping[str, SelectionRate], reference: str | None = None) -> None:
    if not isinstance(rates, Mapping) or not rates:
        raise ValueError("rates must be a non-empty mapping of group to SelectionRate")
    for lab, s in rates.items():
        if not isinstance(s, SelectionRate):
            raise ValueError(f"rates[{lab!r}] must be a SelectionRate")
    if reference is not None and reference not in rates:
        raise ValueError(f"reference {reference!r} is not one of the groups {list(rates)}")


def demographic_parity_difference(rates: Mapping[str, SelectionRate], reference: str) -> dict[str, float]:
    """Rate of each group minus the reference rate, computed on the counts (exact) and rounded once."""
    _need(rates, reference)
    r = rates[reference]
    return {g: (s.favourable * r.n - r.favourable * s.n) / (s.n * r.n) for g, s in rates.items()}


def demographic_parity_ratio(rates: Mapping[str, SelectionRate], reference: str) -> dict[str, float | None]:
    """Rate of each group over the reference rate; None for every group when the reference rate is 0."""
    _need(rates, reference)
    r = rates[reference]
    if r.favourable == 0:
        return {g: None for g in rates}
    return {g: (s.favourable * r.n) / (s.n * r.favourable) for g, s in rates.items()}


def adverse_impact_ratio(rates: Mapping[str, SelectionRate], reference: str) -> dict[str, float | None]:
    """The protected group's selection rate over the chosen reference group's: the parity ratio. It is not the
    four-fifths ratio, which is taken against the highest-rate group (`four_fifths_flag`)."""
    return demographic_parity_ratio(rates, reference)


@dataclass(frozen=True)
class FourFifths:
    """The four-fifths flag per group against the highest-rate group, labelled an employment benchmark."""

    highest_rate_group: str
    ratio_to_highest: dict[str, float | None]
    flag: dict[str, bool]
    threshold: float = FOUR_FIFTHS_THRESHOLD
    label: str = EMPLOYMENT_BENCHMARK
    is_credit_law_threshold: bool = False
    compared_with: str = HIGHEST_RATE_GROUP


def four_fifths_flag(rates: Mapping[str, SelectionRate]) -> FourFifths:
    """A group is flagged when 5 x_g n_h < 4 x_h n_g, with x the favourable count and n the size, h the highest-rate
    group (29 CFR 1607.4(D); dossier 06 C.7 l.1034-1035). The highest-rate group is the first of the groups of the
    highest rate, in the order of `rates`. The ratio is for display; the flag never reads it."""
    _need(rates)
    items = list(rates.items())
    top_name, top = items[0]
    for name, s in items[1:]:
        if s.favourable * top.n > top.favourable * s.n:  # strictly higher rate, by cross-multiplication
            top_name, top = name, s
    ratio: dict[str, float | None] = {}
    flag: dict[str, bool] = {}
    for name, s in items:
        ratio[name] = None if top.favourable == 0 else (s.favourable * top.n) / (s.n * top.favourable)
        flag[name] = 5 * s.favourable * top.n < 4 * top.favourable * s.n
    return FourFifths(highest_rate_group=top_name, ratio_to_highest=ratio, flag=flag)


# ---------------------------------------------------------------------------------------------------------------
# Outcome-conditioned metrics: they need an outcome


@dataclass(frozen=True)
class _Confusion:
    positives: int
    negatives: int
    true_positives: int
    false_positives: int


def _confusion(y: np.ndarray, d: np.ndarray, g: np.ndarray) -> dict[str, _Confusion]:
    labels, codes = np.unique(g, return_inverse=True)
    k = labels.size

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=mask, minlength=k).astype(np.int64)

    pos, neg = count(y == 1), count(y == 0)
    tp, fp = count((y == 1) & (d == 1)), count((y == 0) & (d == 1))
    return {str(lab): _Confusion(int(pos[i]), int(neg[i]), int(tp[i]), int(fp[i])) for i, lab in enumerate(labels)}


def outcome_rates(y: np.ndarray, decision: np.ndarray, group: np.ndarray) -> dict[str, dict[str, float | None]]:
    """The true positive rate and false positive rate of each group (None where the class is absent)."""
    yy = outcome(y, "y")
    d = binary(decision, "decision")
    same_rows(yy, "y", d, "decision")
    g = group_labels(group, yy.shape[0], "group")
    return {
        lab: {
            "tpr": None if c.positives == 0 else c.true_positives / c.positives,
            "fpr": None if c.negatives == 0 else c.false_positives / c.negatives,
        }
        for lab, c in _confusion(yy, d, g).items()
    }


def _gaps(
    y: np.ndarray, decision: np.ndarray, group: np.ndarray, *, need_negatives: bool
) -> tuple[float, float | None]:
    yy = outcome(y, "y")
    d = binary(decision, "decision")
    same_rows(yy, "y", d, "decision")
    g = group_labels(group, yy.shape[0], "group")
    conf = _confusion(yy, d, g)
    if len(conf) < 2:
        raise InsufficientData("at least two groups are needed to compare their error rates")
    for lab, c in conf.items():
        if c.positives == 0:
            raise InsufficientData(f"group {lab!r} has no positives (y = 1): the true positive rate is undefined")
        if need_negatives and c.negatives == 0:
            raise InsufficientData(f"group {lab!r} lacks one of both classes: needs both classes of y")
    tprs = [Fraction(c.true_positives, c.positives) for c in conf.values()]
    tpr_gap = float(max(tprs) - min(tprs))
    if not need_negatives:
        return tpr_gap, None
    fprs = [Fraction(c.false_positives, c.negatives) for c in conf.values()]
    return tpr_gap, float(max(fprs) - min(fprs))


def equalized_odds_difference(y: np.ndarray, decision: np.ndarray, group: np.ndarray) -> float:
    """The larger of the true positive rate gap and the false positive rate gap across groups (Hardt, Price and
    Srebro 2016). Needs both classes of y in every group (`InsufficientData`, a `ValueError`, otherwise) and an outcome:
    never run on HMDA."""
    tpr_gap, fpr_gap = _gaps(y, decision, group, need_negatives=True)
    return max(tpr_gap, fpr_gap)  # type: ignore[type-var]


def equal_opportunity_difference(y: np.ndarray, decision: np.ndarray, group: np.ndarray) -> float:
    """The true positive rate gap across groups. Needs positives in every group and an outcome: never run on HMDA."""
    return _gaps(y, decision, group, need_negatives=False)[0]


def calibration_within_groups(p: np.ndarray, y: np.ndarray, group: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """The expected calibration error of PD `p` against the default indicator `y` within each group, by the engine's
    `pd_ece` (quantile bins). A group with fewer rows than `n_bins` is not evaluated and is absent from the result; the
    caller counts it as the groups of `group` minus the keys returned."""
    yy = outcome(y, "y")
    pv = np.array(p, dtype=float)
    if pv.ndim != 1 or pv.shape[0] != yy.shape[0]:
        raise ValueError(f"p must be one-dimensional with the {yy.shape[0]} rows of y, got shape {pv.shape}")
    if not np.all(np.isfinite(pv)) or np.any(pv <= 0.0) or np.any(pv >= 1.0):
        raise ValueError("p must lie strictly between 0 and 1")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    g = group_labels(group, yy.shape[0], "group")
    out: dict[str, float] = {}
    for lab in np.unique(g):
        m = g == lab
        if int(m.sum()) < n_bins:
            continue
        out[str(lab)] = float(pd_ece(yy[m], pv[m], bins=int(n_bins)).metric)
    return out


def auc_by_group(score: np.ndarray, y: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """The AUC of `score` within each group, by the engine's `disc_auc` (y = 1 is the class the score ranks higher). A
    group with a single class is omitted; the caller counts it as the groups of `group` minus the keys returned."""
    yy = outcome(y, "y")
    s = np.array(score, dtype=float)
    if s.ndim != 1 or s.shape[0] != yy.shape[0]:
        raise ValueError(f"score must be one-dimensional with the {yy.shape[0]} rows of y, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("score contains non-finite values")
    g = group_labels(group, yy.shape[0], "group")
    out: dict[str, float] = {}
    for lab in np.unique(g):
        m = g == lab
        ym = yy[m]
        if ym.min() == ym.max():
            continue
        out[str(lab)] = float(disc_auc(s[m], ym).metric)
    return out


__all__ = [
    "EMPLOYMENT_BENCHMARK",
    "FOUR_FIFTHS_THRESHOLD",
    "FourFifths",
    "SelectionRate",
    "adverse_impact_ratio",
    "auc_by_group",
    "calibration_within_groups",
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "four_fifths_flag",
    "group_selection_rates",
    "outcome_rates",
]
