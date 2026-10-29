"""Anytime-valid monitor of defaults against the predicted PD of each rating grade (BS-15).

At each step `t` (a year, a vintage or a quarter) grade `g` holds `n_{t,g}` obligors of which `d_{t,g}` default. The
null is that the PD of the grade is the predicted `p_g` and that, given the past, the defaults of a step are
Binomial(`n_{t,g}`, `p_g`). The alternative is a *multiplicative miscalibration*: the true PD is `m p_g`. The
monitor bets on that factor by a mixture over a log-spaced grid of `m` (default 16 points from 1/4 to 4, 1 excluded)
whose weights follow the Jeffreys prior of a binomial proportion, `pi(p) = p^{-1/2} (1-p)^{-1/2}`, taken on the
log-spaced grid (so the weight of `m` is proportional to `sqrt(m p_g / (1 - m p_g))`, the Jeffreys density times the
Jacobian `p` of the log scale) and normalised. The mixture is over the path, not over the steps:

    log LR_k(s) = d_s log m_k + (n_s - d_s) log((1 - m_k p_g) / (1 - p_g))
    E_{t,g}     = sum_k w_k exp(sum_{s<=t} log LR_k(s))

The grades combine into one e-process by `combine="product"` (the product of the grade e-processes; it is an
e-process when the grades' defaults are independent given the past, the assumption of the binomial test per grade, and
it concentrates the bet when many grades drift together) or `combine="average"` (the mean; an e-process with no
independence assumption, at a power cost when one grade carries the signal). Defaults are counts, obligors are counts,
the PD is a decimal per step; a step with no obligor in a grade contributes 1.

`direction`: `"upper"` (the true PD is higher than predicted, the case a validator is paid to find), `"lower"` or
`"two-sided"` (default). Sources: Ramdas, Grunwald, Vovk and Shafer (2023), Statistical Science 38(4):576-601,
arXiv 2210.01948; Ville (1939); Jeffreys (1946), Proc. R. Soc. A 186:453-461, doi:10.1098/rspa.1946.0056 for the prior;
the fixed-window tests it supplements are `pd_binomial`, `pd_jeffreys` and `pd_hosmer_lemeshow` of
`riskvalidation.validation`. To the author's knowledge the application of e-processes to PD calibration by vintage is
not published (UNVERIFIED).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.validation.sequential._core import (
    check_alpha_level,
    counts,
    log_mixture,
    multiplier_grid,
)
from riskvalidation.validation.sequential.monitor import GradeEProcessResult

REFERENCE = (
    "Ramdas, Grunwald, Vovk and Shafer (2023), Statistical Science 38(4):576-601, arXiv 2210.01948; Ville (1939); "
    "Jeffreys (1946), doi:10.1098/rspa.1946.0056"
)
_LOW = tuple(float(v) for v in np.geomspace(0.25, 1.0, 9)[:-1])
_HIGH = tuple(float(v) for v in np.geomspace(1.0, 4.0, 9)[1:])


@register(
    "sequential.pd_grades",
    family="sequential",
    name="PD calibration e-process by grade",
    h0="the defaults of every grade are Binomial(obligors, predicted PD) given the past",
    reference=REFERENCE,
    decision="p_value",
)
def pd_grade_eprocess(defaults: object, exposures: object, predicted_pd: object, *, alpha_level: float = 0.05,
                      direction: str = "two-sided", multipliers: Sequence[float] | None = None,
                      combine: str = "product", grades: Sequence[str] | None = None) -> GradeEProcessResult:
    """The e-process of observed defaults against the predicted PD of each grade, and the combined one.

    `defaults` and `exposures` are (steps, grades) arrays of whole numbers (defaults at most the obligors; NaN,
    fractions or negatives raise); `predicted_pd` the PD of each grade (a vector, constant over the steps) or of
    each step and grade (an array like `defaults`, for a PD that changes with age or vintage), decimals strictly
    between 0 and 1 (a zero or a certain PD makes any default infinitely surprising, so it raises: use a floor);
    `grades` labels the columns. The mixture weights use the mean predicted PD of the grade over the steps (fixed
    before any outcome, so the process stays an e-process). `alpha_level` is the size (default 0.05); `direction`
    and `multipliers` choose the alternatives (factors of the predicted PD, positive, not 1); `combine` is
    `"product"` or `"average"` (see the module docstring for what each assumes). Returns a `GradeEProcessResult`:
    `log_e` is the combined path, `grade_log_e` the path of each grade.
    """
    level = check_alpha_level(alpha_level)
    if combine not in ("product", "average"):
        raise ValueError(f"combine must be 'product' or 'average', got {combine!r}")
    d, n = counts(defaults, "defaults"), counts(exposures, "exposures")
    if d.shape != n.shape:
        raise ValueError(f"defaults {d.shape} and exposures {n.shape} must have the same shape")
    pd_in = np.asarray(predicted_pd, dtype=float)
    if pd_in.ndim == 1 and pd_in.size == d.shape[1]:
        pd = np.broadcast_to(pd_in, d.shape)
    elif pd_in.ndim == 2 and pd_in.shape == d.shape:
        pd = pd_in
    else:
        raise ValueError(f"predicted_pd must hold one PD per grade ({d.shape[1]}) or one per step and grade "
                         f"{d.shape}, got shape {pd_in.shape}")
    if not (np.all(np.isfinite(pd)) and np.all((pd > 0.0) & (pd < 1.0))):
        raise ValueError("predicted_pd must lie strictly between 0 and 1 for every grade")
    if np.any(d > n):
        raise ValueError("a grade has more defaults than obligors")
    labels = tuple(str(g) for g in grades) if grades is not None else tuple(f"g{j + 1}" for j in range(d.shape[1]))
    if len(labels) != d.shape[1]:
        raise ValueError("grades must name every column")
    base = multiplier_grid(multipliers, direction, _LOW, _HIGH)
    steps, n_grades = d.shape
    grade_log = np.zeros((steps, n_grades))
    used_m: list[list[float]] = []
    used_w: list[list[float]] = []
    for j in range(n_grades):
        pj = pd[:, j]
        pmax, pbar = (float(pj.max()), float(pj.mean())) if steps else (float(pd_in.max()), float(pd_in.mean()))
        m = base[base * pmax < 1.0]
        if m.size == 0:
            raise ValueError(f"no alternative PD below 1 for the predicted PD {pmax} of grade {labels[j]}")
        w = np.sqrt(m * pbar / (1.0 - m * pbar))
        w = w / w.sum()
        dj, nj = d[:, j][None, :], n[:, j][None, :]
        step = dj * np.log(m)[:, None] + (nj - dj) * np.log((1.0 - m[:, None] * pj[None, :]) / (1.0 - pj)[None, :])
        grade_log[:, j] = log_mixture(np.cumsum(step, axis=1), np.log(w)) if steps else np.empty(0)
        used_m.append(m.tolist())
        used_w.append(w.tolist())
    if combine == "product":
        total = grade_log.sum(axis=1)
    else:
        total = np.logaddexp.reduce(grade_log, axis=1) - np.log(n_grades) if n_grades else np.zeros(steps)
    return GradeEProcessResult(
        monitor_id="sequential.pd_grades", log_e=total, alpha_level=level,
        h0="the defaults of every grade are Binomial(obligors, predicted PD) given the past",
        reference=REFERENCE, inputs_hash=inputs_hash(d, n, np.ascontiguousarray(pd), level, base),
        notes=("grades combined by " + ("product (assumes independence of the grades' defaults)" if combine == "product"
                                         else "average (no independence assumption)")),
        extras={"direction": direction, "multipliers": used_m, "weights": used_w,
                "n_defaults": int(d.sum()), "n_obligor_steps": int(n.sum())},
        grade_log_e=grade_log, grades=labels, combine=combine)


__all__ = [
    "pd_grade_eprocess",
]
