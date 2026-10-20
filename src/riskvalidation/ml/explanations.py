"""Explanation checks on attributions computed in the pipeline (by the pinned ``shap``) and passed as arrays (dossier 06
C.7 l.990, l.1006-1008; I.1 l.1192-1193; dossier 04 J.2 l.881-883 and ladder P4 l.968).

Reason-code stability (the share of the top reasons kept across bootstrap refits), the agreement of attribution ranks
between SHAP and ALE or PDP, the Shapley additivity check ``f(x) = phi_0 + sum phi_i`` and the sign and monotonicity
checks against the expected economic rationale (EBA recommendation 3). Contract: ``engine-ml-api`` 1.1.0, module
``riskvalidation/ml/explanations.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.ml._checks import index, nonneg, scalar, vec


@dataclass(frozen=True)
class OverlapResult:
    """The mean and the 5th percentile, over obligors, of the share of the ``top_k`` reasons kept between each refit and
    the first, over ``n_refits`` refits (the first included). Shares are in [0, 1]. The 5th percentile is not forced
    below the mean: it is not an invariant of a share distribution (finding F1 of unit U14-E1, in ``docs/ml/02``)."""

    top_k: int
    n_refits: int
    mean: float
    p05: float

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.n_refits < 2:
            raise ValueError("top_k must be at least 1 and n_refits at least 2")
        for name in ("mean", "p05"):
            v = getattr(self, name)
            if v < 0.0 or v > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


def reason_code_overlap(attributions_by_refit: np.ndarray, top_k: int = 4) -> OverlapResult:
    """The mean and the 5th percentile over obligors of the share of the ``top_k`` reasons kept between each refit and
    the first refit (dossier 04 J.2 l.881-883; ladder P4 l.968: the bootstrap top-four overlap).

    ``attributions_by_refit`` has shape (n_refits, n, p), n_refits at least two, on the log-odds of the event (the
    default). Sign convention: a positive attribution pushes toward default. The reasons of an obligor in a refit are
    its ``top_k`` largest signed contributions toward the event, in descending order, ties by column order (dossier 04
    J.2 l.878-883: "SHAP shortfalls play the role of Delta_j", Regulation B comment 9(b)(2)-5); a large negative
    attribution is never a reason. The share of an obligor is averaged over the refits after the first, and the mean
    and the 5th percentile are then taken over obligors (not over (refit, obligor) pairs); identical refits give 1."""
    a = np.array(attributions_by_refit, dtype=float)
    if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] < 1:
        raise ValueError("attributions_by_refit must have shape (n_refits, n, p) with at least two refits")
    if not np.all(np.isfinite(a)):
        raise ValueError("attributions_by_refit contains non-finite values")
    k = index(top_k, "top_k", a.shape[2] + 1, lo=1)
    order = np.argsort(-a, axis=2, kind="stable")[:, :, :k]
    kept = np.zeros(a.shape, dtype=bool)
    np.put_along_axis(kept, order, True, axis=2)
    share = (kept[1:] & kept[0]).sum(axis=2) / k  # (n_refits - 1, n)
    per_obligor = share.mean(axis=0)
    return OverlapResult(
        top_k=k, n_refits=a.shape[0], mean=float(per_obligor.mean()), p05=float(np.percentile(per_obligor, 5.0))
    )


def rank_agreement(attribution_importance: np.ndarray, effect_range: np.ndarray) -> float:
    """The Spearman rho (``scipy.stats.spearmanr``), in [-1, 1], between two per-feature vectors: the mean absolute SHAP
    per feature against the range of its ALE or PDP curve (the explainer agreement of C20, dossier 06 C.7 l.1006-1008),
    or the mean absolute SHAP on the clean batches against the shocked ones. A constant vector has no ranks and
    raises."""
    a = vec(attribution_importance, "attribution_importance")
    b = vec(effect_range, "effect_range")
    if a.shape != b.shape:
        raise ValueError(f"effect_range must have the shape {a.shape} of attribution_importance, got {b.shape}")
    if a.size < 2:
        raise ValueError("attribution_importance needs at least two features")
    if a.max() == a.min() or b.max() == b.min():
        raise ValueError("a constant input has no ranks: rank_agreement is undefined")
    # the Pearson correlation of the average ranks (what scipy.stats.spearmanr computes), written as S_ab / sqrt(S_aa
    # S_bb) so that equal vectors give exactly 1 and reversed ones exactly -1 (spearmanr returns 0.9999999999999999)
    ra, rb = stats.rankdata(a) - (a.size + 1) / 2.0, stats.rankdata(b) - (a.size + 1) / 2.0
    rho = float(np.sum(ra * rb) / np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return min(1.0, max(-1.0, rho))


def shapley_additivity_gap(prediction: np.ndarray, base_value: float, attributions: np.ndarray) -> float:
    """``max_i |f(x_i) - (phi_0 + sum_j phi_ij)|`` (dossier 06 C.7 l.990: additivity); zero when it holds."""
    f = vec(prediction, "prediction")
    phi = np.array(attributions, dtype=float)
    if phi.ndim != 2 or phi.shape[0] != f.shape[0]:
        raise ValueError(f"attributions must have shape ({f.shape[0]}, p), got {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("attributions contains non-finite values")
    base = scalar(base_value, "base_value")
    return float(np.max(np.abs(f - (base + phi.sum(axis=1)))))


def _sign(expected_sign: int) -> int:
    if isinstance(expected_sign, bool) or expected_sign not in (-1, 1):
        raise ValueError(f"expected_sign must be -1 or +1, got {expected_sign!r}")
    return int(expected_sign)


def sign_check(effect: np.ndarray, expected_sign: int) -> bool:
    """True when the net change of the effect from the first to the last grid point has the expected sign and is nonzero
    (dossier 06 I.1 l.1192; EBA recommendation 3, A.8 l.480-482)."""
    e = vec(effect, "effect")
    if e.size < 2:
        raise ValueError("effect needs at least two grid points")
    s = _sign(expected_sign)
    net = float(e[-1] - e[0])
    return bool(net != 0.0 and np.sign(net) == s)


def monotonicity_check(effect: np.ndarray, expected_sign: int, tolerance: float) -> float:
    """The largest violation of the expected monotonicity: the largest fall of the curve below its running maximum when
    a rise is expected (above its running minimum when a fall is expected). A violation within ``tolerance`` counts as
    none (0 is returned); otherwise the violation itself is returned, so a planted bump of known height returns that
    height."""
    e = vec(effect, "effect")
    s = _sign(expected_sign)
    tol = nonneg(tolerance, "tolerance")
    violation = np.maximum.accumulate(e) - e if s == 1 else e - np.minimum.accumulate(e)
    worst = float(violation.max())
    return worst if worst > tol else 0.0
