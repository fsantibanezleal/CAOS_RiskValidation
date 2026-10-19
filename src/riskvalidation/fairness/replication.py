"""Year-to-year replication of an estimate (decision Q4; dossier 08 l.839 asks whether the finding is stable year to
year): the estimate fitted on year t and the same estimate refitted on year t+1, compared by sign and by size. No
verdict beyond sign and size is returned (contract `engine-fairness-api` 1.0.0)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from riskvalidation.fairness.conditional import Effect


@dataclass(frozen=True)
class Replication:
    """`same_sign` and the size ratio confirm over fit (signed; None when the fit coefficient is 0)."""

    same_sign: bool
    size_ratio: float | None
    fit_coefficient: float
    confirm_coefficient: float


def _sign(v: float) -> int:
    return (v > 0.0) - (v < 0.0)


def replicate_effect(fit: Effect, confirm: Effect) -> Replication:
    """same_sign is the equality of the signs of the two coefficients (0 has its own sign); size_ratio is
    confirm.coefficient / fit.coefficient, so a flip gives a negative ratio, and None when the fit coefficient is 0."""
    for name, eff in (("fit", fit), ("confirm", confirm)):
        if not isinstance(eff, Effect) or not math.isfinite(eff.coefficient):
            raise ValueError(f"{name} must be an Effect with a finite coefficient")
    ratio = None if fit.coefficient == 0.0 else confirm.coefficient / fit.coefficient
    return Replication(_sign(fit.coefficient) == _sign(confirm.coefficient), ratio, fit.coefficient,
                       confirm.coefficient)


__all__ = ["Replication", "replicate_effect"]
