"""Nested discriminant models fitted on the sample they are compared on: Demler, Pencina and D'Agostino's design.

"We used numerical simulations on multivariate normal data with equal covariance matrices in the event and nonevent
subgroups ... the means and correlation structure of an actual data set from the Framingham Heart Study ... A total of
8261 observations ... cases (621 observations) ... The effect sizes ... were 0.65, -0.47, 0.48, 0.62, and 0.42 for
age, TCL, HDL, SBP, and DBP ... the data for the added fifth predictor (DBP) was simulated under the null ... by making
its means equal in the two subgroups and ... setting nondiagonal elements in the last column of the correlation matrix
to zero. We repeated the simulations 1000 times. Linear discriminant analysis was fitted to each simulated dataset,
first for the reduced model and then for the full model" (Demler et al. 2012, doi:10.1002/sim.5328, section 2.3).
They report a DeLong rejection rate of 0.001 at the 5% level where the exact F-test gives about 5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from scipy import special

from riskvalidation.generators._base import Check, GeneratorBase, frozen_array, mean_check
from riskvalidation.validation.discrimination import auc_with_variance

DEMLER = (
    "Demler, Pencina and D'Agostino (2012), Misuse of DeLong test to compare AUCs for nested models, Statistics in "
    "Medicine 31(23), doi:10.1002/sim.5328, section 2.3"
)
# age, total cholesterol, HDL cholesterol, systolic blood pressure: effect sizes and correlations as printed
FRAMINGHAM_EFFECTS = (0.65, -0.47, 0.48, 0.62)
FRAMINGHAM_CORRELATION = (
    (1.0, 0.39, 0.06, 0.44),
    (0.39, 1.0, 0.08, 0.25),
    (0.06, 0.08, 1.0, -0.04),
    (0.44, 0.25, -0.04, 1.0),
)


def lda_auc(effects: Any, correlation: Any) -> float:
    """Population AUC of the linear discriminant, Phi(sqrt(M^2 / 2)) with M^2 the squared Mahalanobis distance
    (Demler et al. 2012, equation 3)."""
    d = np.asarray(effects, dtype=float)
    m2 = float(d @ np.linalg.solve(np.asarray(correlation, dtype=float), d))
    return float(special.ndtr(math.sqrt(m2 / 2.0)))


@dataclass(frozen=True)
class NestedLda(GeneratorBase):
    """Cases and non-cases with a common correlation; a reduced model on the first predictors and a full model with
    one more, uncorrelated with the others, whose effect is ``added_effect`` (0: the null of equal AUCs). Both models
    are linear discriminants fitted on the drawn sample, and their scores are returned on the same sample."""

    name: ClassVar[str] = "nested_lda"
    n_cases: int = 621
    n_noncases: int = 7640
    effects: Any = FRAMINGHAM_EFFECTS
    correlation: Any = FRAMINGHAM_CORRELATION
    added_effect: float = 0.0

    def __post_init__(self) -> None:
        d = np.asarray(self.effects, dtype=float)
        c = np.asarray(self.correlation, dtype=float)
        if c.shape != (d.size, d.size) or not np.allclose(c, c.T) or np.any(np.linalg.eigvalsh(c) <= 0):
            raise ValueError("correlation must be a positive definite matrix matching the effects")
        if int(self.n_cases) < 3 or int(self.n_noncases) < 3:
            raise ValueError("at least three cases and three non-cases are required")
        object.__setattr__(self, "effects", frozen_array(d))
        object.__setattr__(self, "correlation", frozen_array(c))
        object.__setattr__(self, "n_cases", int(self.n_cases))
        object.__setattr__(self, "n_noncases", int(self.n_noncases))

    @cached_property
    def _full_corr(self) -> np.ndarray:
        c = np.asarray(self.correlation, dtype=float)
        k = c.shape[0]
        out = np.eye(k + 1)
        out[:k, :k] = c
        return out

    @cached_property
    def _chol(self) -> np.ndarray:
        return np.linalg.cholesky(self._full_corr)

    @cached_property
    def _means(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.effects, dtype=float), [float(self.added_effect)]])

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """Cases, then non-cases; the two discriminants fitted on the sample (pooled covariance) and applied to it."""
        n1, n0 = self.n_cases, self.n_noncases
        k = self._means.size
        x1 = rng.standard_normal((n1, k)) @ self._chol.T + self._means
        x0 = rng.standard_normal((n0, k)) @ self._chol.T
        diff = x1.mean(axis=0) - x0.mean(axis=0)
        pooled = (np.cov(x1, rowvar=False) * (n1 - 1) + np.cov(x0, rowvar=False) * (n0 - 1)) / (n1 + n0 - 2)
        w_full = np.linalg.solve(pooled, diff)
        w_red = np.linalg.solve(pooled[: k - 1, : k - 1], diff[: k - 1])
        x = np.vstack([x1, x0])
        y = np.concatenate([np.ones(n1, dtype=np.int64), np.zeros(n0, dtype=np.int64)])
        return {"risk_1": x @ w_full, "risk_2": x[:, : k - 1] @ w_red, "defaults": y}

    def truth(self) -> dict[str, Any]:
        k = self._means.size
        auc_full = lda_auc(self._means, self._full_corr)
        auc_red = lda_auc(self._means[: k - 1], self._full_corr[: k - 1, : k - 1])
        return {"auc_full": auc_full, "auc_reduced": auc_red, "null_holds": float(self.added_effect) == 0.0,
                "reference": DEMLER}

    def verify(self, n_draws: int = 200, seed: int = 20261005) -> list[Check]:
        """The mean in-sample AUC of each model against its population AUC (an in-sample AUC is biased upwards by
        the fit; the checks use a tolerance of four standard errors, so a large ``n_draws`` exposes the bias)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 308]))
        tr = self.truth()
        full, red = [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            full.append(auc_with_variance(d["risk_1"], d["defaults"])[0])
            red.append(auc_with_variance(d["risk_2"], d["defaults"])[0])
        return [mean_check("mean AUC, full model", tr["auc_full"], np.array(full)),
                mean_check("mean AUC, reduced model", tr["auc_reduced"], np.array(red))]


__all__ = ["DEMLER", "FRAMINGHAM_CORRELATION", "FRAMINGHAM_EFFECTS", "NestedLda", "lda_auc"]
