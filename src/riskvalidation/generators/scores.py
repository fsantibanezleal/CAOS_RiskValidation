r"""Scored samples with a known truth: binormal scores (an exact AUC) and a logistic portfolio (an exact PD per
obligor).

Binormal: defaulters score $N(\delta, 1)$, survivors $N(0, 1)$, higher is riskier, so
$\mathrm{AUC} = P(X_D > X_N) = \Phi(\delta/\sqrt 2)$ (the difference of two normals). The variance of the Mann-Whitney
AUC on $n_1$ defaulters and $n_0$ survivors is exact too (Hanley and McNeil 1982, doi:10.1148/radiology.143.1.7063747):
$[A(1-A) + (n_1-1)(Q_1-A^2) + (n_0-1)(Q_2-A^2)]/(n_1 n_0)$, where for equal class variances
$Q_1 = Q_2 = \Phi_2(a, a; 1/2)$ with $A = \Phi(a)$, because $X - Y$ and $X' - Y$ share $Y$.

Logistic: covariates $x \sim N(\mu, \Sigma)$, true $\mathrm{logit}\,\pi = \beta_0 + \beta' x$ (optionally plus an
omitted quadratic term), so the linear predictor is normal and the portfolio's truths are one-dimensional integrals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from scipy import special, stats

from riskvalidation.generators._base import (
    GH_NODES,
    GH_WEIGHTS,
    Check,
    GeneratorBase,
    bvn_equal,
    frozen_array,
    mean_check,
    variance_check,
)
from riskvalidation.validation.discrimination import auc_with_variance

HANLEY_MCNEIL = "Hanley and McNeil (1982), Radiology 143(1), doi:10.1148/radiology.143.1.7063747"
DECILES = special.ndtri(np.arange(1, 10) / 10.0)


def binormal_auc(separation: float, variance: float = 1.0) -> float:
    """AUC of scores N(separation, variance) against N(0, variance): Phi(separation / sqrt(2 variance))."""
    return float(special.ndtr(separation / math.sqrt(2.0 * variance)))


def binormal_auc_variance(separation: float, n_defaulters: int, n_survivors: int, variance: float = 1.0) -> float:
    """Exact variance of the Mann-Whitney AUC for binormal scores with equal class variances."""
    a = separation / math.sqrt(2.0 * variance)
    auc = float(special.ndtr(a))
    q = bvn_equal(a, 0.5)
    return (auc * (1.0 - auc) + (n_defaulters - 1) * (q - auc * auc) + (n_survivors - 1) * (q - auc * auc)) / (
        n_defaulters * n_survivors)


def _decile_counts(x: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(DECILES, x, side="right")
    return np.bincount(idx, minlength=10)


@dataclass(frozen=True)
class BinormalScores(GeneratorBase):
    """Current scores of ``n_defaulters`` and ``n_survivors``; optionally a challenger, a development sample, a leak.

    - ``separation``: the defaulters' mean score now; ``separation_development`` at development (decay when smaller).
    - ``development_size``: ``None`` gives the test the true development AUC (the ECB treats it as a known number);
      a positive ``m`` estimates it on a development sample ``m`` times the current one.
    - ``challenger_separation`` and ``challenger_correlation``: a second, non-nested score for DeLong.
    - ``leak``: ``None`` for no extra feature; otherwise the model score is the genuine score plus a feature that in
      development is shifted by ``leak`` for defaulters (it carries the outcome) and at decision time is noise. A leak
      of 0 is the leakage family's null.
    """

    name: ClassVar[str] = "binormal_scores"
    n_defaulters: int
    n_survivors: int
    separation: float = 1.0
    separation_development: float | None = None
    development_size: float | None = None
    challenger_separation: float | None = None
    challenger_correlation: float = 0.5
    leak: float | None = None

    def __post_init__(self) -> None:
        if int(self.n_defaulters) < 2 or int(self.n_survivors) < 2:
            raise ValueError("at least two defaulters and two survivors are required")
        if self.development_size is not None and float(self.development_size) <= 0.0:
            raise ValueError("development_size must be positive")
        if not -1.0 < float(self.challenger_correlation) < 1.0:
            raise ValueError("challenger_correlation must lie in (-1, 1)")
        object.__setattr__(self, "n_defaulters", int(self.n_defaulters))
        object.__setattr__(self, "n_survivors", int(self.n_survivors))

    @property
    def _delta_dev(self) -> float:
        return float(self.separation if self.separation_development is None else self.separation_development)

    @property
    def _score_variance(self) -> float:
        return 2.0 if self.leak is not None else 1.0

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """Current genuine score(s), then the feature at decision time, then the development sample."""
        n1, n0 = self.n_defaulters, self.n_survivors
        y = np.concatenate([np.ones(n1, dtype=np.int64), np.zeros(n0, dtype=np.int64)])
        out: dict[str, Any] = {"defaults": y}
        if self.challenger_separation is None:
            score = np.concatenate([self.separation + rng.standard_normal(n1), rng.standard_normal(n0)])
        else:
            c = float(self.challenger_correlation)
            chol = np.array([[1.0, 0.0], [c, math.sqrt(1.0 - c * c)]])
            zd = rng.standard_normal((n1, 2)) @ chol.T + np.array([self.separation, self.challenger_separation])
            zs = rng.standard_normal((n0, 2)) @ chol.T
            score = np.concatenate([zd[:, 0], zs[:, 0]])
            out["risk_2"] = np.concatenate([zd[:, 1], zs[:, 1]])
        risk = score
        if self.leak is not None:
            feature_now = rng.standard_normal(n1 + n0)
            risk = score + feature_now
            out["feature_current"] = feature_now
        out["risk"] = risk
        out["risk_1"] = risk
        if self.development_size is not None or self.leak is not None:
            m = 1.0 if self.development_size is None else float(self.development_size)
            m1, m0 = max(2, round(n1 * m)), max(2, round(n0 * m))
            dev = np.concatenate([self._delta_dev + rng.standard_normal(m1), rng.standard_normal(m0)])
            dy = np.concatenate([np.ones(m1, dtype=np.int64), np.zeros(m0, dtype=np.int64)])
            if self.leak is not None:
                feature_dev = np.concatenate([self.leak + rng.standard_normal(m1), rng.standard_normal(m0)])
                dev = dev + feature_dev
                out["feature_development"] = feature_dev
                out["expected_counts"] = _decile_counts(feature_dev)
                out["actual_counts"] = _decile_counts(feature_now)
            out["development_risk"] = dev
            out["development_defaults"] = dy
        if self.development_size is None:
            out["auc_initial"] = self.truth_auc_development
        else:
            out["auc_initial"] = auc_with_variance(out["development_risk"], out["development_defaults"])[0]
        return out

    @property
    def truth_auc_development(self) -> float:
        return binormal_auc(self._delta_dev + (self.leak or 0.0), self._score_variance)

    def truth(self) -> dict[str, Any]:
        v = self._score_variance
        lam = self.leak or 0.0
        auc_now = binormal_auc(self.separation, v)
        auc_dev = self.truth_auc_development
        out: dict[str, Any] = {
            "auc_current": auc_now,
            "auc_development": auc_dev,
            "auc_genuine_current": binormal_auc(self.separation),
            "auc_variance_current": binormal_auc_variance(self.separation, self.n_defaulters, self.n_survivors, v),
            "auc_initial_known": self.development_size is None,
            "discrimination_null_holds": auc_dev <= auc_now + 1e-15,
            "reference": HANLEY_MCNEIL,
        }
        if self.challenger_separation is not None:
            out["auc_challenger"] = binormal_auc(self.challenger_separation)
            out["delong_null_holds"] = math.isclose(self.separation, self.challenger_separation) and self.leak is None
        if self.leak is not None:
            share = self.n_defaulters / (self.n_defaulters + self.n_survivors)
            edges = np.concatenate([[-np.inf], DECILES, [np.inf]])
            p_now = np.diff(special.ndtr(edges))
            p_dev = (1.0 - share) * p_now + share * np.diff(special.ndtr(edges - lam))
            out["feature_bins_development"] = p_dev.tolist()
            out["feature_bins_current"] = p_now.tolist()
            out["feature_population_psi"] = float(np.sum((p_now - p_dev) * np.log(p_now / p_dev)))
        return out

    def verify(self, n_draws: int = 2_000, seed: int = 20261005) -> list[Check]:
        """The empirical AUC's mean and variance against the exact ones; the challenger's and the development AUC."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
        tr = self.truth()
        aucs, aucs2, inits = [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            aucs.append(auc_with_variance(d["risk"], d["defaults"])[0])
            if "risk_2" in d:
                aucs2.append(auc_with_variance(d["risk_2"], d["defaults"])[0])
            inits.append(d["auc_initial"])
        out = [mean_check("mean AUC, current", tr["auc_current"], np.array(aucs)),
               variance_check("variance of the AUC, current", tr["auc_variance_current"], np.array(aucs))]
        if aucs2:
            out.append(mean_check("mean AUC, challenger", tr["auc_challenger"], np.array(aucs2)))
        if self.development_size is not None:
            out.append(mean_check("mean development AUC", tr["auc_development"], np.array(inits)))
        return out


@dataclass(frozen=True)
class LogisticPortfolio(GeneratorBase):
    """``n`` obligors with equicorrelated standard normal covariates shifted by ``shift``; true PD logistic in them.

    The model under test gives PD = expit(``calibration_intercept`` + ``calibration_slope`` times the true linear
    predictor): intercept 0 and slope 1 is a right model, a negative intercept underestimates every PD. ``quadratic``
    adds ``quadratic`` (x_1^2 - 1) to the true logit, a term the model omits (the misspecification of Hosmer et al.
    1997). ``rho`` makes the defaults of the sample depend on one systematic factor (the true PD stays the
    unconditional one).
    """

    name: ClassVar[str] = "logistic_portfolio"
    n: int = 5000
    intercept: float = -3.5
    coefficients: Any = (0.8, 0.5, 0.3)
    correlation: float = 0.3
    shift: Any = 0.0
    quadratic: float = 0.0
    calibration_intercept: float = 0.0
    calibration_slope: float = 1.0
    rho: float = 0.0

    def __post_init__(self) -> None:
        beta = np.asarray(self.coefficients, dtype=float).reshape(-1)
        p = beta.size
        if p < 1:
            raise ValueError("at least one coefficient is required")
        if int(self.n) < 2:
            raise ValueError("at least two obligors are required")
        lower = -1.0 / (p - 1) if p > 1 else -1.0
        if not lower < float(self.correlation) < 1.0:
            raise ValueError("the equicorrelation must keep the covariance positive definite")
        shift = np.broadcast_to(np.asarray(self.shift, dtype=float).reshape(-1), (p,))
        if not 0.0 <= float(self.rho) < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "coefficients", frozen_array(beta))
        object.__setattr__(self, "shift", frozen_array(shift))

    @cached_property
    def _beta(self) -> np.ndarray:
        return np.asarray(self.coefficients, dtype=float)

    @cached_property
    def _sigma(self) -> np.ndarray:
        p = self._beta.size
        return np.full((p, p), float(self.correlation)) + (1.0 - float(self.correlation)) * np.eye(p)

    @cached_property
    def _chol(self) -> np.ndarray:
        return np.linalg.cholesky(self._sigma)

    @cached_property
    def _mu(self) -> np.ndarray:
        return np.asarray(self.shift, dtype=float)

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """Covariates, then the systematic factor (when ``rho`` > 0), then one uniform per obligor for its default."""
        x = rng.standard_normal((self.n, self._beta.size)) @ self._chol.T + self._mu
        eta = self.intercept + x @ self._beta
        logit = eta + self.quadratic * (x[:, 0] ** 2 - 1.0) if self.quadratic else eta
        pd_true = special.expit(logit)
        out: dict[str, Any] = {}
        if self.rho > 0.0:
            factor = float(rng.standard_normal())
            cond = special.ndtr((special.ndtri(pd_true) - math.sqrt(self.rho) * factor) / math.sqrt(1.0 - self.rho))
            out["factor"] = factor
        else:
            cond = pd_true
        y = (rng.random(self.n) < cond).astype(np.int64)
        model_logit = self.calibration_intercept + self.calibration_slope * eta
        out.update({"defaults": y, "pd": special.expit(model_logit), "pd_true": pd_true, "risk": model_logit, "x": x})
        return out

    def _eta_moments(self) -> tuple[float, float]:
        mean = float(self.intercept + self._beta @ self._mu)
        sd = float(math.sqrt(self._beta @ self._sigma @ self._beta))
        return mean, sd

    def truth(self) -> dict[str, Any]:
        """Exact by Gauss-Hermite quadrature over the linear predictor (and over x_1 when a quadratic term is set)."""
        m, s = self._eta_moments()
        a, b = float(self.calibration_intercept), float(self.calibration_slope)
        if self.quadratic == 0.0:
            eta = m + s * GH_NODES
            pt = special.expit(eta)
            pm = special.expit(a + b * eta)
            w = GH_WEIGHTS
            mean_pt = float(w @ pt)
            auc = self._auc_of_eta(m, s) if b > 0 else None
        else:
            # x_1 ~ N(mu_1, 1); eta | x_1 normal with slope c = (Sigma beta)_1 and residual variance s^2 - c^2
            c = float((self._sigma @ self._beta)[0])
            r = math.sqrt(max(s * s - c * c, 0.0))
            n1, w1 = np.polynomial.hermite_e.hermegauss(96)
            w1 = w1 / math.sqrt(2.0 * math.pi)
            x1 = self._mu[0] + n1[:, None]
            eta = m + c * n1[:, None] + r * n1[None, :]
            pt = special.expit(eta + self.quadratic * (x1**2 - 1.0))
            pm = special.expit(a + b * eta)
            w = np.outer(w1, w1)
            mean_pt = float(np.sum(w * pt))
            auc = None
        mean_pm = float(np.sum(w * pm))
        brier_true = float(np.sum(w * pt * (1.0 - pt)))
        brier_model = float(np.sum(w * (pm * pm - 2.0 * pm * pt + pt)))
        null = a == 0.0 and b == 1.0 and self.quadratic == 0.0
        return {
            "mean_pd_true": mean_pt,
            "mean_pd_model": mean_pm,
            "expected_brier_true_pd": brier_true,
            "expected_brier_model": brier_model,
            "linear_predictor_mean": m,
            "linear_predictor_sd": s,
            "auc_model": auc,
            "calibration_null_holds": null,
        }

    @staticmethod
    def _auc_of_eta(m: float, s: float) -> float:
        """P(eta_D > eta_N) for eta ~ N(m, s^2) with P(D | eta) = expit(eta): the defaulters' and the survivors'
        densities are the tilted normals expit(eta) phi and (1 - expit(eta)) phi; Simpson's rule on a fine grid."""
        u = np.linspace(m - 12.0 * s, m + 12.0 * s, 40_001)
        h = u[1] - u[0]
        phi = stats.norm.pdf(u, m, s)
        f1 = special.expit(u) * phi
        f0 = (1.0 - special.expit(u)) * phi
        simpson = np.ones_like(u)
        simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
        simpson *= h / 3.0
        p1 = float(simpson @ f1)
        p0 = float(simpson @ f0)
        # F0(u): cumulative trapezoid of the survivors' density, then the defaulters' expectation of it
        f0_cum = np.concatenate([[0.0], np.cumsum((f0[1:] + f0[:-1]) * h / 2.0)])
        return float(simpson @ (f1 * f0_cum)) / (p1 * p0)

    def verify(self, n_draws: int = 1_000, seed: int = 20261005) -> list[Check]:
        """Default rate, mean true and model PDs and the Brier score of the model against their exact values."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 304]))
        tr = self.truth()
        dr, mpt, mpm, brier = [], [], [], []
        for _ in range(n_draws):
            d = self.draw(rng)
            dr.append(d["defaults"].mean())
            mpt.append(d["pd_true"].mean())
            mpm.append(d["pd"].mean())
            brier.append(np.mean((d["pd"] - d["defaults"]) ** 2))
        return [mean_check("default rate", tr["mean_pd_true"], np.array(dr)),
                mean_check("mean true PD", tr["mean_pd_true"], np.array(mpt)),
                mean_check("mean model PD", tr["mean_pd_model"], np.array(mpm)),
                mean_check("Brier score of the model", tr["expected_brier_model"], np.array(brier))]


__all__ = ["BinormalScores", "HANLEY_MCNEIL", "LogisticPortfolio", "binormal_auc", "binormal_auc_variance"]
