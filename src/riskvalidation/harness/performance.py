r"""Performance of an estimator against its known truth, with Monte Carlo standard errors (Morris, White and Crowther
2019, doi:10.1002/sim.8086, Table 6).

- bias $\bar\theta - \theta$, Monte Carlo SE $\sqrt{\sum(\hat\theta_i - \bar\theta)^2/(n(n-1))}$;
- empirical SE $\sqrt{\sum(\hat\theta_i-\bar\theta)^2/(n-1)}$, Monte Carlo SE $\mathrm{EmpSE}/\sqrt{2(n-1)}$;
- average model SE $\sqrt{\overline{\widehat{\mathrm{Var}}}}$ and its relative error against the empirical SE,
  Monte Carlo SE
  $100\,(\mathrm{ModSE}/\mathrm{EmpSE})\sqrt{\mathrm{Var}[\widehat{\mathrm{Var}}]/(4n\,\mathrm{ModSE}^4) + 1/(2(n-1))}$;
- coverage of an interval, Monte Carlo SE $\sqrt{c(1-c)/n}$.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def estimator_performance(estimates: Any, truth: float, *, model_se: Any = None, lower: Any = None,
                          upper: Any = None) -> dict[str, Any]:
    """Bias, empirical SE, model SE and interval coverage of ``estimates`` of ``truth``, each with its Monte Carlo
    standard error."""
    est = np.asarray(estimates, dtype=float)
    est = est[np.isfinite(est)]
    n = est.size
    if n < 2:
        raise ValueError("at least two estimates are required")
    mean = float(est.mean())
    emp_se = float(est.std(ddof=1))
    out: dict[str, Any] = {
        "n": n,
        "truth": float(truth),
        "mean": mean,
        "bias": mean - float(truth),
        "bias_mcse": emp_se / math.sqrt(n),
        "empirical_se": emp_se,
        "empirical_se_mcse": emp_se / math.sqrt(2.0 * (n - 1)),
    }
    if model_se is not None:
        var = np.asarray(model_se, dtype=float) ** 2
        var = var[np.isfinite(var)]
        mod_se = float(math.sqrt(var.mean()))
        out["model_se"] = mod_se
        ratio = mod_se / emp_se
        out["model_se_relative_error_pct"] = 100.0 * (ratio - 1.0)
        var_var = float(np.var(var, ddof=1))
        out["model_se_relative_error_mcse_pct"] = 100.0 * ratio * math.sqrt(
            var_var / (4.0 * var.size * mod_se**4) + 1.0 / (2.0 * (n - 1)))
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        hit = (lo <= truth) & (truth <= hi)
        c = float(hit.mean())
        out["coverage"] = c
        out["coverage_mcse"] = math.sqrt(c * (1.0 - c) / hit.size)
    return out


__all__ = ["estimator_performance"]
