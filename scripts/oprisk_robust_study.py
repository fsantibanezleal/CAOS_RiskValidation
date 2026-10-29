"""The S-3 study: the capital bias and spread of the robust severity fits beside the maximum likelihood fits.

Generators of known truth (left-truncated at the collection threshold ``H`` = EUR 20,000, ``n`` observed losses, an
observed annual rate of 40 losses above ``H``, capital = the mean-adjusted single loss approximation at 0.999):

- ``lognormal``: lognormal(10, 1.6), the family the lognormal estimators assume (flattering to them);
- ``lognormal-contaminated``: the same with 3 % of the losses multiplied by 50 (gross errors);
- ``splice``: lognormal(10, 1.2) body with a GPD tail (xi 0.6, beta 60,000) above EUR 100,000 (misspecified for them);
- ``splice-contaminated``: the same with 3 % of the losses multiplied by 50.

Five estimators per history: the truncated lognormal maximum likelihood (``mle-lognormal``), the Cramer-von Mises fit
(``cvm``), the optimally bias-robust fit at 95 % efficiency (``obre``) and at 80 % (``obre-80``, declared before the
run) and the maximum likelihood lognormal and GPD splice (``mle-splice``, splice threshold at the 90th percentile of
the history). The truth is the same functional of the true law (the mean-adjusted SLA with the true rescaled
frequency), so the ratio isolates the estimation. Deterministic under ``--seed``; the history ``r`` of a scenario uses
the seed ``seed + r``.

    python scripts/oprisk_robust_study.py --scenario lognormal-contaminated --reps 1000 --out result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from riskvalidation.oprisk import distributions, robust

H = 20_000.0
LEVEL = 0.999
RATE_OBSERVED = 40.0
CONTAMINATION_SHARE = 0.03
CONTAMINATION_FACTOR = 50.0
_SPLICE_U = 100_000.0
_SPLICE_MU, _SPLICE_SIGMA = 10.0, 1.2
_SPLICE = (_SPLICE_MU, _SPLICE_SIGMA, _SPLICE_U, 0.6, 60_000.0,
           float(ndtr((math.log(_SPLICE_U) - _SPLICE_MU) / _SPLICE_SIGMA)))
TRUTH = {
    "lognormal": ("lognormal", (10.0, 1.6)),
    "lognormal-contaminated": ("lognormal", (10.0, 1.6)),
    "splice": ("lognormal-gpd-splice", _SPLICE),
    "splice-contaminated": ("lognormal-gpd-splice", _SPLICE),
}
SCENARIOS = tuple(TRUTH)
METHODS = ("mle-lognormal", "cvm", "obre", "obre-80", "mle-splice")


def true_capital(scenario: str) -> float:
    law, params = TRUTH[scenario]
    lam = distributions.rescale_frequency(RATE_OBSERVED, law, params, H)
    return robust.sla_var_mean_adjusted(lam, law, params, LEVEL).var


def draw_history(scenario: str, n: int, seed: int) -> np.ndarray:
    law, params = TRUTH[scenario]
    rng = np.random.default_rng(seed)
    f_h = float(distributions.severity_cdf(law, params, np.array([H]))[0])
    x = distributions.severity_ppf(law, params, f_h + rng.random(n) * (1.0 - f_h))
    if scenario.endswith("contaminated"):
        idx = rng.choice(n, size=int(round(CONTAMINATION_SHARE * n)), replace=False)
        x[idx] *= CONTAMINATION_FACTOR
    return x


def _capital(law: str, params) -> float:
    lam = distributions.rescale_frequency(RATE_OBSERVED, law, params, H)
    return robust.sla_var_mean_adjusted(lam, law, params, LEVEL).var


def capitals(x: np.ndarray) -> dict[str, float]:
    """The capital of each method for one history; ``nan`` when the method has none (a failed fit or an infinite
    mean)."""
    out = dict.fromkeys(METHODS, math.nan)
    fits = {
        "mle-lognormal": lambda: distributions.fit_severity("lognormal", x, H, truncated=True).params,
        "cvm": lambda: robust.fit_severity_cvm("lognormal", x, H).params,
        "obre": lambda: robust.fit_severity_obre("lognormal", x, H, efficiency=0.95).params,
        "obre-80": lambda: robust.fit_severity_obre("lognormal", x, H, efficiency=0.80).params,
    }
    for name, fit in fits.items():
        try:
            out[name] = _capital("lognormal", fit())
        except ValueError:
            pass
    try:
        out["mle-splice"] = _capital("lognormal-gpd-splice",
                                     distributions.fit_splice(x, H, float(np.quantile(x, 0.9))).params)
    except ValueError:
        pass
    return out


def _summary(ratios: np.ndarray) -> dict[str, float]:
    ok = ratios[np.isfinite(ratios)]
    if ok.size == 0:
        return {"n_defined": 0}
    q1, med, q3 = (float(v) for v in np.quantile(ok, [0.25, 0.5, 0.75]))
    return {"n_defined": int(ok.size), "median_ratio": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
            "median_abs_log_ratio": float(np.median(np.abs(np.log(ok)))),
            "share_underestimating": float(np.mean(ok < 1.0)), "share_below_half": float(np.mean(ok < 0.5)),
            "share_above_double": float(np.mean(ok > 2.0))}


def run(scenario: str, reps: int, seed: int, n: int = 250, max_seconds: float | None = None,
        checkpoint: Path | None = None) -> dict:
    """The ratio of each method's capital to the true capital over up to ``reps`` histories of ``n`` losses.

    With ``max_seconds`` the loop stops after the history that crosses the deadline and ``reps_done`` states how many
    histories the summary covers; with ``checkpoint`` the summary so far is written every 10 histories."""
    if scenario not in TRUTH:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    truth = true_capital(scenario)
    ratios = {m: np.full(reps, math.nan) for m in METHODS}
    start = time.monotonic()
    done = 0

    def result() -> dict:
        return {"scenario": scenario, "reps": reps, "reps_done": done, "n_losses": n, "seed": seed, "level": LEVEL,
                "threshold": H, "true_capital": truth,
                "methods": {m: _summary(ratios[m][:done]) for m in METHODS}}

    for r in range(reps):
        cap = capitals(draw_history(scenario, n, seed + r))
        for m in METHODS:
            ratios[m][r] = cap[m] / truth
        done = r + 1
        if checkpoint is not None and done % 10 == 0:
            checkpoint.write_text(json.dumps(result(), indent=2), encoding="utf-8")
        if max_seconds is not None and time.monotonic() - start > max_seconds:
            break
    return result()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a.scenario, a.reps, a.seed, a.n, a.max_seconds, a.out)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
