"""BS-16 evaluation: adaptive conformal recalibration of VaR and ES against the unwrapped model in stress windows.

Two sources of windows:

* `--source generator` (default): three stress scenarios of known truth (a persistent crisis, a crash with a decay, a
  slow volatility drift), each `--reps` replications of a calm history of 1,000 days followed by the window; the true
  process is a Student-t(5) GARCH(1,1) times a deterministic volatility multiplier, so the truth of every hit is known.
  HS-250 is the full rolling model; GARCH-t and FHS use parameters fitted once on the calm history and a frozen filter
  (a refit per day on every replication is not affordable), which is stated in the output.
* `--source npz PATH`: real windows. The file holds, per window id `w`, `w__returns` (the return series up to the window
  end, decimals) and `w__start` (the index of the first forecast day). GARCH-t and FHS are refitted every
  `--refit-every` forecast days on the last 1,000 returns. The French equity series is derived-only, so the file
  lives in the device data root and only this script's outputs are committed.

The statistics follow the plan: Kupiec and conditional coverage at 5 percent, the independence test, mean FZ0 change and
Z2, wrapped against unwrapped, at the levels 0.01 and 0.025. Deterministic under `--seed`. Nothing is tuned:
`--gamma` is fixed in advance (0.005, the Gibbs-Candes default) and the output reports whatever it gives.

    python scripts/e16_aci_evaluation.py --reps 200 --out results.json
    python scripts/e16_aci_evaluation.py --source npz windows.npz --out real.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from riskvalidation.market import conformal as cf
from riskvalidation.market.garch import fit_garch, innovation_var_es
from riskvalidation.market.historical import filtered_hs_var_es

LEVELS = (0.01, 0.025)
HISTORY = 1000
SIGNIFICANCE = 0.05
SCENARIOS = {"crisis-persistent": 500, "crash-decay": 100, "drift-up": 250}


# ----------------------------------------------------------------------------------------------- the generator
def _multiplier(name: str, n: int) -> np.ndarray:
    k = np.arange(n, dtype=float)
    if name == "crisis-persistent":      # 2007-09 like: a ramp to three times, a plateau, a slow decay
        return np.interp(k, [0, 250, 350, n - 1], [1.0, 3.0, 3.0, 1.5])
    if name == "crash-decay":            # 2020 like: a jump to four times after 20 days, half-life 30 days
        m = np.ones(n)
        after = k >= 20
        m[after] = 1.0 + 3.0 * 0.5 ** ((k[after] - 20) / 30.0)
        return m
    if name == "drift-up":               # 2022 like: a smooth rise to 1.8 times
        return np.interp(k, [0, n - 1], [1.0, 1.8])
    raise ValueError(f"unknown scenario {name!r}")


def simulate_window(name: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Calm GARCH(1,1) history of `HISTORY` days then the window; returns `(series, start)`."""
    length = SCENARIOS[name]
    n = HISTORY + length
    omega, a, b, nu = 2e-6, 0.08, 0.90, 5.0
    z = rng.standard_t(nu, n + 300) * math.sqrt((nu - 2.0) / nu)
    mult = np.concatenate([np.ones(300 + HISTORY), _multiplier(name, length)])
    s2 = omega / (1.0 - a - b)
    out = np.empty(n + 300)
    for t in range(n + 300):
        e = math.sqrt(s2) * z[t]
        out[t] = mult[t] * e
        s2 = omega + a * e * e + b * s2
    return out[300:], HISTORY


# ----------------------------------------------------------------------------------------------- the models
class GarchBank:
    """GARCH(1,1)-t fits and filters, shared by the GARCH-t and the FHS forecasters (both read the same fit).

    `refit_every = None`: one fit on `r[:start]`, frozen. Otherwise a refit on the last `HISTORY` returns every
    `refit_every` forecast days. The filter is the plain recursion from the sample variance of the window.
    """

    def __init__(self, r: np.ndarray, start: int, refit_every: int | None) -> None:
        self.r, self.start, self.every = r, start, refit_every
        self.fit = None
        self.cache: dict[int, tuple[float, float, np.ndarray, float]] = {}

    def _fit_for(self, t: int):
        if self.every is None:
            if self.fit is None:
                self.fit = fit_garch(self.r[max(0, self.start - HISTORY):self.start], "garch", "student-t", "constant")
            return self.fit
        if self.fit is None or (t - self.start) % self.every == 0:
            fit = fit_garch(self.r[max(0, t - HISTORY):t], "garch", "student-t", "constant")
            if fit.converged or self.fit is None:
                self.fit = fit
        return self.fit

    def state(self, t: int) -> tuple[float, float, np.ndarray, float]:
        if t in self.cache:
            return self.cache[t]
        fit = self._fit_for(t)
        p = fit.params
        w = self.r[max(0, t - HISTORY):t]
        res = w - p["mu"]
        s2 = np.empty(w.size + 1)
        s2[0] = float(np.var(w))
        for i in range(w.size):
            s2[i + 1] = p["omega"] + p["alpha[1]"] * res[i] ** 2 + p["beta[1]"] * s2[i]
        sigma = np.sqrt(s2[:-1])
        out = (float(p["mu"]), float(math.sqrt(s2[-1])), res / sigma, float(p["nu"]))
        self.cache[t] = out
        return out


def make_forecasters(r: np.ndarray, start: int, refit_every: int | None) -> dict[str, cf.Forecaster]:
    bank = GarchBank(r, start, refit_every)

    def garch_t(t: int, level: float) -> tuple[float, float]:
        mu, sigma, _, _ = bank.state(t)
        q, e = innovation_var_es(bank._fit_for(t), level)
        return mu + sigma * q, mu + sigma * e

    def fhs(t: int, level: float) -> tuple[float, float]:
        mu, sigma, z, _ = bank.state(t)
        return filtered_hs_var_es(z, mu, sigma, level)

    return {"hs-250": cf.historical_forecaster(r, 250), "garch-t": garch_t, "fhs": fhs}


# ----------------------------------------------------------------------------------------------- the study
def passes(s: cf.ConformalScores) -> bool:
    return (s.kupiec_p is not None and s.kupiec_p >= SIGNIFICANCE and s.cc_p is not None
            and s.cc_p >= SIGNIFICANCE)


def one_window(r: np.ndarray, start: int, refit_every: int | None, gamma: float) -> dict:
    out: dict = {}
    fcs = make_forecasters(r, start, refit_every)
    for model, f in fcs.items():
        for level in LEVELS:
            try:
                c = cf.aci_compare(r, f, level, gamma=gamma, start=start)
            except ValueError as exc:     # a model that cannot forecast the window is reported, never filled
                out[f"{model}@{level}"] = {"error": str(exc)}
                continue
            row = {}
            for side, s in (("unwrapped", c.unwrapped), ("wrapped", c.wrapped)):
                row[side] = {"hit_rate": s.hit_rate, "n": s.n, "n_hits": s.n_hits, "kupiec_p": s.kupiec_p,
                             "independence_p": s.independence_p, "cc_p": s.cc_p, "fz0": s.fz0_mean, "z2": s.z2,
                             "pass": passes(s)}
            row["fz0_change"] = c.fz0_change
            row["n_clipped"] = c.result.n_clipped
            out[f"{model}@{level}"] = row
    return out


def run_generator(reps: int, gamma: float, seed: int) -> dict:
    res: dict = {"source": "generator", "reps": reps, "gamma": gamma, "seed": seed,
                 "note": "HS-250 rolling; GARCH-t and FHS with parameters fitted once on the calm history (frozen)",
                 "scenarios": {}}
    for si, name in enumerate(SCENARIOS):
        rows = []
        for rep in range(reps):
            rng = np.random.default_rng([seed, si, rep])
            r, start = simulate_window(name, rng)
            rows.append(one_window(r, start, None, gamma))
        res["scenarios"][name] = summarise(rows)
    return res


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0]]
    out = {}
    for k in keys:
        ok = [row[k] for row in rows if "error" not in row[k]]
        if not ok:
            out[k] = {"n": 0}
            continue
        fz = [x["fz0_change"] for x in ok if x["fz0_change"] is not None]
        out[k] = {
            "n": len(ok),
            "pass_unwrapped": float(np.mean([x["unwrapped"]["pass"] for x in ok])),
            "pass_wrapped": float(np.mean([x["wrapped"]["pass"] for x in ok])),
            "kupiec_pass_unwrapped": float(np.mean([(x["unwrapped"]["kupiec_p"] or 0) >= SIGNIFICANCE for x in ok])),
            "kupiec_pass_wrapped": float(np.mean([(x["wrapped"]["kupiec_p"] or 0) >= SIGNIFICANCE for x in ok])),
            "indep_pass_unwrapped": float(np.mean([(x["unwrapped"]["independence_p"] or 0) >= SIGNIFICANCE
                                                   for x in ok])),
            "indep_pass_wrapped": float(np.mean([(x["wrapped"]["independence_p"] or 0) >= SIGNIFICANCE
                                                 for x in ok])),
            "hit_rate_unwrapped": float(np.mean([x["unwrapped"]["hit_rate"] for x in ok])),
            "hit_rate_wrapped": float(np.mean([x["wrapped"]["hit_rate"] for x in ok])),
            "fz0_change_median": float(np.median(fz)) if fz else None,
            "fz0_change_mean": float(np.mean(fz)) if fz else None,
            "z2_unwrapped": float(np.mean([x["unwrapped"]["z2"] for x in ok if x["unwrapped"]["z2"] is not None])),
            "z2_wrapped": float(np.mean([x["wrapped"]["z2"] for x in ok if x["wrapped"]["z2"] is not None])),
        }
    return out


def run_npz(path: Path, gamma: float, refit_every: int) -> dict:
    data = np.load(path)
    ids = sorted({k.split("__")[0] for k in data.files})
    res: dict = {"source": f"npz:{path.name}", "gamma": gamma, "refit_every": refit_every, "windows": {}}
    for w in ids:
        r, start = np.asarray(data[f"{w}__returns"], dtype=float), int(data[f"{w}__start"])
        t0 = time.time()
        res["windows"][w] = one_window(r, start, refit_every, gamma)
        print(f"{w}: {r.size - start} forecast days in {time.time() - t0:.0f}s", file=sys.stderr)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("generator", "npz"), default="generator")
    ap.add_argument("path", nargs="?", help="the npz file of the real windows (source npz)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.source == "npz":
        if not a.path:
            ap.error("source npz needs the path of the file")
        res = run_npz(Path(a.path), a.gamma, a.refit_every)
    else:
        res = run_generator(a.reps, a.gamma, a.seed)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
