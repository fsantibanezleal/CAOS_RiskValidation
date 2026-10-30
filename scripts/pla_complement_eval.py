"""BS-13 evaluation: the power and size of the PLA complement on the six C13 variants and a stale-volatility lag ladder.

Runs the committed C13 desk P&L generator of the CAOS_Contraste product (read only; pass its checkout with
--contraste) through `riskvalidation.market.pla_complement`. The thresholds are calibrated on 10,000 independent null
desk-years (the `none` variant under ten case seeds that share no seed with the committed ones), then applied,
unchanged, to the 6,000 committed replications (case seed 13, 1,000 per variant) and to a ladder of stale-volatility
lags (5, 10, 20, 40 days; 1,000 replications each, the committed stale-volatility seeds). Nothing is tuned: the
sizes, the window and the correction are fixed before the run.

    PYTHONPATH=<worktree>/src python scripts/pla_complement_eval.py \
        --contraste <CAOS_Contraste checkout> --out result.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from riskvalidation.market import frtb
from riskvalidation.market import pla_complement as pc

NULL_CASE_SEEDS = tuple(range(5013, 5023))   # ten null blocks of 1,000: seeds 5,013,000 to 5,022,999
SIZES = (0.001, 0.005, 0.01, 0.02, 0.05)
LAGS = (5, 10, 20, 40)
FIELDS = ("wald", "spearman_abs", "spearman_roll")


def wilson(k: int, n: int) -> list[float]:
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return [round(float(ci.low), 5), round(float(ci.high), 5)]


def rate(flags: list[bool]) -> dict:
    k, n = int(np.sum(flags)), len(flags)
    return {"k": k, "n": n, "rate": round(k / n, 5), "wilson95": wilson(k, n)}


@contextlib.contextmanager
def patched(module, **values):
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def run_block(gen, did: str, n: int, window: int, case_seed: int = 13):
    out = []
    for i in range(n):
        rep = gen.replicate(did, i, case_seed)
        out.append((pc.pla_complement_statistics(rep.hpl, rep.rtpl, window), frtb.pla_zone_exact(rep.hpl, rep.rtpl),
                    frtb.pla_ks(rep.hpl, rep.rtpl)))
    return out


def summarise(block, th_none, th_bon) -> dict:
    zones = [z for _, z, _ in block]
    ks = [k for _, _, k in block]
    res = {"n": len(block), "pla_amber_or_red": rate([z != "green" for z in zones]),
           "pla_red": rate([z == "red" for z in zones]),
           "ks_not_green": rate([k >= 0.09 for k in ks]), "ks_red": rate([k > 0.12 for k in ks])}
    for tag, th in (("per_statistic_1pct", th_none), ("family_1pct_bonferroni", th_bon)):
        dec = [pc.pla_complement_decide(s, th) for s, _, _ in block]
        res[tag] = {
            "wald": rate([d.flag_wald for d in dec]),
            "spearman_abs": rate([d.flag_spearman_abs for d in dec]),
            "spearman_roll": rate([d.flag_spearman_roll for d in dec]),
            "any": rate([d.flagged for d in dec]),
            "any_or_pla": rate([d.flagged or z != "green" for d, z in zip(dec, zones, strict=True)]),
        }
    res["median_slope"] = round(float(np.median([s.slope for s, _, _ in block])), 4)
    res["median_spearman_roll"] = round(float(np.median([s.spearman_roll for s, _, _ in block])), 4)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contraste", required=True, help="a CAOS_Contraste checkout (read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=pc.DEFAULT_WINDOW)
    ap.add_argument("--reps", type=int, default=1000, help="replications per variant (the committed run is 1000)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.contraste) / "data-pipeline"))
    from pipeline.cases.c13 import generator as gen
    from pipeline.cases.c13 import params

    t0 = time.time()
    null = []
    for cs in NULL_CASE_SEEDS:
        null += [s for s, _, _ in run_block(gen, "none", args.reps, args.window, cs)]
    th_none = pc.pla_complement_calibrate(null, 0.01, "none")
    th_bon = pc.pla_complement_calibrate(null, 0.01, "bonferroni")
    result = {
        "generator": params.GENERATOR, "window": args.window, "n_null": len(null), "null_case_seeds":
        list(NULL_CASE_SEEDS), "thresholds_per_statistic_1pct": th_none.__dict__, "thresholds_family_1pct":
        th_bon.__dict__, "variants": {}, "lag_ladder": {}, "roc_stale_volatility": {}, "size_robustness": {}}

    blocks = {}
    for did in params.IDS:
        blocks[did] = run_block(gen, did, args.reps, args.window)
        result["variants"][did] = summarise(blocks[did], th_none, th_bon)

    # ROC of the three statistics on the committed stale-volatility variant (power at several sizes)
    for size in SIZES:
        th = pc.pla_complement_calibrate(null, size, "none")
        dec = [pc.pla_complement_decide(s, th) for s, _, _ in blocks["stale-volatility"]]
        dec0 = [pc.pla_complement_decide(s, th) for s, _, _ in blocks["none"]]
        result["roc_stale_volatility"][str(size)] = {
            f: {"power": rate([getattr(d, "flag_" + f) for d in dec]),
                "size_held_out": rate([getattr(d, "flag_" + f) for d in dec0])}
            for f in FIELDS}

    # lag ladder on the stale-volatility seeds; lag 20 is the committed variant
    stale = params.deficiency("stale-volatility")["parameters"][0]
    committed_lag = stale["value"]
    try:
        for lag in LAGS:
            stale["value"] = float(lag)
            block = run_block(gen, "stale-volatility", args.reps, args.window)
            result["lag_ladder"][str(lag)] = summarise(block, th_none, th_bon)
    finally:
        stale["value"] = committed_lag

    # size robustness: a null whose volatility is less persistent and less dispersed than the one the thresholds were
    # calibrated on
    for phi, sd in ((0.90, 0.35), (0.98, 0.20)):
        eta = sd * np.sqrt(1.0 - phi ** 2)
        with patched(gen, VOL_PHI=phi, VOL_SD=sd, _ETA=eta, _MU=-sd ** 2):
            block = run_block(gen, "none", args.reps, args.window)
        result["size_robustness"][f"phi={phi},sd={sd}"] = summarise(block, th_none, th_bon)
    result["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_null", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
