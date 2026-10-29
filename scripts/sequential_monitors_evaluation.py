#!/usr/bin/env python3
"""BS-15: size, power and detection delay of the e-process sequential monitors against the fixed-window tests.

Three parts, each deterministic under ``--seed``:

1. VaR hits (generator G17, ``riskvalidation.generators.VarHits``): ``alpha = 0.01``, horizon ``T = 1000`` days,
   ``--reps``
   series per scenario. Scenarios: the null; a hit rate multiplied by 2, 3 and 4 (the severity ladder; the hypothesis is
   judged at 3 and 4); a first-order clustering with P(hit | hit) = 0.1, 0.2 and 0.4 at the right rate. Monitors:
   the rate
   e-process, the clustering e-process, their average (the battery). Comparators: the Kupiec test on the first 250 days
   (one reading), the Kupiec test on the trailing 250 days read every day from day 250, the Kupiec test on all
   days so far
   read every day from day 50, the Basel traffic light (first day with the trailing 250-day count outside the green
   zone) and the Christoffersen independence test on the trailing 250 days read every day from day 250.
2. Defaults against PD by grade: five grades (PD 0.5, 1, 2, 5, 10 %), 2000 obligors per grade and year, ten yearly
   steps; the null and every grade's PD multiplied by 1.05, 1.1, 1.25, 1.5 and 2 (1.05 and 1.1 were added after a
   first run in which 1.25 and above were all detected in the first two years, which does not separate the monitors).
   Monitors: the e-process combined by product and by average. Comparators: the chi-square test across grades on the
   cumulative counts, read once at year 10 and read every year.
3. Real data (only with ``--contraste-data DIR``, the ``data/derived`` folder of Contraste): C11 exceedance days
   of the 17
   models, ``equity-2007-09`` and ``fx-2008``, alpha 1 %, against the Basel traffic light and the Kupiec test on the
   trailing 250 days; C03 charge-offs by year of age against the predicted cumulative PD, per vintage.

The verdict rules are fixed here, before any run, from the plan (method-tiers-sdd-plan, BS-15):
  H1 size: the rate e-process rejects the null at most ``0.05 + 3 se`` of the series over the horizon, while the
     Kupiec test read every day on the trailing window rejects more than that.
  H2 delay: at hit-rate factor 3 and 4 the median detection day of the rate e-process is below 250 and below the median
     detection day of the Basel traffic light (undetected series count as never).
  H3 power cost: the Kupiec power at the 250-day horizon minus the e-process power by day 250 is below 0.10 at factor
     3 and 4.
  H4 real data: on C11 equity-2007-09 and fx-2008, among the models the traffic light flags, the median detection day
     of the e-process is below the median detection day of the traffic light, in both windows.
The hypothesis holds when H1 to H4 all hold; any that fails is reported as failed, the monitors are not retuned.

Usage: python scripts/sequential_monitors_evaluation.py [--reps 2000] [--seed 20261028] [--out result.json]
       [--contraste-data DIR] [--skip-pd]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import special, stats

from riskvalidation.generators import VarHits
from riskvalidation.market import frtb
from riskvalidation.validation.sequential import (
    detection_time,
    eprocess_battery,
    pd_grade_eprocess,
    var_hit_clustering_eprocess,
    var_hit_eprocess,
)

ALPHA, HORIZON, WINDOW, LEVEL = 0.01, 1000, 250, 0.05
INF = math.inf


def first_true(mask: np.ndarray, start: int = 0) -> np.ndarray:
    """1-based first column index (day) where ``mask[:, start:]`` is true, per row; inf when never."""
    m = mask[:, start:]
    any_ = m.any(axis=1)
    idx = np.argmax(m, axis=1) + start + 1
    return np.where(any_, idx, INF)


def summarise(days: np.ndarray, by: int) -> dict:
    """Rejection by the horizon, by day ``by``, detected fraction and median day (null when more than half never)."""
    med = float(np.median(days))
    return {"ever": float(np.mean(np.isfinite(days))), "by": float(np.mean(days <= by)),
            "median_day": None if math.isinf(med) else med,
            "median_day_among_detected": float(np.median(days[np.isfinite(days)])) if np.isfinite(days).any() else None}


def kupiec_reject(k: np.ndarray, t: np.ndarray, p: float, level: float = LEVEL) -> np.ndarray:
    pi = np.where(t > 0, k / t, 0.0)
    lr = -2.0 * (special.xlogy(t - k, 1 - p) + special.xlogy(k, p) - special.xlogy(t - k, 1 - pi)
                 - special.xlogy(k, pi))
    return stats.chi2.sf(np.maximum(lr, 0.0), 1) < level


def independence_reject(h: np.ndarray, window: int, level: float = LEVEL) -> np.ndarray:
    """Christoffersen independence LR on the trailing ``window`` days, column t true when the p-value is below level."""
    a, b = h[:, :-1].astype(float), h[:, 1:].astype(float)
    pairs = {"00": (1 - a) * (1 - b), "01": (1 - a) * b, "10": a * (1 - b), "11": a * b}
    n_rows, n_cols = h.shape
    cs = {k: np.concatenate([np.zeros((n_rows, 1)), np.cumsum(v, axis=1)], axis=1) for k, v in pairs.items()}
    out = np.zeros((n_rows, n_cols), dtype=bool)
    w = window - 1
    for t in range(window, n_cols + 1):  # day t uses pairs ending at pair index t-2 (0-based): w pairs
        n00, n01, n10, n11 = (cs[k][:, t - 1] - cs[k][:, t - 1 - w] for k in ("00", "01", "10", "11"))
        tot = n00 + n01 + n10 + n11
        pi = (n01 + n11) / tot
        d0, d1 = n00 + n01, n10 + n11
        pi01 = np.divide(n01, d0, out=np.zeros_like(n01), where=d0 > 0)
        pi11 = np.divide(n11, d1, out=np.zeros_like(n11), where=d1 > 0)
        null = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
        alt = (special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01) + special.xlogy(n10, 1 - pi11)
               + special.xlogy(n11, pi11))
        lr = np.maximum(-2.0 * (null - alt), 0.0)
        out[:, t - 1] = (stats.chi2.sf(lr, 1) < level) & (lr > 0)
    return out


def basel_first_non_green(h: np.ndarray, window: int = WINDOW, coverage: float = 0.99) -> np.ndarray:
    c = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, axis=1)], axis=1)
    k_min = next(k for k in range(window + 1) if frtb.traffic_light_zone(k, window, coverage) != "green")
    out = np.zeros(h.shape, dtype=bool)
    for t in range(window, h.shape[1] + 1):
        out[:, t - 1] = (c[:, t] - c[:, t - window]) >= k_min
    return first_true(out, window - 1), k_min


def run_hits(reps: int, seed: int) -> dict:
    scenarios = [("null", dict()), ("rate x2", dict(rate_factor=2.0)), ("rate x3", dict(rate_factor=3.0)),
                 ("rate x4", dict(rate_factor=4.0)), ("cluster 0.1", dict(cluster=0.1)),
                 ("cluster 0.2", dict(cluster=0.2)), ("cluster 0.4", dict(cluster=0.4))]
    out: dict = {"alpha": ALPHA, "horizon": HORIZON, "window": WINDOW, "reps": reps, "scenarios": {}}
    for si, (name, kw) in enumerate(scenarios):
        gen = VarHits(t=HORIZON, alpha=ALPHA, with_var=False, **kw)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 15, si]))
        h = np.array([gen.draw(rng)["hits"] for _ in range(reps)])
        rate = np.array([var_hit_eprocess(x, ALPHA).log_e for x in h])
        rate_up = np.array([var_hit_eprocess(x, ALPHA, direction="upper").log_e for x in h])
        clus = np.array([var_hit_clustering_eprocess(x, ALPHA).log_e for x in h])
        bat = np.array([eprocess_battery({"rate": var_hit_eprocess(x, ALPHA),
                                          "cluster": var_hit_clustering_eprocess(x, ALPHA)}).log_e for x in h])
        cross = -math.log(LEVEL)
        days = np.arange(1, HORIZON + 1, dtype=float)
        c = np.cumsum(h, axis=1).astype(float)
        c250 = np.concatenate([np.zeros((reps, 1)), c], axis=1)
        trailing = c250[:, WINDOW:] - c250[:, :-WINDOW]
        k_trail = np.zeros((reps, HORIZON))
        k_trail[:, WINDOW - 1:] = trailing
        kup_trail = np.zeros((reps, HORIZON), dtype=bool)
        kup_trail[:, WINDOW - 1:] = kupiec_reject(trailing, np.full(trailing.shape, float(WINDOW)), ALPHA)
        kup_cum = kupiec_reject(c, np.broadcast_to(days, c.shape), ALPHA)
        kup_cum[:, :49] = False
        tl_days, k_min = basel_first_non_green(h)
        ind = independence_reject(h, WINDOW)
        one_shot = kupiec_reject(c[:, WINDOW - 1], np.full(reps, float(WINDOW)), ALPHA)
        res = {
            "e_rate": summarise(first_true(rate >= cross), WINDOW),
            "e_rate_upper": summarise(first_true(rate_up >= cross), WINDOW),
            "e_cluster": summarise(first_true(clus >= cross), WINDOW),
            "e_battery": summarise(first_true(bat >= cross), WINDOW),
            "kupiec_one_shot_250": {"power": float(one_shot.mean())},
            "kupiec_trailing_250_daily": summarise(first_true(kup_trail), WINDOW),
            "kupiec_cumulative_daily": summarise(first_true(kup_cum), WINDOW),
            "basel_traffic_light": summarise(tl_days, WINDOW),
            "independence_trailing_250_daily": summarise(first_true(ind), WINDOW),
            "basel_first_non_green_count": int(k_min),
        }
        out["scenarios"][name] = res
        print(f"[hits] {name}: e_rate {res['e_rate']}  kupiec_trailing {res['kupiec_trailing_250_daily']}  "
              f"basel {res['basel_traffic_light']}", flush=True)
    return out


def run_pd(reps: int, seed: int) -> dict:
    pd = np.array([0.005, 0.01, 0.02, 0.05, 0.10])
    n_obl, steps = 2000, 10
    out: dict = {"pd": pd.tolist(), "obligors_per_grade_year": n_obl, "years": steps, "reps": reps, "scenarios": {}}
    for si, factor in enumerate([1.0, 1.05, 1.1, 1.25, 1.5, 2.0]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 16, si]))
        d = rng.binomial(n_obl, np.minimum(pd * factor, 0.999), size=(reps, steps, pd.size)).astype(float)
        n = np.full((steps, pd.size), float(n_obl))
        prod_days, avg_days = [], []
        for r in range(reps):
            prod_days.append(detection_time(pd_grade_eprocess(d[r], n, pd, combine="product").log_e) or INF)
            avg_days.append(detection_time(pd_grade_eprocess(d[r], n, pd, combine="average").log_e) or INF)
        cum_d, cum_n = np.cumsum(d, axis=1), np.cumsum(np.broadcast_to(n, d.shape), axis=1)
        chi2 = (((cum_d - cum_n * pd) ** 2) / (cum_n * pd * (1 - pd))).sum(axis=2)
        rej = stats.chi2.sf(chi2, pd.size) < LEVEL
        res = {"e_product": summarise(np.array(prod_days), 3), "e_average": summarise(np.array(avg_days), 3),
               "chi2_one_shot_year_10": {"power": float(rej[:, -1].mean())},
               "chi2_read_every_year": summarise(first_true(rej), 3)}
        out["scenarios"][f"PD x{factor:g}"] = res
        print(f"[pd] x{factor:g}: {res}", flush=True)
    return out


def run_c11(data: Path) -> dict:
    out: dict = {}
    for window in ("equity-2007-09", "fx-2008"):
        doc = json.loads((data / "C11" / f"{window}.json").read_text(encoding="utf-8"))
        n_days = doc["outputs"]["window"]["n_forecast_days"]
        rows = []
        for m in doc["outputs"]["models"]:
            lvl = next(b for b in m["by_level"] if abs(b["alpha"] - ALPHA) < 1e-12)
            h = np.zeros((1, n_days), dtype=bool)
            h[0, lvl["exceedance_days"]] = True
            e_day = detection_time(var_hit_eprocess(h[0], ALPHA).log_e)
            tl, _ = basel_first_non_green(h) if n_days >= WINDOW else (np.array([INF]), None)
            trail = np.zeros((1, n_days), dtype=bool)
            if n_days >= WINDOW:
                c = np.concatenate([[0.0], np.cumsum(h[0])])
                k = c[WINDOW:] - c[:-WINDOW]
                trail[0, WINDOW - 1:] = kupiec_reject(k, np.full(k.shape, float(WINDOW)), ALPHA)
            cum = kupiec_reject(np.cumsum(h[0]).astype(float), np.arange(1, n_days + 1, dtype=float), ALPHA)[None, :]
            cum[:, :49] = False
            rows.append({"model": m["model_id"], "n_exceedances": int(h.sum()), "e_process_day": e_day,
                         "kupiec_cumulative_day": None if not cum.any() else int(first_true(cum)[0]),
                         "traffic_light_day": None if math.isinf(tl[0]) else int(tl[0]),
                         "kupiec_trailing_day": None if not trail.any() else int(first_true(trail)[0])})
        flagged = [r for r in rows if r["traffic_light_day"] is not None]
        e_days = [r["e_process_day"] if r["e_process_day"] is not None else INF for r in flagged]
        tl_days = [r["traffic_light_day"] for r in flagged]
        med_e, med_tl = (float(np.median(e_days)), float(np.median(tl_days))) if flagged else (None, None)
        out[window] = {"n_days": n_days, "models": rows, "n_flagged_by_traffic_light": len(flagged),
                       "n_flagged_by_e_process": sum(r["e_process_day"] is not None for r in rows),
                       "n_flagged_by_traffic_light_missed_by_e_process": sum(math.isinf(x) for x in e_days),
                       "median_e_day_among_traffic_light_flagged": med_e if med_e is None or math.isfinite(med_e)
                       else None, "median_traffic_light_day": med_tl,
                       "e_process_earlier": bool(flagged) and med_e is not None and med_e < med_tl}
        print(f"[c11] {window}: flagged TL {len(flagged)}, E {out[window]['n_flagged_by_e_process']}, "
              f"median E day {med_e}, median TL day {med_tl}", flush=True)
    return out


def run_c03(data: Path) -> dict:
    out: dict = {}
    for path in sorted((data / "C03").glob("fy*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        curves = [c for c in doc["outputs"]["vintage_curves"] if c.get("reason") is None]
        if not curves:
            continue
        n_vals = np.array([c["n"] for c in curves], dtype=float)
        pred = np.array([c["predicted_cumulative_pd"] for c in curves], dtype=float)
        obs = np.array([c["observed_cumulative_pd"] for c in curves], dtype=float)
        # steps are the five years of age; grades are the vintages; the interval PD is the increment of the
        # cumulative PD
        p_int = np.diff(np.concatenate([np.zeros((len(curves), 1)), pred], axis=1), axis=1).T
        o_int = np.diff(np.concatenate([np.zeros((len(curves), 1)), obs], axis=1), axis=1).T
        n_mat = np.tile(n_vals, (p_int.shape[0], 1))
        d_mat = np.round(o_int * n_mat)
        res = pd_grade_eprocess(d_mat, n_mat, np.clip(p_int, 1e-9, 1 - 1e-9), direction="upper",
                                grades=[str(c["fiscal_year"]) for c in curves])
        out[path.stem] = {"vintages": [c["fiscal_year"] for c in curves], "months": curves[0]["months"],
                          "first_age_step_rejected": res.rejection_time, "anytime_p_value": res.anytime_p_value,
                          "grade_first_age_step": res.grade_rejection_times(),
                          "log_e_by_age": [round(float(v), 3) for v in res.log_e]}
        print(f"[c03] {path.stem}: {out[path.stem]}", flush=True)
    return out


def verdict(hits: dict, c11: dict | None) -> dict:
    s = hits["scenarios"]
    mc = 3.0 * math.sqrt(0.05 * 0.95 / hits["reps"])
    size_e = s["null"]["e_rate"]["ever"]
    size_k = s["null"]["kupiec_trailing_250_daily"]["ever"]
    size_kc = s["null"]["kupiec_cumulative_daily"]["ever"]
    h1 = size_e <= 0.05 + mc and max(size_k, size_kc) > 0.05 + mc
    h2, h3, detail = True, True, {}
    for name in ("rate x3", "rate x4"):
        r = s[name]
        e_med, tl_med = r["e_rate"]["median_day"], r["basel_traffic_light"]["median_day"]
        ok2 = e_med is not None and e_med < WINDOW and (tl_med is None or e_med < tl_med)
        cost = s[name]["kupiec_one_shot_250"]["power"] - r["e_rate"]["by"]
        ok3 = cost < 0.10
        h2, h3 = h2 and ok2, h3 and ok3
        detail[name] = {"e_median_day": e_med, "traffic_light_median_day": tl_med, "H2": ok2,
                        "kupiec_power_250": s[name]["kupiec_one_shot_250"]["power"],
                        "e_power_by_250": r["e_rate"]["by"],
                        "power_cost": cost, "H3": ok3}
    out = {"H1_size": {"holds": h1, "e_process_size": size_e, "kupiec_trailing_daily_size": size_k,
                       "kupiec_cumulative_daily_size": size_kc, "bound": 0.05 + mc},
           "H2_delay": {"holds": h2}, "H3_power_cost": {"holds": h3}, "detail": detail}
    if c11 is not None:
        h4 = all(c11[w]["e_process_earlier"] for w in c11)
        out["H4_real_data"] = {"holds": h4, **{w: {"median_e_day": c11[w]["median_e_day_among_traffic_light_flagged"],
                                                   "median_traffic_light_day": c11[w]["median_traffic_light_day"]}
                                              for w in c11}}
    parts = [v["holds"] for k, v in out.items() if k.startswith("H")]
    out["hypothesis_holds"] = all(parts) and c11 is not None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--contraste-data", type=Path, default=None)
    ap.add_argument("--skip-pd", action="store_true")
    args = ap.parse_args()
    result: dict = {"seed": args.seed, "hits": run_hits(args.reps, args.seed)}
    if not args.skip_pd:
        result["pd_grades"] = run_pd(args.reps, args.seed)
    c11 = None
    if args.contraste_data is not None:
        c11 = run_c11(args.contraste_data)
        result["c11"] = c11
        result["c03"] = run_c03(args.contraste_data)
    result["verdict"] = verdict(result["hits"], c11)
    print(json.dumps(result["verdict"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
