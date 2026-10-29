#!/usr/bin/env python3
"""The evaluation of BS-14 (``riskvalidation.transitions.withdrawals``): coverage of the true PD under informative
rating withdrawals, and the anchored k on S&P's real cohorts.

Known truth: the 7-grade S&P-like generator of ``tests/fixtures/withdrawals_sp_generator.json`` (the EM fit of S&P's
pooled annual CEREP counts, computed by Contraste's C04 case) drives ``generators.paths.RatingPaths`` with withdrawal at
6% a year and 6% (1 + k) in the year before default (``informative`` k = 0, 1, 3, 9), five years, annual snapshots. Each
repetition draws the latent paths and the observed ones from the same stream (the engine draws the withdrawals after the
path, so the observed paths are the latent ones truncated at a withdrawal), and the true PD of a grade is
``RatingPaths.truth()``'s one-year PD.

Two designs:

- ``first_year`` (the model's exact setting): the first annual cohort, where nobody has been withdrawn before it.
  Treatments compared: the **industry** estimate (withdrawals removed, k = 1; CEREP's transition page), the
  **unfollowed** one (kept in the denominator, k = 0), the **followed** one (EBA/GL/2017/16 paragraph 76: the cohort's
  end read on the latent paths; it needs information no transition page has, so it is an oracle here, and it is the
  default page's count that the anchor takes) and the **anchored** interval of ``withdrawal_bounds`` with the second
  source counting the defaults of the same cohort (``anchor="same"``: CEREP's default page against its transition
  page), or of an independent cohort of the same size (``anchor="independent"``), optionally capturing only a fraction
  of the true defaults (``capture`` < 1: a stricter definition of default in the second source). Every interval is the
  Jeffreys interval of the count it uses at 95%, except the anchored one (the profile over the anchor's k band).
- ``pooled``: the five annual cohorts pooled, as Contraste's C04 pools them. A rating withdrawn in an earlier year is
  missing from a later cohort, a selection the one-period model does not see; this design measures that limit.

Real data: ``real_data_study`` reads the S&P cohorts of ``tests/fixtures/withdrawals_sp_cerep.json`` (ESMA CEREP, tab 2
default page against tab 4 transitions) and computes the anchored k per cohort and pooled, per grade. The stability
criterion is fixed here before the run: the anchored k is *stable* when at least 80% of the cohorts' k bands contain the
pooled k (the cohorts are consistent with one informativeness).

Usage: python scripts/withdrawal_bounds_evaluation.py [--reps 200] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "src"))

from riskvalidation.generators.paths import RatingPaths  # noqa: E402
from riskvalidation.transitions.intervals import pd_jeffreys  # noqa: E402
from riskvalidation.transitions.withdrawals import withdrawal_bounds  # noqa: E402

WITHDRAWAL_RATE = 0.06
WINDOW = 1.0
YEARS = 5
LEVEL = 0.95
STABILITY_SHARE = 0.80
CRITERION = (
    "stable when at least 80% of the cohorts' k bands (Jeffreys interval of the default page's rate at 95%) contain "
    "the pooled anchored k; fixed before the run"
)


def load_generator() -> dict[str, Any]:
    return json.loads((FIXTURES / "withdrawals_sp_generator.json").read_text(encoding="utf-8"))


def _paths(fx: dict[str, Any], scale: int = 1, **defect: Any) -> RatingPaths:
    return RatingPaths(np.asarray(fx["q"], dtype=float), tuple(int(o) * scale for o in fx["obligors"]),
                       horizon=float(YEARS),
                       snapshots=tuple(float(t) for t in range(YEARS + 1)), **defect)


def _record(cover: np.ndarray, width: np.ndarray, point: np.ndarray, truth: float) -> dict[str, float]:
    reps = cover.shape[0]
    cov = float(cover.mean())
    return {"coverage": cov, "coverage_mcse": math.sqrt(cov * (1 - cov) / reps), "mean_width": float(width.mean()),
            "mean_point": float(np.nanmean(point)), "relative_bias": float(np.nanmean(point) / truth - 1.0)}


def coverage_study(informative: float, reps: int = 200, seed: int = 20261028, *, design: str = "first_year",
                   anchor: str = "same", capture: float = 1.0, scale: int = 1, level: float = LEVEL) -> dict[str, Any]:
    """Coverage of the true one-year PD by grade, mean interval width and relative bias of the point, for the
    industry, unfollowed, followed and anchored treatments, over ``reps`` paired draws; ``scale`` multiplies every
    grade's cohort (the true PD does not change). See the module text."""
    if design not in ("first_year", "pooled") or anchor not in ("same", "independent"):
        raise ValueError("design is 'first_year' or 'pooled'; anchor is 'same' or 'independent'")
    if not 0.0 < capture <= 1.0:
        raise ValueError("capture must lie in (0, 1]")
    fx = load_generator()
    grades = fx["grades"]
    g = len(grades)
    latent_gen = _paths(fx, scale)
    gen = _paths(fx, scale, withdrawal_rate=WITHDRAWAL_RATE, informative=informative, window=WINDOW)
    truth = np.asarray(latent_gen.truth()["pd_by_grade"]["1"], dtype=float)
    children = np.random.SeedSequence(seed).spawn(reps)
    second = np.random.SeedSequence(seed + 1).spawn(reps)
    names = ("industry", "unfollowed", "followed", "anchored", "identified")
    cover = {k: np.zeros((reps, g), dtype=bool) for k in names}
    width = {k: np.zeros((reps, g)) for k in names}
    point = {k: np.zeros((reps, g)) for k in names}
    k_hat = np.zeros((reps, g))
    m = latent_gen.n_states
    for r, (ch, ch2) in enumerate(zip(children, second, strict=True)):
        lat = latent_gen.draw(np.random.default_rng(ch))
        obs = gen.draw(np.random.default_rng(ch))
        if design == "first_year":
            steps = [obs["steps"][0]]
            follow = [latent_gen.snapshot_counts(np.column_stack([obs["snapshot_states"][:, 0],
                                                                  lat["snapshot_states"][:, 1]]))[0]]
        else:
            steps = obs["steps"]
            o, la = obs["snapshot_states"], lat["snapshot_states"]
            follow = [latent_gen.snapshot_counts(np.column_stack([o[:, t], la[:, t + 1]]))[0] for t in range(YEARS)]
        counts = sum(s["counts"] for s in steps)
        withdrawn = sum(s["withdrawn"] for s in steps)
        followed_defaults = sum(f["counts"][:, m - 1] for f in follow)[: m - 1]
        if anchor == "same":
            a = followed_defaults.copy()
            na = None
        else:  # an independent cohort of the same size: its latent defaults, counted whether withdrawn or not
            lat2 = latent_gen.draw(np.random.default_rng(ch2))
            a = lat2["steps"][0]["counts"][: m - 1, m - 1].copy()
            na = lat2["steps"][0]["counts"][: m - 1].sum(axis=1)
        if capture < 1.0:
            a = np.random.default_rng(ch2).binomial(a.astype(int), capture).astype(float)
        res = withdrawal_bounds(counts, withdrawn, anchor_defaults=a, anchor_obligors=na, level=level)
        d, s, n = res["defaults"], res["survivors"], res["n"]
        intervals = {
            "industry": pd_jeffreys(d, d + s, level=level),
            "unfollowed": pd_jeffreys(d, n, level=level),
            "followed": pd_jeffreys(followed_defaults, n, level=level),
        }
        for name, iv in intervals.items():
            lo, hi, est = np.asarray(iv["lower"]), np.asarray(iv["upper"]), np.asarray(iv["estimate"])
            cover[name][r] = (lo <= truth) & (truth <= hi)
            width[name][r], point[name][r] = hi - lo, est
        an = res["anchor"]
        cover["anchored"][r] = (an["lower"] <= truth) & (truth <= an["upper"])
        width["anchored"][r], point["anchored"][r] = an["upper"] - an["lower"], an["pd"]
        idn = res["identified"]
        cover["identified"][r] = (idn["lower"] <= truth) & (truth <= idn["upper"])
        width["identified"][r], point["identified"][r] = idn["upper"] - idn["lower"], res["pd_industry"]
        k_hat[r] = an["k"]
    out: dict[str, Any] = {"informative": informative, "design": design, "anchor": anchor, "capture": capture,
                           "scale": scale, "reps": reps, "seed": seed, "level": level, "grades": {},
                           "truth": truth.tolist()}
    for i, name in enumerate(grades):
        rec = {t: _record(cover[t][:, i], width[t][:, i], point[t][:, i], truth[i]) for t in names}
        finite = k_hat[:, i][np.isfinite(k_hat[:, i])]
        rec["anchored_k"] = {"median": float(np.median(finite)) if finite.size else None,
                             "q05": float(np.quantile(finite, 0.05)) if finite.size else None,
                             "q95": float(np.quantile(finite, 0.95)) if finite.size else None,
                             "share_infinite": float(np.mean(np.isinf(k_hat[:, i])))}
        out["grades"][name] = rec
    return out


def verdict(res: dict[str, Any], grade: str = "B") -> dict[str, Any]:
    """The plan's hypothesis (i) read off one coverage run, for a grade: the anchored interval covers at least 90%
    (within one Monte Carlo standard error), and each treatment a transition page alone supports (industry,
    unfollowed) covers under 50%. The followed treatment is reported apart: it reads the cohort's end on the latent
    paths, information no transition page has (it is the default page's count, the anchor's input)."""
    r = res["grades"][grade]
    a, ind, unf, fol = (r[t]["coverage"] for t in ("anchored", "industry", "unfollowed", "followed"))
    return {"grade": grade, "reps": res["reps"], "anchored_coverage": a,
            "anchored_covers_at_least_90": a + r["anchored"]["coverage_mcse"] >= 0.90,
            "industry_coverage": ind, "unfollowed_coverage": unf, "followed_coverage": fol,
            "industry_and_unfollowed_below_50": ind < 0.5 and unf < 0.5,
            "anchored_better_than_industry": a > ind}


def real_data_study(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """The anchored k of S&P's real cohorts: the default page (tab 2) as the anchor of the transition page (tab 4)."""
    fx = fixture if fixture is not None else json.loads((FIXTURES / "withdrawals_sp_cerep.json").read_text(
        encoding="utf-8"))
    out: dict[str, Any] = {"criterion": CRITERION, "grades": {}}

    def one(d: float, s: float, w: float, a: float, na: float) -> dict[str, Any]:
        r = withdrawal_bounds([[s, d], [0.0, 0.0]], [w, 0.0], anchor_defaults=[a], anchor_obligors=[na])
        an = r["anchor"]
        return {"d4_defaults": d, "stayers": s, "withdrawn": w, "d2_defaults": a, "d2_cohort": na,
                "pd_d4_industry": float(r["pd_industry"][0]), "pd_d2": float(an["anchor_rate"][0]),
                "pd_anchored": float(an["pd"][0]), "k": float(an["k"][0]), "k_low": float(an["k_low"][0]),
                "k_high": float(an["k_high"][0]), "flag": an["flag"][0], "reason": r["reason"][0]}

    for gi, grade in enumerate(fx["grades"]):
        cohorts, sums = [], np.zeros(5)
        for c in fx["cohorts"]:
            if c["d2_defaults"] is None or c["d2_cohort"] is None:
                continue
            d, s, w = float(c["d4_defaults"][gi]), float(c["stayers"][gi]), float(c["withdrawn"][gi])
            a, na = float(c["d2_defaults"][gi]), float(c["d2_cohort"][gi])
            if d + s + w <= 0 or na <= 0 or a > na or w <= 0:
                continue
            row = one(d, s, w, a, na)
            row["cohort"] = c["label"]
            cohorts.append(row)
            sums += (d, s, w, a, na)
        if not cohorts:
            continue
        pooled = one(*sums)
        usable = [c for c in cohorts if c["reason"] == "ok" and np.isfinite(c["k_low"]) and not np.isnan(c["k_high"])]
        kp = pooled["k"]
        contain = [c for c in usable if c["k_low"] <= kp <= c["k_high"]]
        ks = np.array([c["k"] for c in usable if np.isfinite(c["k"])])
        out["grades"][grade] = {
            "cohorts": cohorts, "pooled": pooled,
            "k_over_cohorts": {"n": int(ks.size), "median": float(np.median(ks)) if ks.size else None,
                               "q25": float(np.quantile(ks, 0.25)) if ks.size else None,
                               "q75": float(np.quantile(ks, 0.75)) if ks.size else None,
                               "share_above_1": float(np.mean(ks > 1.0)) if ks.size else None},
            "stability": {"usable_cohorts": len(usable), "cohorts_containing_pooled_k": len(contain),
                          "share_cohorts_containing_pooled_k": len(contain) / len(usable) if usable else 0.0,
                          "stable": bool(usable) and len(contain) / len(usable) >= STABILITY_SHARE},
            "k_above_1": {"pooled_k": kp, "pooled_k_low": pooled["k_low"],
                          "holds": bool(kp > 1.0), "holds_with_lower_bound": bool(pooled["k_low"] > 1.0)},
        }
    return out


def _fmt(x: float) -> str:
    return "inf" if x == math.inf else f"{x:.3g}"


def report(coverage: list[dict[str, Any]], real: dict[str, Any]) -> str:
    lines = []
    for res in coverage:
        lines.append(f"\n### {res['design']}, anchor {res['anchor']}, capture {res['capture']}, scale "
                     f"{res['scale']}, informative {res['informative']:g}, "
                     f"{res['reps']} repetitions (seed {res['seed']})\n")
        lines.append("| Grade | true PD | " + " | ".join(f"{t} cov." for t in ("industry", "unfollowed", "followed",
                     "anchored", "identified")) + " | anchored width | anchored bias | median k |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for gi, (name, rec) in enumerate(res["grades"].items()):
            cov = " | ".join(f"{rec[t]['coverage']:.3f}" for t in ("industry", "unfollowed", "followed", "anchored",
                                                                       "identified"))
            lines.append(f"| {name} | {res['truth'][gi]:.4%} | {cov} | {rec['anchored']['mean_width']:.4%} | "
                         f"{rec['anchored']['relative_bias']:+.1%} | {_fmt(rec['anchored_k']['median'] or math.nan)} |")
    lines.append("\n### S&P real cohorts\n")
    lines.append("| Grade | cohorts | pooled k | pooled k band | median k | share k>1 | cohorts containing pooled k "
                 "| stable |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, g in real["grades"].items():
        p, o, st = g["pooled"], g["k_over_cohorts"], g["stability"]
        med = _fmt(o["median"]) if o["median"] is not None else "n/a"
        sh = "n/a" if o["share_above_1"] is None else format(o["share_above_1"], ".2f")
        lines.append(f"| {name} | {len(g['cohorts'])} | {_fmt(p['k'])} | {_fmt(p['k_low'])} to {_fmt(p['k_high'])} | "
                     f"{med} | {sh} | {st['cohorts_containing_pooled_k']}/{st['usable_cohorts']} | {st['stable']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="only informative 9, first-year design")
    args = ap.parse_args()
    runs = [dict(informative=9.0)] if args.quick else (
        [dict(informative=k) for k in (0.0, 1.0, 3.0, 9.0)]
        + [dict(informative=9.0, scale=10), dict(informative=9.0, anchor="independent"),
           dict(informative=9.0, capture=0.7), dict(informative=9.0, design="pooled")])
    coverage = [coverage_study(reps=args.reps, seed=args.seed, **r) for r in runs]
    real = real_data_study()
    print(report(coverage, real))
    if args.out is not None:
        args.out.write_text(json.dumps({"coverage": coverage, "real": real}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
