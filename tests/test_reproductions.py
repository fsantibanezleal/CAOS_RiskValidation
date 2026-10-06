"""RV-308 to RV-310: published simulation studies reproduced by the harness, cell by cell.

Each published rate is itself a simulation estimate, so a cell agrees when the difference is within 3.29 standard
errors of the difference at the pooled rate (both Monte Carlo errors). The cells that do not reproduce are listed with
the evidence; the suite fails if any other cell disagrees, or if a listed one starts to agree (the documentation would
then be wrong).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from riskvalidation.generators import BinormalScores, DefaultCounts, NestedLda, TwoSample, feed_series
from riskvalidation.harness import (
    Rate,
    agrees_with_published,
    metric_above,
    simulate,
    traffic_lights_rejection_probability,
    within_size,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TESTS = {"normal": ("pd.normal_multiperiod", {}), "traffic": ("pd.traffic_lights", {"ties": "upper"})}

# Table 8, DV_LV, traffic lights at 0.1%: printed 0.955; this suite 0.9467 at 25,000 runs and 0.9475 (SE 0.0005) at
# 200,000 runs; the neighbouring levels agree (0.935 and 0.936 against 0.937 and 0.938). A misprint of 0.945 would fit.
WP14_NOT_REPRODUCED = {("8", "DV_LV", "traffic", 0.001)}
# Table 4, m 100, n 400, shift 1/2 SD, PSI > 0.25: printed 0.787; this suite 0.817 (0.8175 at 20,000 runs); no
# empty-bin convention explains it (half-observation floor 0.8175, dropping empty bins 0.8075, adding one half to every
# bin 0.7406 but then the other rows fail); the other 53 cells agree.
YN_NOT_REPRODUCED = {(100, 400, 0.5, "metric>0.25")}


def _wp14_runs(table: str, scenarios: dict, published: dict, levels: list[float], seed0: int) -> dict:
    outcome = {}
    seed = seed0
    for name, sc in scenarios.items():
        gen = DefaultCounts.over_years([1000] * 5, np.array(sc["true_pct"]) / 100.0, rho=sc["rho"],
                                       pd_forecast=np.array(sc["forecast_pct"]) / 100.0, theta=sc["theta"])
        for short, (tid, kw) in TESTS.items():
            seed += 1
            sim = simulate(tid, gen, n_rep=25_000, seed=seed, feed=feed_series(), alphas=levels, test_kwargs=kw)
            for level, pub in zip(levels, published[name][short], strict=True):
                r = sim.rate(level)
                if table == "8":  # type II error = 1 - power
                    r = Rate(r.n - r.rejections, r.n, r.undefined)
                ok, z = agrees_with_published(r, pub, 25_000)
                outcome[(table, name, short, level)] = (ok, r.rate, pub, z)
    return outcome


def test_wp14_tables_7_and_8():
    """RV-308: 143 of WP14's 144 cells, the normal test as printed and the traffic lights with ``ties="upper"``."""
    fx = json.loads((FIXTURES / "wp14_tables_5_to_8.json").read_text(encoding="utf-8"))
    levels = fx["levels"]
    out = _wp14_runs("7", fx["scenarios_type_i"], fx["table_7_type_i"], levels, 1000)
    out.update(_wp14_runs("8", fx["scenarios_type_ii"], fx["table_8_type_ii"], levels, 2000))
    assert len(out) == 144
    failed = {k for k, v in out.items() if not v[0]}
    assert failed == WP14_NOT_REPRODUCED, {k: out[k] for k in failed ^ WP14_NOT_REPRODUCED}


def test_wp14_traffic_lights_tie_rule():
    """The printed colour mapping (a year on a boundary takes the better colour) does not give WP14's published
    rates; the upper colour does, exactly: with independent years the rejection probability is exact."""
    fx = json.loads((FIXTURES / "wp14_tables_5_to_8.json").read_text(encoding="utf-8"))
    for name in ("I_SC", "I_LC", "I_SV", "I_LV"):
        sc = fx["scenarios_type_i"][name]
        pf = np.array(sc["forecast_pct"]) / 100.0
        for level, pub in zip(fx["levels"], fx["table_7_type_i"][name]["traffic"], strict=True):
            upper = traffic_lights_rejection_probability([1000] * 5, pf, level, ties="upper")["probability"]
            assert agrees_with_published(Rate(round(upper * 1e6), 1_000_000), pub, 25_000)[0], (name, level, upper)
        lower = traffic_lights_rejection_probability([1000] * 5, pf, 0.1, ties="lower")["probability"]
        assert lower < 0.75 * fx["table_7_type_i"][name]["traffic"][0], (name, lower)


def test_yurdakul_naranjo_table4():
    """RV-309: 53 of the 54 cells of Table 4, the PSI's rule-of-thumb bands and the chi-square benchmark, with the
    suite's seed. The harness sits slightly above the paper on average (a mean z near +0.8, a spread near 1.5), so the
    count depends on the seed; the mean z is held below 1.2 so the documented offset cannot grow unnoticed."""
    fx = json.loads((FIXTURES / "yurdakul_naranjo_table4.json").read_text(encoding="utf-8"))
    rules = (metric_above(0.10), metric_above(0.25))
    labels = ("metric>0.1", "metric>0.25", "p<0.05")
    out = {}
    seed = 2000
    for row in fx["rows"]:
        for si, shift in enumerate((0.0, 0.25, 0.5)):
            seed += 1
            gen = TwoSample(n_base=row["n"], n_target=row["m"], shift=shift, sd=8.0, bins=10)
            sim = simulate("stability.psi", gen, n_rep=10_000, seed=seed, alphas=(0.05,), rules=rules)
            for label, pub in zip(labels, row["rates"][3 * si: 3 * si + 3], strict=True):
                ok, z = agrees_with_published(sim.rate(label), pub, 10_000)
                out[(row["m"], row["n"], shift, label)] = (ok, sim.rate(label).rate, pub, z)
    assert len(out) == 54
    failed = {k for k, v in out.items() if not v[0]}
    assert failed == YN_NOT_REPRODUCED, {k: out[k] for k in failed ^ YN_NOT_REPRODUCED}
    zs = [v[3] for v in out.values() if 0.0 < v[2] < 1.0]
    assert 0.0 < sum(zs) / len(zs) < 1.2
    # the paper's reading: PSI > 0.10 has too large a type I error unless both samples exceed 400
    assert out[(100, 100, 0.0, "metric>0.1")][1] > 0.8 and out[(400, 400, 0.0, "metric>0.1")][1] < 0.05


def test_delong_nested_and_non_nested():
    """RV-310: nested discriminants fitted and compared on the same sample reject almost never (Demler et al.: 0.001
    at 5%); two non-nested scores of equal AUC on an independent sample hold the size."""
    nested = simulate("disc.delong", NestedLda(), n_rep=1000, seed=31)
    assert nested.rate(0.05).rate < 0.01
    fair = simulate("disc.delong", BinormalScores(621, 7640, separation=1.2, challenger_separation=1.2),
                    n_rep=2000, seed=32)
    assert within_size(fair.rate(0.05), 0.05) and fair.rate(0.05).rate > 0.03
