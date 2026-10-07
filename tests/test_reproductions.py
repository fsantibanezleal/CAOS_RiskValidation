"""RV-308 to RV-310: published simulation studies reproduced by the harness, cell by cell.

Each published rate is itself a simulation estimate, so a cell agrees when the difference is within 3.29 standard
errors of the difference at the pooled rate (both Monte Carlo errors). The cells that do not reproduce are listed with
the evidence; the suite fails if any other cell disagrees, or if a listed one starts to agree (the documentation would
then be wrong).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------------------------------------------------
# Israel, Rosenthal and Wei (2001): the embedding problem's three published matrices. The matrices are S&P and Moody's
# data reprinted in the paper, so the paper is read from the device data root (RISKVALIDATION_DATA) and never
# committed; without it the test is skipped.

_DATA = os.environ.get("RISKVALIDATION_DATA")
IRW_PDF = Path(_DATA) / "israel-rosenthal-wei-2001" / "wei.pdf" if _DATA else None


def _irw_text() -> str:
    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(IRW_PDF)).pages)
    return text.replace(" .", ".").replace("\u2212", "-")


def _irw_matrix(text: str, anchor: str, n: int = 8) -> np.ndarray:
    import re

    seg = text[text.index(anchor):]
    nums = [float(x) for x in re.findall(r"-?\d\.\d{3,6}", seg)[: n * n]]
    return np.array(nums).reshape(n, n)


@pytest.mark.skipif(IRW_PDF is None or not IRW_PDF.is_file(), reason="the paper is read from RISKVALIDATION_DATA")
def test_israel_rosenthal_wei():
    """RV-403: the nine L1 distances of section 4 and the first matrix's series generator. Eight distances follow from
    the printed matrices by equations (1), (2), (2') and (3); the first matrix's JLT distance follows from the printed
    four-digit Q_JLT (which is not equation (3) on the printed P: dossier 13, F-IRW-JLT)."""
    from riskvalidation.transitions import embedding as emb

    text = _irw_text()
    mats = {
        "jlt": _irw_matrix(text, "their Table 3"),
        "moodys": _irw_matrix(text, "we obtain the\nfollowing transition matrix"),
        "sp1999": _irw_matrix(text, "we obtain the following annual transition matrix"),
    }
    for p in mats.values():
        # the print: rows summing to 1 up to round-off, default absorbing
        assert np.allclose(p.sum(axis=1), 1.0, atol=3e-4) and p[7, 7] == 1.0
    published = {"jlt": (0.116900, 0.002736, 0.002686), "moodys": (0.100047, 0.001401, 0.001371),
                 "sp1999": (0.103516, 0.001096, 0.001088)}
    for name, p in mats.items():
        diag = emb.embedding_diagnostics(p)
        assert diag["theorem3_c"], name  # "By Theorem 3(c) ... does not have an exact generator"
        assert round(emb.generator(p, method="diagonal")["l1_distance"], 6) == published[name][1], name
        assert round(emb.generator(p, method="weighted")["l1_distance"], 6) == published[name][2], name
        if name != "jlt":
            assert round(emb.l1_distance(p, emb.jlt_generator(p)), 6) == published[name][0], name
    q_jlt_printed = _irw_matrix(text, "their approximate generator is")
    assert round(emb.l1_distance(mats["jlt"], q_jlt_printed), 6) == 0.116900
    assert round(emb.l1_distance(mats["jlt"], emb.jlt_generator(mats["jlt"])), 6) == 0.116477  # equation (3) itself
    series = emb.log_series(mats["jlt"])["generator"]
    printed_series = _irw_matrix(text, "Retaining six digits, we obtain")
    assert np.max(np.abs(series - printed_series)) < 6e-7
    assert emb.monotonicity_violations(mats["jlt"])[0]["tail"] == pytest.approx(0.9949)



def test_schuermann_hanson_table5():
    """RV-409: Schuermann and Hanson (2004), Table 5, BB in 2002 by the cohort method, 531 obligors: the Wald and
    Agresti-Coull intervals in basis points, independent and with a default correlation of 1% and 2% (3.4). The count
    of defaults is not printed; 15 is the one count that gives every printed bound."""
    from riskvalidation.transitions.intervals import pd_agresti_coull, pd_wald

    printed = {  # rho: (N-dagger, Wald (lower, upper, length), Agresti-Coull (lower, upper, length))
        0.0: (531, (141.56, 423.41, 281.84), (168.03, 464.71, 296.68)),
        0.01: (84.3, (0.00, 636.20, 636.20), (38.25, 938.00, 899.75)),
        0.02: (45.8, (0.00, 762.45, 762.45), (0.00, 1332.56, 1332.56)),
    }
    for rho, (n_dagger, wald, ac) in printed.items():
        w = pd_wald(15, 531, rho=rho)
        a = pd_agresti_coull(15, 531, rho=rho)
        assert round(w["n_effective"], 1) == n_dagger
        assert tuple(round(1e4 * w[k], 2) for k in ("lower", "upper", "length")) == wald, rho
        assert tuple(round(1e4 * a[k], 2) for k in ("lower", "upper", "length")) == ac, rho
    for d in (14, 16):
        assert round(1e4 * pd_wald(d, 531)["lower"], 2) != 141.56


# ---------------------------------------------------------------------------------------------------------------------
# Engelmann (2024), section 4: the TTC portfolio of the Trueck and Rachev (2009) matrix (11), reprinted in the paper
# from a rating agency's data; read from the data root like Israel et al., skipped without it.

ENGELMANN_PDF = Path(_DATA) / "engelmann-2024" / "2401.08892v1.pdf" if _DATA else None


@pytest.mark.skipif(ENGELMANN_PDF is None or not ENGELMANN_PDF.is_file(),
                    reason="the paper is read from RISKVALIDATION_DATA")
def test_engelmann_ttc_portfolio():
    """RV-411: section 4's TTC portfolio, its PD of 1.198%, the PDs of the three starting portfolios, the minimum
    projected PD of 0.722% and the maximum of 2.14% over 50 years under the unstressed matrix."""
    import re

    from pypdf import PdfReader

    from riskvalidation.transitions import ttc

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(ENGELMANN_PDF)).pages).replace(" .", ".")
    t = np.array([float(x) for x in re.findall(r"\d\.\d{4}", text[text.index("page 3, is used"):])[:64]])
    t = t.reshape(8, 8)
    assert np.allclose(t.sum(axis=1), 1.0, atol=1e-4) and t[7, 7] == 1.0  # the print rounds to four digits
    o = [0.00, 0.20, 0.30, 0.30, 0.20, 0.00, 0.00, 0.00]
    r = ttc.ttc_portfolio(t, o)
    assert tuple(np.round(r["portfolio"], 4)) == (0.0183, 0.1423, 0.3379, 0.2633, 0.1321, 0.0911, 0.0150, 0.0)
    assert round(100 * r["default_rate"], 3) == 1.198
    # (start, its printed PD in %, the printed extreme of its projection), each printed value with its decimals
    starts = {
        "W_init": ([0.00, 0.00, 0.20, 0.40, 0.30, 0.10, 0.00, 0.00], (1.161, 3), None),
        "W_init 2": ([0.70, 0.00, 0.00, 0.00, 0.00, 0.25, 0.05, 0.00], (2.725, 3), ("min", 0.722, 3)),
        "W tilde": ([0.01, 0.02, 0.10, 0.30, 0.41, 0.15, 0.01, 0.00], (1.83, 2), ("max", 2.14, 2)),
    }

    def prints_as(value: float, printed: float, decimals: int) -> bool:
        return abs(value - printed) <= 0.5 * 10.0**-decimals + 1e-9  # rounding half up, as 2.7245 prints 2.725

    for name, (w0, (pd0, dec0), extreme) in starts.items():
        rates = 100 * ttc.project(t, w0, o, 50)["default_rate"]
        assert prints_as(rates[0], pd0, dec0), (name, rates[0])
        if extreme:
            kind, printed, dec = extreme
            assert prints_as(rates.min() if kind == "min" else rates.max(), printed, dec), name
