"""U2: PD curve calibration (RV-207 to RV-210, RV-212, RV-213) against Tasche (2013), arXiv 1212.3716, read page by page
on 2026-10-05 (Contraste dossier 10, section 2). Every expected value below is a printed cell of the paper; every
computation starts from the S&P counts of its Table 2, never from a rounded intermediate."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.engines import pd_curve as pc
from riskvalidation.validation.calibration import pd_default_profile

# Table 2: obligors rated at the start of the year and defaults during it, best grade (AAA) first. The S&P counts are
# third-party data: they are read from the device data root (BL-049, decision D16-6), never committed. The file is
# <RISKVALIDATION_DATA>/tasche2013/table2.json, {"years": {"2009": {"obligors": [...], "defaults": [...]}, ...}};
# without it the tests that start from the counts are skipped with this reason.
_DATA = os.environ.get("RISKVALIDATION_DATA")
TABLE2 = Path(_DATA) / "tasche2013" / "table2.json" if _DATA else None
needs_table2 = pytest.mark.skipif(
    TABLE2 is None or not TABLE2.is_file(),
    reason="the S&P Table 2 counts are read from RISKVALIDATION_DATA/tasche2013/table2.json (BL-049)")


def _load_counts() -> dict[int, tuple[list[int], list[int]]]:
    if TABLE2 is None or not TABLE2.is_file():
        return {}
    years = json.loads(TABLE2.read_text(encoding="utf-8"))["years"]
    return {int(y): (v["obligors"], v["defaults"]) for y, v in years.items()}


SP = _load_counts()
# Table 2, the default rate columns (%)
T2_DR = {
    2009: [0.00, 0.00, 0.00, 0.00, 0.29, 0.39, 0.00, 0.40, 0.18, 1.09, 0.00, 1.02, 0.91, 5.48, 9.96, 17.16, 48.42],
    2010: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.79, 0.36, 0.53, 0.00, 0.69, 2.07, 22.27],
    2011: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.39, 1.19, 3.99, 15.94],
}
# Table 3: rating profiles at the start of each year (%)
T3 = {
    2009: [1.38, 0.63, 3.21, 4.18, 5.80, 8.70, 9.32, 8.50, 9.23, 7.83, 4.54, 5.03, 7.53, 7.47, 8.23, 5.17, 3.24],
    2010: [1.30, 0.45, 2.59, 3.78, 6.39, 8.58, 9.56, 8.28, 10.56, 7.79, 4.60, 5.00, 6.86, 7.12, 7.90, 5.25, 3.98],
    2011: [0.87, 0.62, 2.05, 3.54, 6.11, 8.04, 9.58, 8.09, 9.39, 8.69, 4.45, 5.46, 6.89, 8.71, 10.02, 5.15, 2.36],
}
# Table 5: the 2009 PD curve smoothed by quasi moment matching (%)
T5 = [0.003, 0.006, 0.012, 0.025, 0.047, 0.091, 0.173, 0.299, 0.495, 0.797, 1.138, 1.518, 2.280, 3.943, 7.999, 19.557,
      48.355]
# Table 6: the empirical and the implied 2009 default profiles (%)
T6_EMPIRICAL = [0.0000, 0.0000, 0.0000, 0.0000, 0.4274, 0.8547, 0.0000, 0.8547, 0.4274, 2.1368, 0.0000, 1.2821, 1.7094,
                10.2564, 20.5128, 22.2222, 39.3162]
T6_IMPLIED = [0.0010, 0.0009, 0.0094, 0.0261, 0.0685, 0.1989, 0.4041, 0.6355, 1.1436, 1.5642, 1.2934, 1.9143, 4.2968,
              7.3802, 16.4769, 25.3242, 39.2628]
# Table 7: forecasts of the 2010 and 2011 grade default rates (%) and the p-values (%) of the default-profile tests
T7 = {
    2010: {
        "invariant_default_profile": [0.0012, 0.0023, 0.0041, 0.0083, 0.0163, 0.0319, 0.0593, 0.0995, 0.1647, 0.2660,
                                      0.3706, 0.4847, 0.6907, 1.1043, 2.0554, 4.5380, 12.9712],
        "invariant_accuracy_ratio": [0.0004, 0.0009, 0.0018, 0.0040, 0.0086, 0.0183, 0.0366, 0.0652, 0.1145, 0.1955,
                                     0.2827, 0.3806, 0.5631, 0.9460, 1.8843, 4.5164, 14.5179],
        "scaled_pds": [0.0007, 0.0015, 0.0031, 0.0066, 0.0125, 0.0241, 0.0458, 0.0789, 0.1307, 0.2107, 0.3006, 0.4012,
                       0.6024, 1.0417, 2.1134, 5.1671, 12.7755],
        "scaled_likelihood_ratio": [0.0005, 0.0012, 0.0023, 0.0049, 0.0093, 0.0180, 0.0342, 0.0590, 0.0979, 0.1581,
                                    0.2263, 0.3029, 0.4576, 0.8023, 1.6844, 4.5716, 15.5760],
    },
    2011: {
        "invariant_default_profile": [0.0006, 0.0013, 0.0024, 0.0048, 0.0095, 0.0186, 0.0345, 0.0579, 0.0923, 0.1468,
                                      0.2065, 0.2694, 0.3828, 0.6291, 1.3483, 3.7460, 12.1942],
        "invariant_accuracy_ratio": [0.0003, 0.0006, 0.0013, 0.0027, 0.0058, 0.0120, 0.0236, 0.0416, 0.0691, 0.1147,
                                     0.1662, 0.2219, 0.3248, 0.5567, 1.2710, 3.7941, 13.3871],
        "scaled_pds": [0.0006, 0.0012, 0.0024, 0.0050, 0.0095, 0.0183, 0.0347, 0.0599, 0.0992, 0.1600, 0.2282, 0.3046,
                       0.4573, 0.7909, 1.6046, 3.9231, 9.6998],
        "scaled_likelihood_ratio": [0.0004, 0.0009, 0.0018, 0.0039, 0.0074, 0.0143, 0.0271, 0.0468, 0.0777, 0.1256,
                                    0.1797, 0.2405, 0.3635, 0.6378, 1.3414, 3.6627, 12.7721],
    },
}
T7_P = {
    2010: {"invariant_default_profile": 4.6, "invariant_accuracy_ratio": 8.0, "scaled_pds": 4.0,
           "scaled_likelihood_ratio": 11.3},
    2011: {"invariant_default_profile": 78.5, "invariant_accuracy_ratio": 89.4, "scaled_pds": 36.6,
           "scaled_likelihood_ratio": 82.2},
}
# Table 8: case 3 forecasts of the unconditional default rate (%) and the p-values of the profile fits
T8 = {
    2010: {"invariant_pd_curve": 4.32, "least_squares": 4.80, "least_chi2": 5.35, "invariant_likelihood_ratio": 5.38},
    2011: {"invariant_pd_curve": 3.75, "least_squares": 3.50, "least_chi2": 2.84, "invariant_likelihood_ratio": 2.79},
}
T8_P = {2010: {"least_squares": 0.0037, "least_chi2": 0.0051}, 2011: {"least_squares": 1e-10, "least_chi2": 1e-10}}
# Table 9: the smoothed likelihood ratios of each year (3.8), the paper's own QMM fits
T9 = {
    2009: [1501.55, 715.28, 353.69, 167.12, 88.18, 45.53, 23.98, 13.89, 8.37, 5.17, 3.61, 2.70, 1.78, 1.01, 0.48, 0.17,
           0.04],
    2010: [53720.74, 20477.36, 8583.13, 3099.52, 1161.49, 436.37, 177.13, 83.35, 40.00, 19.85, 12.23, 8.28, 4.93, 2.46,
           0.96, 0.27, 0.05],
    2011: [20221.19, 7734.58, 3410.45, 1365.00, 547.00, 227.20, 100.78, 51.03, 27.65, 15.00, 9.58, 6.74, 4.24, 2.19,
           0.79, 0.20, 0.04],
}


def smoothed(year: int) -> tuple[dict, dict]:
    """The empirical model of ``year`` and its QMM curve with the year's own targets (Table 4 for 2009)."""
    m = pc.model_from_counts(*SP[year])
    return m, pc.qmm(m["profile"], m["pd"], m["accuracy_ratio"], survival_profile=m["survival_profile"])


def calibrate(year: int) -> dict[str, list[float]]:
    """The four case 1 forecasts for ``year`` from the 2009 estimation model, at the observed ("prophetic") PD."""
    m0, q0 = smoothed(2009)
    m1 = pc.model_from_counts(*SP[year])
    pi1, p1 = m1["profile"], m1["pd"]
    return {
        "invariant_default_profile": pc.invariant_default_profile(m0["default_profile"], pi1, p1)["pd_curve"],
        "invariant_accuracy_ratio": pc.invariant_accuracy_ratio(pi1, p1, m0["accuracy_ratio"])["pd_curve"],
        "scaled_pds": pc.scaled_pds(q0["pd_curve"], pi1, p1)["pd_curve"],
        "scaled_likelihood_ratio": pc.scaled_likelihood_ratio(q0["pd_curve"], q0["pd"], pi1, p1)["pd_curve"],
    }


@needs_table2
def test_model_identities():
    """RV-207: the empirical models reproduce Tables 2 to 4, and the identities of section 3 hold."""
    for year, (n, d) in SP.items():
        m = pc.model_from_counts(n, d)
        assert np.allclose(np.round(100 * np.array(m["default_rate"]), 2), T2_DR[year], atol=1e-9), year
        assert np.allclose(np.round(100 * np.array(m["profile"]), 2), T3[year], atol=1e-9), year
        # the same model seen through its curve: (3.5a), (3.5b), (3.5c), and (A.11b) against (3.7)
        assert pc.unconditional_pd(m["default_rate"], m["profile"]) == pytest.approx(m["pd"], abs=1e-15)
        assert np.allclose(pc.implied_default_profile(m["default_rate"], m["profile"]), m["default_profile"],
                           atol=1e-15)
        assert np.allclose(pc.implied_survival_profile(m["default_rate"], m["profile"]), m["survival_profile"],
                           atol=1e-15)
        assert pc.accuracy_ratio_from_curve(m["default_rate"], m["profile"]) == pytest.approx(m["accuracy_ratio"],
                                                                                              abs=1e-12)
        assert pc.accuracy_ratio(m["default_profile"], m["survival_profile"]) == pytest.approx(m["accuracy_ratio"],
                                                                                             abs=1e-15)
    m09 = pc.model_from_counts(*SP[2009])
    assert round(100 * m09["pd"], 2) == 3.99 and round(100 * m09["accuracy_ratio"], 1) == 82.7  # Table 4

    _, q = smoothed(2009)
    c, pi, p = np.array(q["pd_curve"]), np.array(m09["profile"]), q["pd"]
    dp = np.array(pc.implied_default_profile(c, pi))
    sp = np.array(pc.implied_survival_profile(c, pi))
    lam = np.array(pc.likelihood_ratio(c, p))
    assert np.allclose(pi, p * dp + (1 - p) * sp, atol=1e-15)  # (3.4a)
    assert np.allclose(lam, sp / dp, rtol=1e-12)  # (3.8)
    assert np.allclose(pc.pd_curve_from_likelihood_ratio(lam, p), c, rtol=1e-12)  # (3.9a) inverts (3.9b)
    assert np.allclose(dp, pi / (p + (1 - p) * lam), rtol=1e-12)  # (3.10a)
    assert float(np.sum(pi / (p + (1 - p) * lam))) == pytest.approx(1.0, abs=1e-14)  # (3.10b)
    assert pc.pd_from_likelihood_ratio(pi, lam) == pytest.approx(p, rel=1e-12)  # theorem 3.3, the unique solution
    assert pc.accuracy_ratio(dp, sp) == pytest.approx(pc.accuracy_ratio_from_curve(c, pi), abs=1e-12)  # (3.7), (A.11b)

    # case 2, Elkan's change in base rate (4.11): the odds move by the ratio of the base-rate odds, grade by grade
    p1 = 0.02
    c1 = np.array(pc.invariant_likelihood_ratio(c, p, pd_1=p1)["pd_curve"])
    assert np.allclose(c1 / (1 - c1), c / (1 - c) * (p1 / (1 - p1)) / (p / (1 - p)), rtol=1e-12)
    assert np.allclose(pc.invariant_likelihood_ratio(c, p, pd_1=p)["pd_curve"], c, rtol=1e-12)
    # case 3, the invariant PD curve carries its curve and implies the forecast PD (4.13)
    pi10 = pc.model_from_counts(*SP[2010])["profile"]
    assert pc.invariant_pd_curve(c, pi10)["pd"] == pytest.approx(float(np.sum(c * np.array(pi10))), abs=1e-15)


@needs_table2
def test_qmm_table5():
    """RV-208: QMM from the 2009 counts and the targets of Table 4 reproduces Tables 5 and 6; the 2010 and 2011 fits
    reproduce the smoothed likelihood ratios of Table 9 up to the paper's own solver offset."""
    m, q = smoothed(2009)
    assert q["pd"] == pytest.approx(m["pd"], abs=1e-13)
    assert q["accuracy_ratio"] == pytest.approx(m["accuracy_ratio"], abs=1e-12)
    c = np.array(q["pd_curve"])
    assert np.all(np.diff(c) > 0), "the robust logistic curve is monotone (A.8)"
    ours = np.round(100 * c, 3)
    assert np.all(np.abs(ours - T5) <= 0.001 + 1e-9)  # within one unit of the printed third decimal
    assert int(np.sum(np.abs(ours - T5) < 1e-9)) == 15  # BB and CCC-C sit one unit off
    assert np.allclose(np.round(100 * np.array(m["default_profile"]), 4), T6_EMPIRICAL, atol=1e-9)
    implied = np.round(100 * np.array(pc.implied_default_profile(c, m["profile"])), 4)
    assert np.all(np.abs(implied - T6_IMPLIED) <= 1e-4 + 1e-4 * np.array(T6_IMPLIED))

    for year in (2009, 2010, 2011):
        _, qy = smoothed(year)
        lam = np.array(pc.likelihood_ratio(qy["pd_curve"], qy["pd"]))
        paper = np.array(T9[year])
        assert np.all(np.abs(np.round(lam, 2) - paper) <= 0.005 + 0.002 * paper), year
        # log lambda is alpha + beta * score + log(p / (1 - p)) exactly, so a different (alpha, beta) moves log lambda
        # by an affine function of the scores; on the cells printed with five or more digits the gap is exactly that
        keep = paper >= 100.0
        scores = np.array(qy["scores"])[keep]
        gap = np.log(paper[keep] / lam[keep])
        design = np.vstack([np.ones(keep.sum()), scores]).T
        coef, *_ = np.linalg.lstsq(design, gap, rcond=None)
        assert np.max(np.abs(gap - design @ coef)) < 2e-5, year
        assert np.all(np.abs(coef) < 1e-3), (year, coef)


@needs_table2
def test_table7_forecasts():
    """RV-209: the four case 1 approaches reproduce the 136 forecast cells of Table 7."""
    for year in (2010, 2011):
        ours = calibrate(year)
        for approach, printed in T7[year].items():
            ref = np.array(printed)
            got = np.round(100 * np.array(ours[approach]), 4)
            # one unit of the printed fourth decimal plus the relative offset the paper's 2009 curve carries (Table 9)
            assert np.all(np.abs(got - ref) <= 1e-4 + 1e-4 * ref + 1e-12), (year, approach, got - ref)


@needs_table2
def test_table7_p_values():
    """RV-212: the Monte Carlo default-profile tests reproduce the p-values of Table 7 and the 9% of section 3.3."""
    n_sim = 100_000

    def tolerance(p_pct: float, se_pct: float, printed_unit: float) -> float:
        # the paper does not state its number of simulations: allow three standard errors of a 2,000-draw run (the
        # default of R's chisq.test, an assumption), three of ours, and half a unit of the printed decimal
        return 3 * 100 * math.sqrt(p_pct / 100 * (1 - p_pct / 100) / 2000) + 3 * se_pct + printed_unit / 2

    n09, d09 = SP[2009]
    _, q = smoothed(2009)
    r = pd_default_profile(n09, d09, q["pd_curve"], n_sim=n_sim, seed=2013)
    p_pct, se_pct = 100 * r.p_value, 100 * r.extras["mc_standard_error"]
    assert abs(p_pct - 9.0) <= tolerance(p_pct, se_pct, 1.0)
    for year in (2010, 2011):
        n1, d1 = SP[year]
        for approach, curve in calibrate(year).items():
            r = pd_default_profile(n1, d1, curve, n_sim=n_sim, seed=2013)
            p_pct, se_pct = 100 * r.p_value, 100 * r.extras["mc_standard_error"]
            printed = T7_P[year][approach]
            assert abs(p_pct - printed) <= tolerance(p_pct, se_pct, 0.1), (year, approach, p_pct, printed)


@needs_table2
def test_case3_table8():
    """RV-213: the case 3 approaches reproduce the forecast default rates and the profile-fit p-values of Table 8."""
    m0, q0 = smoothed(2009)
    c0, pi0 = q0["pd_curve"], m0["profile"]
    dp0 = pc.implied_default_profile(c0, pi0)  # the smoothed conditional profiles of the estimation model
    sp0 = pc.implied_survival_profile(c0, pi0)
    for year in (2010, 2011):
        n1 = SP[year][0]
        pi1 = pc.model_from_counts(*SP[year])["profile"]
        got = {
            "invariant_pd_curve": pc.invariant_pd_curve(c0, pi1),
            "least_squares": pc.invariant_conditional_profiles(dp0, sp0, pi1, fit="least_squares"),
            "least_chi2": pc.invariant_conditional_profiles(dp0, sp0, pi1, fit="least_chi2"),
            "invariant_likelihood_ratio": pc.invariant_likelihood_ratio(c0, q0["pd"], profile_1=pi1),
        }
        for approach, printed in T8[year].items():
            # 2011 least chi-square: 2.834 here against 2.84 printed, one unit of the printed precision
            assert abs(round(100 * got[approach]["pd"], 2) - printed) <= 0.01 + 1e-9, (year, approach)
        for fit, printed in T8_P[year].items():
            test = pc.profile_chi2(n1, got[fit]["profile"])
            assert test["dof"] == 16
            if year == 2010:
                assert abs(round(test["p_value"], 4) - printed) <= 1e-12, (fit, test["p_value"])
            else:
                assert test["p_value"] < printed, (fit, test["p_value"])
        for fit in ("least_squares", "least_chi2"):
            r = got[fit]
            assert np.allclose(np.array(r["pd_curve"]) * np.array(r["profile"]), r["pd"] * np.array(dp0), rtol=1e-12)


@needs_table2
def test_improper_models_refused():
    """RV-210: every approach refuses inputs that give no proper model, and the scaled likelihood ratio never does."""
    m0, q0 = smoothed(2009)
    c0, p0 = q0["pd_curve"], q0["pd"]
    pi10 = pc.model_from_counts(*SP[2010])["profile"]
    with pytest.raises(pc.ImproperModel, match="c_PD"):
        pc.scaled_pds(c0, pi10, 0.5)  # lifts CCC-C above one (section 4.2.3)
    with pytest.raises(pc.ImproperModel, match="4.6a"):
        pc.invariant_default_profile(m0["default_profile"], pi10, 0.5)
    with pytest.raises(pc.ImproperModel, match="not attainable"):
        pc.qmm([0.5, 0.5], 0.1, 0.9)  # two grades at PD 10% allow an accuracy ratio of 0.556 at most
    with pytest.raises(pc.ImproperModel, match="theorem 3.3"):
        pc.pd_from_likelihood_ratio([0.5, 0.5], [0.5, 1.2])  # sum pi lambda = 0.85, (3.11b) fails
    best_only = [1.0] * 10 + [0.0] * 7  # a forecast portfolio with the ten best grades only
    with pytest.raises(pc.ImproperModel, match="theorem 3.3"):
        pc.invariant_likelihood_ratio(c0, p0, profile_1=best_only)
    with pytest.raises(ValueError, match="exactly one"):
        pc.invariant_likelihood_ratio(c0, p0, pd_1=0.01, profile_1=pi10)
    for p1 in (1e-4, 0.01, 0.5, 0.9, 0.999):  # section 4.2.4: a unique proper solution at every forecast PD
        r = pc.scaled_likelihood_ratio(c0, p0, pi10, p1)
        cur = np.array(r["pd_curve"])
        assert np.all((cur > 0) & (cur < 1))
        assert r["bracket"][0] < r["c_lr"] < r["bracket"][1]
        assert float(np.sum(cur * np.array(pi10))) == pytest.approx(p1, rel=1e-10)
        assert sum(r["default_profile"]) == pytest.approx(1.0, abs=1e-12)


def test_inputs_validated():
    with pytest.raises(ValueError):
        pc.model_from_counts([10, 10], [0, 0])  # no defaults
    with pytest.raises(ValueError):
        pc.model_from_counts([10, 10], [11, 0])
    with pytest.raises(ValueError):
        pc.qmm([0.5, 0.5], 0.1, 1.2)
    with pytest.raises(ValueError):
        pc.likelihood_ratio([0.0, 0.5], 0.1)
    with pytest.raises(ValueError):
        pc.scaled_pds([0.01, 0.02], [0.5, 0.3, 0.2], 0.01)
    with pytest.raises(ValueError):
        pc.invariant_conditional_profiles([0.2, 0.8], [0.6, 0.4], [0.5, 0.5], fit="median")
    with pytest.raises(ValueError):
        pd_default_profile([100, 100], [0, 0], [0.01, 0.02])


def test_table2_counts_are_not_tracked():
    """BL-049: no row of the S&P Table 2 obligor or default counts is committed in the tests or the package."""
    root = Path(__file__).resolve().parents[1]
    # the first rows of the three printed years, built from parts so that this file does not contain them
    needles = [", ".join(row.split()) for row in ("81 37 188 245", "72 25 143 209", "51 36 120 207")]
    hits = [str(p.relative_to(root)) for base in ("tests", "src") for p in (root / base).rglob("*.py")
            if any(n in p.read_text(encoding="utf-8", errors="replace") for n in needles)]
    assert hits == [], hits


def test_counts_tests_skip_without_the_data_root():
    """BL-049: without RISKVALIDATION_DATA the tests that start from the counts are skipped with the variable named
    in the reason, never failed and never silently passed."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "RISKVALIDATION_DATA"}
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", __file__, "-k",
                          "not table2_counts_are_not_tracked and not counts_tests_skip"],
                         capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("SKIPPED [1]") == 6 and "RISKVALIDATION_DATA" in out.stdout, out.stdout
