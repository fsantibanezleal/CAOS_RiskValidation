"""BS-14 (engine 1.1.0): the PD by grade under informative rating withdrawals, ``transitions.withdrawals``.

The defining properties are exact: the family PD(k) reproduces the three usual treatments at k = 1, 0 and infinity,
inverts, and recovers the true PD when the counts are the expected counts of a known informative-withdrawal process.
The hypothesis of the plan (coverage of the anchored interval against the existing treatments) is decided on the
known-truth generator at a size a test can afford; the full 200 repetitions are in
``scripts/withdrawal_bounds_evaluation.py`` and ``docs/transitions/06_withdrawal-bounds.md``."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.transitions.estimators import cohort
from riskvalidation.transitions.withdrawals import (
    odds_multiplier,
    tipping_k,
    withdrawal_bounds,
    withdrawal_pd,
    withdrawn_defaulters,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

# one grade's one-year cohort: 100 defaults, 700 stayers, 200 withdrawn (N = 1000)
D, S, W = 100.0, 700.0, 200.0


def _expected_counts(n: float, p: float, w_stay: float, k: float) -> tuple[float, float, float]:
    """The expected (D, S, W) of a cohort of n ratings with true PD p, withdrawal probability w_stay for the ratings
    that do not default and odds k times larger for those that do: the known-truth process."""
    odds = k * w_stay / (1.0 - w_stay)
    w_def = odds / (1.0 + odds)
    d = n * p * (1.0 - w_def)
    w = n * p * w_def + n * (1.0 - p) * w_stay
    return d, n - d - w, w


# ------------------------------------------------------------------------------------------ the family PD(k)
def test_the_family_reproduces_the_three_usual_treatments():
    """k = 1 is the industry estimate of ``cohort``, k = 0 the unfollowed one, k = infinity Manski's upper bound."""
    counts = [[800, 90, 10], [60, 640, 100], [0, 0, 0]]
    wd = [100, 200, 0]
    d = np.array([10.0, 100.0])
    s = np.array([890.0, 700.0])
    w = np.array([100.0, 200.0])
    ex = cohort(counts, wd)["matrix"][:2, 2]
    st = cohort(counts, wd, treatment="state")["matrix"][:2, 2]
    np.testing.assert_allclose(withdrawal_pd(d, s, w, 1.0), ex, rtol=1e-12)
    np.testing.assert_allclose(withdrawal_pd(d, s, w, 0.0), st, rtol=1e-12)
    np.testing.assert_allclose(withdrawal_pd(d, s, w, math.inf), (d + w) / (d + s + w), rtol=1e-12)
    np.testing.assert_allclose(withdrawal_pd(d, s, w, 1.0), d / (d + s), rtol=1e-12)


def test_the_closed_form_and_monotonicity():
    k = np.array([0.0, 0.5, 1.0, 2.0, 9.0, 1e6, math.inf])
    x = np.array([withdrawn_defaulters(D, S, W, kk)[0] for kk in k])
    np.testing.assert_allclose(x[:-2], W * k[:-2] * D / (k[:-2] * D + S), rtol=1e-12)
    assert x[-1] == W and x[0] == 0.0
    assert np.all(np.diff(x) >= 0)
    pd = np.array([withdrawal_pd(D, S, W, kk)[0] for kk in k])
    assert np.all(np.diff(pd) >= 0)
    assert pd[0] == pytest.approx(D / 1000.0) and pd[-1] == pytest.approx(300.0 / 1000.0)


def test_the_odds_of_the_withdrawn_defaulters_are_k_times_the_others():
    """The definition of k, checked from the counts: (X/D) / ((W - X)/S) = k."""
    for k in (0.3, 1.0, 4.0, 9.0, 25.0):
        x = withdrawn_defaulters(D, S, W, k)[0]
        assert (x / D) / ((W - x) / S) == pytest.approx(k, rel=1e-12)


def test_odds_multiplier_inverts_and_marks_the_unidentified():
    for k in (0.0, 0.2, 1.0, 3.0, 9.0, 40.0):
        x = withdrawn_defaulters(D, S, W, k)
        assert odds_multiplier(D, S, W, x)[0] == pytest.approx(k, abs=1e-9, rel=1e-9)
    assert odds_multiplier(D, S, W, W)[0] == math.inf
    assert np.isnan(odds_multiplier(0.0, S, W, 5.0)[0])  # no observed default: k undefined
    assert np.isnan(odds_multiplier(D, S, 0.0, 0.0)[0])  # nothing withdrawn: k undefined
    with pytest.raises(ValueError):
        odds_multiplier(D, S, W, W + 1.0)
    with pytest.raises(ValueError):
        odds_multiplier(D, S, W, -1.0)


# ----------------------------------------------------------------------------------- a known-truth process
@pytest.mark.parametrize("k", [1.0, 3.0, 9.0])
def test_expected_counts_of_a_known_process_recover_the_true_pd(k):
    """The counts a process with true PD p and odds ratio k produces, on average, give PD(k) = p exactly, whatever k
    is, while the industry estimate understates it as soon as k > 1."""
    n, p, w_stay = 100_000.0, 0.0117, 0.058
    d, s, w = _expected_counts(n, p, w_stay, k)
    assert withdrawal_pd(d, s, w, k)[0] == pytest.approx(p, rel=1e-12)
    ind = withdrawal_pd(d, s, w, 1.0)[0]
    assert ind == pytest.approx(p, rel=0.02) if k == 1.0 else ind < p * 0.95
    assert odds_multiplier(d, s, w, withdrawn_defaulters(d, s, w, k))[0] == pytest.approx(k, rel=1e-9)


@pytest.mark.parametrize("k", [1.0, 3.0, 9.0])
def test_the_anchor_recovers_k_and_covers_the_true_pd(k):
    """With a second source that counts the true defaults of the same cohort, the anchored k is the process's k, the
    anchored PD the true one, and the interval contains it (and k lies in [k_low, k_high])."""
    n, p, w_stay = 20_000.0, 0.0117, 0.058
    d, s, w = _expected_counts(n, p, w_stay, k)
    counts = [[s, d], [0.0, 0.0]]
    r = withdrawal_bounds(counts, [w, 0.0], anchor_defaults=[n * p], anchor_obligors=[n])
    a = r["anchor"]
    assert a["k"][0] == pytest.approx(k, rel=1e-6)
    assert a["pd"][0] == pytest.approx(p, rel=1e-9)
    assert a["lower"][0] <= p <= a["upper"][0]
    assert a["k_low"][0] <= k <= a["k_high"][0]
    assert a["flag"] == ["ok"]


# ---------------------------------------------------------------------------------------------------- tipping
def test_tipping_k_solves_pd_equals_threshold():
    t = 0.2  # PD(1) = 100/800 = 0.125 < 0.2 < 0.3 = (D + W)/N: reached at a k above 1
    k = tipping_k(D, S, W, t)[0]
    assert withdrawal_pd(D, S, W, k)[0] == pytest.approx(t, rel=1e-9)
    assert k > 1.0
    k_low = tipping_k(D, S, W, 0.11)[0]  # 0.11 < PD(1): reached at a k below 1
    assert withdrawal_pd(D, S, W, k_low)[0] == pytest.approx(0.11, rel=1e-9)
    assert k_low < 1.0
    # unreachable thresholds: below D/N or above (D+W)/N
    assert np.isnan(tipping_k(D, S, W, 0.05)[0])
    assert np.isnan(tipping_k(D, S, W, 0.5)[0])
    # exactly the bounds
    assert tipping_k(D, S, W, D / 1000.0)[0] == 0.0
    assert tipping_k(D, S, W, 0.3)[0] == math.inf
    with pytest.raises(ValueError):
        tipping_k(D, S, W, 1.0)


def test_the_tipping_status_of_a_decision():
    # grade 0: PD(1) = 0.125 below 0.2, upper bound 0.3 above it: flips up; grade 1 same counts, threshold 0.35: robust
    counts = [[S, D], [0.0, 0.0]]
    up = withdrawal_bounds(counts, [W, 0.0], thresholds=0.2)["tipping"]
    assert up["status"] == ["flips_up"] and up["decision_industry"] == ["below"]
    assert withdrawal_pd(D, S, W, up["k"][0])[0] == pytest.approx(0.2, rel=1e-9)
    down = withdrawal_bounds(counts, [W, 0.0], thresholds=0.11)["tipping"]
    assert down["status"] == ["flips_down"] and down["decision_industry"] == ["above"]
    assert withdrawal_bounds(counts, [W, 0.0], thresholds=0.35)["tipping"]["status"] == ["robust"]
    assert withdrawal_bounds(counts, [W, 0.0], thresholds=0.05)["tipping"]["status"] == ["robust"]


# ------------------------------------------------------------------------------------------ the result record
def test_withdrawal_bounds_record_and_the_identified_band():
    counts = [[800, 90, 10], [60, 640, 100], [0, 0, 0]]
    r = withdrawal_bounds(counts, [100, 200, 0], k_grid=(0, 1, 9, math.inf))
    assert r["grades"] == [0, 1]
    np.testing.assert_array_equal(r["defaults"], [10, 100])
    np.testing.assert_array_equal(r["survivors"], [890, 700])
    np.testing.assert_array_equal(r["n"], [1000, 1000])
    assert r["pd_by_k"].shape == (2, 4)
    np.testing.assert_allclose(r["pd_by_k"][:, 0], r["pd_unfollowed"])
    np.testing.assert_allclose(r["pd_by_k"][:, 1], r["pd_industry"])
    np.testing.assert_allclose(r["pd_by_k"][:, 3], r["pd_manski_upper"])
    assert np.all(r["identified"]["lower"] <= r["pd_unfollowed"])
    assert np.all(r["identified"]["upper"] >= r["pd_manski_upper"])
    assert r["anchor"] is None and r["tipping"] is None
    assert r["reason"] == ["ok", "ok"]
    assert "Manski" in r["reference"] and len(r["assumptions"]) >= 3


def test_withdrawal_bounds_is_deterministic():
    counts = [[800, 90, 10], [60, 640, 100], [0, 0, 0]]
    a = withdrawal_bounds(counts, [100, 200, 0], anchor_defaults=[20, 150], thresholds=0.15)
    b = withdrawal_bounds(counts, [100, 200, 0], anchor_defaults=[20, 150], thresholds=0.15)
    np.testing.assert_array_equal(a["anchor"]["pd"], b["anchor"]["pd"])
    np.testing.assert_array_equal(a["pd_by_k"], b["pd_by_k"])


def test_anchor_flags_and_clipping():
    counts = [[S, D], [0.0, 0.0]]
    low = withdrawal_bounds(counts, [W, 0.0], anchor_defaults=[50])["anchor"]  # fewer than the D already seen
    assert low["flag"] == ["anchor_below_observed"] and low["k"][0] == 0.0
    assert low["pd"][0] == pytest.approx(D / 1000.0)
    high = withdrawal_bounds(counts, [W, 0.0], anchor_defaults=[400])["anchor"]  # more than D + W
    assert high["flag"] == ["anchor_above_identified"] and high["k"][0] == math.inf
    assert high["pd"][0] == pytest.approx(0.3)
    assert high["lower"][0] <= high["pd"][0] <= high["upper"][0]
    k_mid = withdrawal_bounds(counts, [W, 0.0], anchor_defaults=[200], anchor_obligors=[1000])["anchor"]
    assert k_mid["k_low"][0] <= k_mid["k"][0] <= k_mid["k_high"][0]
    assert k_mid["lower"][0] <= k_mid["pd"][0] <= k_mid["upper"][0]
    # an anchor from another cohort: the same rate over a larger cohort is a narrower k band
    wide = withdrawal_bounds(counts, [W, 0.0], anchor_defaults=[200], anchor_obligors=[1000])["anchor"]
    narrow = withdrawal_bounds(counts, [W, 0.0], anchor_defaults=[2000], anchor_obligors=[10000])["anchor"]
    assert narrow["k_high"][0] - narrow["k_low"][0] < wide["k_high"][0] - wide["k_low"][0]


def test_reason_codes_for_degenerate_grades():
    # grade 0: no defaults; grade 1: no withdrawals; grade 2: empty; grade 3: normal
    counts = [[500, 0, 0, 0, 0], [0, 400, 0, 0, 10], [0, 0, 0, 0, 0], [0, 0, 0, 90, 10], [0, 0, 0, 0, 0]]
    r = withdrawal_bounds(counts, [100, 0, 0, 100, 0])
    assert r["reason"] == ["no_observed_defaults", "no_withdrawals", "no_cohort", "ok"]
    assert r["pd_by_k"][0, 1] == 0.0 and r["pd_manski_upper"][0] == pytest.approx(100 / 600)
    assert np.isnan(r["pd_by_k"][2]).all()
    assert r["pd_unfollowed"][1] == r["pd_industry"][1] == r["pd_manski_upper"][1] == pytest.approx(10 / 410)
    assert tipping_k(0.0, 500.0, 100.0, 0.1)[0] == math.inf  # a threshold reached only at Manski's bound
    assert np.isnan(tipping_k(0.0, 500.0, 100.0, 0.5)[0])


def test_missing_and_malformed_input_raises():
    ok = [[800, 90, 10], [60, 640, 100], [0, 0, 0]]
    with pytest.raises(ValueError, match="square"):
        withdrawal_bounds([[1, 2, 3], [4, 5, 6]], [0, 0])
    with pytest.raises(ValueError, match="withdrawn"):
        withdrawal_bounds(ok, [1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        withdrawal_bounds([[1, -1], [0, 0]], [0, 0])
    with pytest.raises(ValueError, match="level"):
        withdrawal_bounds(ok, [1, 2, 0], level=1.5)
    with pytest.raises(ValueError, match="k"):
        withdrawal_bounds(ok, [1, 2, 0], k_grid=(1.0, -2.0))
    with pytest.raises(ValueError, match="k"):
        withdrawal_pd(D, S, W, float("nan"))
    with pytest.raises(ValueError, match="anchor"):
        withdrawal_bounds(ok, [1, 2, 0], anchor_defaults=[1.0])
    with pytest.raises(ValueError, match="anchor"):
        withdrawal_bounds(ok, [1, 2, 0], anchor_defaults=[5000, 1], anchor_obligors=[10, 10])
    with pytest.raises(ValueError, match="finite"):
        withdrawal_pd(float("nan"), S, W, 1.0)


# --------------------------------------------------------------------------------- the hypothesis, in a test
def _script():
    path = ROOT / "scripts" / "withdrawal_bounds_evaluation.py"
    spec = importlib.util.spec_from_file_location("withdrawal_bounds_evaluation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_hypothesis_on_the_known_truth_generator_at_k_9():
    """BS-14 hypothesis (i), first-year cohorts of the S&P-like generator with withdrawal ten times as likely in the
    year before default (informative k = 9), 60 repetitions (the full 200 in the script, the measured values in
    docs/transitions/06_withdrawal-bounds.md): the anchored interval covers the true PD of grade B in at least 85% of
    them (90% in the plan, less the Monte Carlo error at 60) and beats the industry treatment. The plan's second
    clause (the existing treatments under 50%) depends on the cohort size: it is decided at 10 times the cohort of
    the case in the next test and reported at the case's size in the docs, whichever way it falls."""
    ev = _script()
    res = ev.coverage_study(informative=9.0, reps=60, seed=20261028)
    v = ev.verdict(res, "B")
    assert res["reps"] == 60 and v["reps"] == 60
    assert v["anchored_coverage"] >= 0.85, v
    assert v["anchored_better_than_industry"], v
    # the verdict is read off the recorded numbers, not asserted: its booleans agree with them
    assert v["industry_and_unfollowed_below_50"] == (v["industry_coverage"] < 0.5 and v["unfollowed_coverage"] < 0.5)


def test_the_existing_treatments_fail_where_the_bias_exceeds_the_sampling_noise():
    """With 10 times the cohort (so the 15 to 38% understatement is larger than the Jeffreys interval) the industry and
    unfollowed treatments cover grade B's true PD in under 50% of 60 repetitions at k = 9; the anchored one still
    covers it in at least 85%."""
    ev = _script()
    res = ev.coverage_study(informative=9.0, reps=60, seed=20261028, scale=10)
    v = ev.verdict(res, "B")
    assert v["industry_and_unfollowed_below_50"], v
    assert v["anchored_coverage"] >= 0.85, v


def test_the_real_data_check_reports_per_cohort_k_and_its_stability():
    """Hypothesis (ii) is measured on the S&P cohorts (default page against transition page): the record has the
    anchored k per cohort for CCC-C, the pooled k, and the pre-stated stability criterion; the numbers themselves are
    reported in docs/transitions/06_withdrawal-bounds.md whatever they are."""
    ev = _script()
    real = ev.real_data_study(json.loads((FIXTURES / "withdrawals_sp_cerep.json").read_text(encoding="utf-8")))
    c = real["grades"]["CCC-C"]
    assert len(c["cohorts"]) >= 20
    assert c["pooled"]["k"] > 0
    assert 0.0 <= c["stability"]["share_cohorts_containing_pooled_k"] <= 1.0
    assert real["criterion"]
