"""RV-104, RV-105, RV-106, RV-114: discrimination statistics against brute-force computations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation import Light
from riskvalidation.validation import (
    bootstrap_ci,
    cap_accuracy_ratio,
    disc_auc,
    disc_auc_vs_initial,
    disc_delong,
)


def _sample(seed=3, n=300, ties=True):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.15).astype(int)
    score = rng.normal(size=n) + 0.9 * y
    if ties:
        score = np.round(score, 1)  # create ties, as rating grades do
    return score, y


def _components_bruteforce(score, y):
    a, b = score[y == 1], score[y == 0]
    u = np.where(a[:, None] > b[None, :], 1.0, np.where(a[:, None] == b[None, :], 0.5, 0.0))
    return u.mean(axis=1), u.mean(axis=0), u


def test_auc_matches_pairwise_and_cap():
    s, y = _sample()
    v10, v01, u = _components_bruteforce(s, y)
    r = disc_auc(s, y)
    assert r.metric == pytest.approx(u.mean(), abs=1e-12)
    s2 = np.var(v10, ddof=1) / v10.size + np.var(v01, ddof=1) / v01.size
    assert r.extras["s2"] == pytest.approx(s2, rel=1e-12)
    assert r.extras["gini"] == pytest.approx(2 * u.mean() - 1)
    # AR from the CAP equals 2 AUC - 1 with consistent tie handling (BCBS WP14)
    assert cap_accuracy_ratio(s, y) == pytest.approx(2 * r.metric - 1, abs=1e-12)
    # equivalent to SciPy's Mann-Whitney U
    mw = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert r.metric == pytest.approx(mw / (v10.size * v01.size), abs=1e-12)


def test_auc_vs_initial_ecb_statistic():
    s, y = _sample(seed=5)
    cur = disc_auc(s, y)
    r = disc_auc_vs_initial(0.80, s, y)
    stat = (0.80 - cur.metric) / math.sqrt(cur.extras["s2"])
    assert r.statistic == pytest.approx(stat)
    assert r.p_value == pytest.approx(1 - stats.norm.cdf(stat))
    assert r.extras["auc_current"] == pytest.approx(cur.metric)


def test_delong_matches_bruteforce():
    s1, y = _sample(seed=9, ties=False)
    rng = np.random.default_rng(1)
    s2 = s1 + rng.normal(scale=0.8, size=s1.size)
    c1 = _components_bruteforce(s1, y)
    c2 = _components_bruteforce(s2, y)
    na, nb = c1[0].size, c1[1].size
    s10 = np.cov(np.vstack([c1[0], c2[0]]), ddof=1)
    s01 = np.cov(np.vstack([c1[1], c2[1]]), ddof=1)
    S = s10 / na + s01 / nb
    z = (c1[0].mean() - c2[0].mean()) / math.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    r = disc_delong(s1, s2, y)
    assert r.statistic == pytest.approx(z, rel=1e-10)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
    same = disc_delong(s1, s1, y)
    assert same.light is Light.NOT_EVALUATED


def test_bootstrap_ci_seeded():
    s, y = _sample(seed=13)

    def auc(sc, yy):
        return disc_auc(sc, yy).metric

    a = bootstrap_ci(auc, s, y, n_boot=200, seed=42)
    b = bootstrap_ci(auc, s, y, n_boot=200, seed=42)
    c = bootstrap_ci(auc, s, y, n_boot=200, seed=43)
    assert a == b and a != c
    assert a[0] < disc_auc(s, y).metric < a[1]
    # stratified: every replicate keeps the observed default count, so the default rate never varies
    lo, hi = bootstrap_ci(lambda sc, yy: float(yy.mean()), s, y, n_boot=100, seed=0)
    assert lo == hi == pytest.approx(y.mean())


def test_auc_vs_initial_with_the_development_variance():
    """The extension: with the development AUC's variance the statistic divides by sqrt(s^2 + s_init^2)."""
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(80, dtype=int), np.zeros(920, dtype=int)]
    risk = np.r_[rng.normal(1.0, 1.0, 80), rng.normal(0.0, 1.0, 920)]
    ecb = disc_auc_vs_initial(0.78, risk, y)
    ext = disc_auc_vs_initial(0.78, risk, y, auc_initial_variance=ecb.extras["s2"])
    assert ext.statistic == pytest.approx(ecb.statistic / math.sqrt(2.0))
    assert ext.extras["s2_total"] == pytest.approx(2.0 * ecb.extras["s2"]) and "extension" in ext.notes
    assert ecb.notes == "" and ecb.extras["auc_initial_variance"] == 0.0
    with pytest.raises(ValueError):
        disc_auc_vs_initial(0.78, risk, y, auc_initial_variance=-1.0)
