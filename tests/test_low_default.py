"""U2: most prudent estimation (RV-206) against Pluto and Tasche's published tables (dossier 04 C.3)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.engines.low_default import (
    most_prudent_pd,
    most_prudent_pd_multiperiod,
    most_prudent_pd_scaled,
)

GAMMAS = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
N = (100, 400, 300)  # grades A, B, C (best first)
# percent, to the printed two decimals (Pluto and Tasche 2005, Tables 1 to 4 and 7)
ZERO_INDEPENDENT = {
    "A": (0.09, 0.17, 0.29, 0.37, 0.57, 0.86),
    "B": (0.10, 0.20, 0.33, 0.43, 0.66, 0.98),
    "C": (0.23, 0.46, 0.76, 0.99, 1.52, 2.28),
}
# defaults 0, 2 and 1 in grades A, B and C (Tables 4, 5 and 6, read from the paper on 2026-10-05)
DEFAULTS_021 = {
    "A": (0.46, 0.65, 0.83, 0.97, 1.25, 1.62),
    "B": (0.52, 0.73, 0.95, 1.10, 1.43, 1.85),
    "C": (0.56, 0.90, 1.29, 1.57, 2.19, 3.04),
}
#: Table 4 prints 0.65% at gamma 75%; the exact Clopper-Pearson bound the paper computes (its footnote 2) is 0.638%,
#: since P(Binomial(800, 0.65%) <= 3) = 0.237, not 0.25: 0.65% is the bound at gamma 76.3%. Not a typesetting
#: slip: Table 11 scales with the same 0.65% (test_scaling_tables).
MISPRINT = ("A", 0.75, 0.64)
ZERO_RHO12 = {
    "A": (0.15, 0.40, 0.86, 1.31, 2.65, 5.29),
    "B": (0.17, 0.45, 0.96, 1.45, 2.92, 5.77),
    "C": (0.37, 0.92, 1.89, 2.78, 5.30, 9.84),
}


def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def test_pluto_tasche_tables():
    """RV-206: the closed-form bounds to the printed two decimals (Tables 1 to 6, one misprint in Table 4); the
    correlated ones, whose one-factor integral is computed numerically, within one unit of it (Table 7: three of the
    eighteen sit at the rounding boundary, 0.0063 at most)."""
    exact = 0
    for k, g in enumerate(GAMMAS):
        ind = most_prudent_pd(N, (0, 0, 0), g)["pd"]
        corr = most_prudent_pd(N, (0, 0, 0), g, rho=0.12)["pd"]
        for i, grade in enumerate("ABC"):
            assert _pct(ind[i]) == ZERO_INDEPENDENT[grade][k], (grade, g, ind[i])
            assert abs(100.0 * corr[i] - ZERO_RHO12[grade][k]) <= 0.01, (grade, g, corr[i])
            exact += _pct(corr[i]) == ZERO_RHO12[grade][k]
        with_defaults = most_prudent_pd(N, (0, 2, 1), g)["pd"]
        for i, grade in enumerate("ABC"):
            if (grade, g) == MISPRINT[:2]:
                assert _pct(with_defaults[i]) == MISPRINT[2]
                continue
            assert _pct(with_defaults[i]) == DEFAULTS_021[grade][k], (grade, g, with_defaults[i])
    assert exact >= 15
    res = most_prudent_pd(N, (0, 2, 1), 0.9)
    assert res["pooled_obligors"] == [800, 700, 300] and res["pooled_defaults"] == [3, 3, 1]
    assert res["pd"][0] <= res["pd"][1] <= res["pd"][2]  # monotone by construction


# Table 8 (defaults 0, 2, 1 with rho = 12%), Tables 9 to 12 (section 5, scaling) and Tables 13 and 14 (section 6, five
# years, rho = 12%, theta = 0.3), read from arXiv cond-mat/0411699v3 on 2026-10-05; percent, K unitless
DEFAULTS_021_RHO12 = {
    "A": (0.72, 1.42, 2.50, 3.42, 5.88, 10.08),
    "B": (0.81, 1.59, 2.77, 3.77, 6.43, 10.92),
    "C": (0.84, 1.76, 3.19, 4.41, 7.68, 13.14),
}
SCALED = {  # table: (rho, target, K, scaled A, B, C)
    9: (0.0, "central_tendency", (0.71, 0.48, 0.35, 0.30, 0.22, 0.17),
        ((0.33, 0.31, 0.29, 0.29, 0.28, 0.27), (0.37, 0.35, 0.34, 0.33, 0.32, 0.31),
         (0.40, 0.43, 0.46, 0.47, 0.49, 0.50))),
    10: (0.12, "central_tendency", (0.46, 0.23, 0.13, 0.09, 0.05, 0.03),
         ((0.33, 0.33, 0.32, 0.32, 0.32, 0.32), (0.38, 0.37, 0.36, 0.36, 0.35, 0.35),
          (0.39, 0.40, 0.41, 0.42, 0.42, 0.42))),
    11: (0.0, "upper_bound", (0.87, 0.83, 0.78, 0.77, 0.74, 0.71),
         ((0.40, 0.54, 0.65, 0.74, 0.92, 1.16), (0.45, 0.61, 0.74, 0.84, 1.06, 1.32),
          (0.49, 0.75, 1.01, 1.22, 1.62, 2.17))),
    12: (0.12, "upper_bound", (0.89, 0.87, 0.86, 0.86, 0.86, 0.87),
         ((0.64, 1.24, 2.16, 2.95, 5.06, 8.72), (0.72, 1.38, 2.39, 3.25, 5.54, 9.54),
          (0.75, 1.53, 2.76, 3.80, 6.61, 11.37))),
}
#: Table 12 prints 9.54% for grade B at 99.9%; the exact value is 9.44%, and the paper's own K (0.87) times its own
#: Table 8 bound (10.92%) is 9.50%: neither gives 9.54, a misprint
SCALED_MISPRINT = (12, "B", 0.999)
MULTI = {  # table: (defaults over five years, bounds A, B, C)
    13: ((0, 0, 0), ((0.03, 0.06, 0.11, 0.16, 0.30, 0.55), (0.03, 0.07, 0.13, 0.18, 0.33, 0.62),
                     (0.07, 0.14, 0.26, 0.37, 0.67, 1.23))),
    14: ((0, 2, 1), ((0.12, 0.21, 0.33, 0.43, 0.70, 1.17), (0.14, 0.24, 0.38, 0.49, 0.77, 1.29),
                     (0.15, 0.27, 0.46, 0.61, 1.01, 1.70))),
}


def test_correlated_with_defaults_table8():
    """RV-206: Table 8 solves (4.3a), the one-factor integral with the defaults; every printed cell lies at or above the
    exact bound and within one unit of the printed precision of it (6 of 18 identical), and an adaptive quadrature
    confirms the 256-node rule where the gap is largest."""
    from scipy import integrate, optimize, stats
    from scipy.special import ndtr, ndtri

    identical = 0
    for k, g in enumerate(GAMMAS):
        res = most_prudent_pd(N, (0, 2, 1), g, rho=0.12)["pd"]
        for i, grade in enumerate("ABC"):
            printed, ours = DEFAULTS_021_RHO12[grade][k], 100.0 * res[i]
            assert ours <= printed + 1e-9 and printed - ours <= 0.01, (grade, g, ours, printed)
            identical += _pct(res[i]) == printed
    assert identical == 6

    def tail(p: float) -> float:  # grade A at 50%: 800 pooled obligors, 3 pooled defaults
        def f(y: float) -> float:
            return stats.norm.pdf(y) * stats.binom.cdf(3, 800, ndtr((ndtri(p) - 0.12**0.5 * y) / 0.88**0.5))

        return integrate.quad(f, -12.0, 12.0, epsabs=1e-14, epsrel=1e-12, limit=500)[0]

    exact = optimize.brentq(lambda p: tail(p) - 0.5, 1e-6, 0.5, xtol=1e-15)  # 0.7106%; Table 8 prints 0.72%
    assert most_prudent_pd(N, (0, 2, 1), 0.5, rho=0.12)["pd"][0] == pytest.approx(exact, rel=1e-9)


def test_scaling_tables():
    """RV-215: section 5, scaling to the central tendency (Tables 9, 10) and to the portfolio's upper bound (Tables 11,
    12): every K within one unit of its print, every scaled bound within two units, except one misprint; Table 9, whose
    inputs the paper computed exactly, is identical in 17 of its 18 bounds."""
    for table, (rho, target, k_printed, cells) in SCALED.items():
        identical = 0
        for j, g in enumerate(GAMMAS):
            res = most_prudent_pd_scaled(N, (0, 2, 1), g, rho, target=target)
            assert abs(round(res["k"], 2) - k_printed[j]) <= 0.01 + 1e-9, (table, g, res["k"])
            # K times the unscaled bounds hits the target exactly (5.1)
            assert float(np.dot(res["pd"], N)) / sum(N) == pytest.approx(res["target"], rel=1e-12)
            for i, grade in enumerate("ABC"):
                printed, ours = cells[i][j], _pct(res["pd"][i])
                if (table, grade, g) == SCALED_MISPRINT:
                    assert ours == 9.44
                    continue
                assert abs(ours - printed) <= 0.02 + 1e-9, (table, grade, g, ours, printed)
                identical += ours == printed
        if table == 9:
            assert identical == 17
    central = most_prudent_pd_scaled(N, (0, 2, 1), 0.9, target="central_tendency")
    assert central["target"] == pytest.approx(3 / 800)
    with pytest.raises(ValueError, match="at least one default"):
        most_prudent_pd_scaled(N, (0, 0, 0), 0.9, target="central_tendency")


def test_multiperiod():
    """RV-216: section 6, five years with intertemporally correlated factors. The Monte Carlo bound agrees with a
    deterministic Gauss-Hermite product rule within 0.5%; the printed Tables 13 and 14 lie above the model's bounds
    (3% to 5% at the 99% and 99.9% levels, more at the low levels, where the print has one significant digit) and never
    below them beyond rounding."""
    import itertools

    from scipy import optimize, stats
    from scipy.special import ndtr, ndtri

    years, rho, theta = 5, 0.12, 0.3
    for table, (defaults, cells) in MULTI.items():
        for j, g in enumerate(GAMMAS):
            res = most_prudent_pd_multiperiod(N, defaults, g, rho=rho, theta=theta, years=years, n_sim=100_000, seed=5)
            for i in range(3):
                ours, printed = 100.0 * res["pd"][i], cells[i][j]
                assert ours <= printed + 0.005, (table, i, g, ours, printed)
                if g >= 0.99:
                    assert ours >= 0.94 * printed, (table, i, g, ours, printed)

    # the deterministic check: a 12-node probabilists' Gauss-Hermite rule in each of the five factor dimensions
    nodes, weights = np.polynomial.hermite_e.hermegauss(12)
    weights = weights / np.sqrt(2.0 * np.pi)
    grid = np.array(list(itertools.product(nodes, repeat=years)))
    w = np.prod(np.array(list(itertools.product(weights, repeat=years))), axis=1)
    t = np.arange(years)
    factors = grid @ np.linalg.cholesky(theta ** np.abs(np.subtract.outer(t, t)).astype(float)).T

    def bound(n: int, k: int, g: float) -> float:
        def f(p: float) -> float:
            cond = ndtr((ndtri(p) - rho**0.5 * factors) / (1.0 - rho) ** 0.5)
            return float(np.sum(w * stats.binom.cdf(k, n, 1.0 - np.prod(1.0 - cond, axis=1)))) - (1.0 - g)

        return optimize.brentq(f, 1e-9, 0.5, xtol=1e-14)

    for n, k, g, grade, defaults in ((800, 0, 0.999, 0, (0, 0, 0)), (300, 1, 0.95, 2, (0, 2, 1))):
        mc = most_prudent_pd_multiperiod(N, defaults, g, rho=rho, theta=theta, years=years, n_sim=100_000, seed=5)
        assert mc["pd"][grade] == pytest.approx(bound(n, k, g), rel=0.005)
