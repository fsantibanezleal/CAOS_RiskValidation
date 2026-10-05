"""U2: most prudent estimation (RV-206) against Pluto and Tasche's published tables (dossier 04 C.3)."""

from __future__ import annotations

from riskvalidation.engines.low_default import most_prudent_pd

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
#: since P(Binomial(800, 0.65%) <= 3) = 0.237, not 0.25: 0.65% is the bound at gamma 76.3%. A misprint.
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
