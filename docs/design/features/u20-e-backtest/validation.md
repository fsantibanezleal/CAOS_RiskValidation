# U20 E-backtests of VaR and expected shortfall, validation

Oracles, by the order of strength (SDD section 6):

| Requirement | Oracle | Independent of the code under test by |
|---|---|---|
| RV-2101, RV-2102 | the paper's definitions on values computed by hand; the conventions 0/0 = 1 and 1/0 = infinity of Section 3.3 | arithmetic on written numbers |
| RV-2103 | quadrature of `E[(X - z)+] / (alpha (r - z))` for a standard normal X (equal to 1 at the true VaR and ES: equality in Theorem 1); the normal tail probability for the VaR; a Monte Carlo of 200,000 draws (20,000,000 under `sizepower`) | `scipy.integrate.quad`, `scipy.stats.norm` |
| RV-2104 | exact fractions of a five-day hand computation of equations (7) and (15) | `fractions.Fraction` |
| RV-2105 | loops over days and over each day's past sample written from equations (13) to (15), for VaR and ES, with moving forecasts, with and without a window | scalar Python, no shared code |
| RV-2106 | `wwz_log_e` of the Contraste manuscript `e-process-sequential-monitors`, copied into the test with its source; the closed form of Appendix A | the manuscript's own implementation, written before this unit |
| RV-2107 | the Kelly fraction `(h - alpha) / (1 - alpha)` of a Bernoulli bet; `scipy.optimize.minimize_scalar` (bounded) on the empirical growth rate | a closed form and another optimiser |
| RV-2108, RV-2109 | the definition of GREM as an average of two wealths and its betting process (Section 5.1); a loop over the last `window` days; a hand computation of the swing series of Remark 8 | the paper's identities |
| RV-2110 | the sum over every hit sequence of probability times `M_T` (2^10 sequences, 2^8 for the exact solver) and over every loss sequence of a three-point law (3^7) with VaR and ES known exactly (`P = 0.7, 0.2, 0.1` at `0, 1, 3`: `VaR_0.9 = 1`, `ES_0.9 = 3`) | exact enumeration, no simulation |
| RV-2111, RV-2112 | simulation under the null and under a planted under-forecast with a binomial tolerance of three standard errors | the generator (`numpy.random.default_rng`) |
| RV-2113, RV-2114 | the edge cases stated by the paper (Remark 8: no bet when the e-statistics do not exceed 1; a hit every day: `lambda = alpha / (1 - alpha)`, `M_t = 2^(t-1)` by hand), the record contract | hand computation |

Mutation check run before the unit was closed: changing `x > r` to `x >= r`, 0/0 to 0, the ES numerator to `x - r`, the GREM
mixture weights to a fixed half, or GREL to the past forecasts each fails at least one test of the module.

Measured with the page's recipe (1,000 series of 1,000 days at `alpha = 0.025`, standard normal losses, the true VaR and ES):
the VaR e-process reaches the thresholds 2, 5, 10 and 20 in 0.172, 0.037, 0.010 and 0.006 of the null series, each below its
guaranteed bound (0.5, 0.2, 0.1, 0.05); the table is on `docs/sequential/02_e-backtest.md`.
