# Tasks: U4 rating transitions

| # | Task | Requirement | State |
|---|---|---|---|
| 1 | Estimators: cohort (withdrawal treatments), pooled and per-period, duration, Aalen-Johansen | RV-401, RV-402 | done |
| 2 | Embedding: log series, Theorem 3 diagnostics, diagonal, weighted and JLT generators, distance, monotonicity; Israel et al. reproduced | RV-403 | done |
| 3 | The EM generator with the closed-form E-step | RV-404 | done |
| 4 | The registered tests: time homogeneity, order, reference matrix, momentum | RV-405 to RV-408, RV-413 | done |
| 5 | Intervals for a PD by grade; mobility; the TTC portfolio | RV-409 to RV-411 | done |
| 6 | The rating-path generator with momentum, regimes, withdrawals and snapshots | RV-412 | done |
| 7 | Size and power of the new tests in the harness; docs pages with equations, sources and figures; README, CHANGELOG, 0.04.000 | RV-413, RV-414 | done |

## Convergence verdict (2026-10-06)

Converged. The estimators recover the rating-path generator's truth (duration and Aalen-Johansen within four Monte
Carlo standard errors; the EM generator from ten annual snapshots within 15% on the lower grades' PDs, its
log-likelihood never decreasing, its closed-form expectations agreeing with the simulated jumps). The embedding module
reproduces Israel, Rosenthal and Wei's section 4 from the paper: eight of the nine distances from the printed matrices,
the ninth (0.116900) only from the printed four-digit JLT generator, which is not equation (3) on the printed matrix
(Contraste finding F-IRW-JLT). The four registered tests hold their size at 4,000 repetitions in their default forms,
each chosen by measurement where a printed form does not: the time-homogeneity likelihood ratio (3.5) rejects 7.85% at
5%, the full-table order chi-square (3.14) 13.78%, the reference chi-square (3.1) 3.02% at 1%; the defaults 4.85%,
4.80% and 4.05% at 5%. Schuermann and Hanson's Table 5 (twelve bounds and lengths) and Engelmann's section 4 (the TTC
portfolio, its PD, the starting PDs and the projection's extremes) reproduce to the printed digits, the latter only
with the printed matrix kept as printed and the balance held at one (renormalising its rows misses two entries).

The harness found, and the unit acted on: a test's seed in the size study depended on how many tests sorted before it,
so adding the four tests moved the measurements the pages quote for five U3 tests; every test now has a fixed seed
(`SEEDS`). The time-homogeneity test rejects under rating momentum (13.5% to 95.8% as alpha goes from 0.25 to 1),
refuting part of the dossier's F-MOMENTUM, while the order and momentum tests reject momentum at every strength and
stay near their level under a recession regime: the three together separate the two defects. With one expected
default, the Wald interval for a PD covers 63%; Agresti-Coull and Jeffreys 98%. With 1,000 obligors in a grade whose
true PD is 2.5bp, the cohort PD is zero in 78% of years and the duration PD never (F-COHORT-ZERO confirmed on the
generator).

Open, for the product: Contraste case C04 runs the estimators, the embedding and the aggregate tests on ESMA CEREP's
published matrices, the path tests and the power ladders on the generator, and Engelmann's drift live.
