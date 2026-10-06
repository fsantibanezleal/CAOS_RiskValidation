# Tasks: U3, known-truth generators and the size and power harness

| # | Task | Requirement | State |
|---|---|---|---|
| 1 | Read the primary sources and persist them (Contraste dossier 12); pilot the suspected findings | all | done |
| 2 | Generator protocol, verifier, `DefaultCounts` with WP14's model and the default correlation | RV-301, RV-302 | done |
| 3 | `BinormalScores` and `LogisticPortfolio`, their exact truths and verifiers | RV-301, RV-303 | done |
| 4 | `TwoSample`, `Migrations`, `GradeFrequencies`, `NestedLda` | RV-304, RV-305, RV-310 | done |
| 5 | Harness: streams, caching, rules, rates with the Morris SE and the Wilson interval; estimator performance | RV-306, RV-314 | done |
| 6 | Exact rejection probabilities of the count tests, held to the simulation | RV-307 | done |
| 7 | The traffic-lights table; the Hosmer-Lemeshow degrees of freedom decided by measurement | RV-312, RV-313 | done |
| 8 | Published reproductions: WP14 Tables 7 and 8, Yurdakul-Naranjo Table 4, Demler et al. | RV-308, RV-309, RV-310 | done |
| 9 | The size of every p-value test under its null | RV-311 | done |
| 10 | Example, docs pages with figures, changelog, version 0.03.000 | RV-315, RV-316 | done |

## Convergence verdict (2026-10-05)

Converged. Every generator reproduces its exact truth on its own stream; the exact rejection probabilities of the
count tests and of the traffic-lights test with independent years agree with the harness cell by cell; the published
simulations reproduce: BCBS WP14 Tables 7 and 8 in 143 of 144 cells (the normal test in all 72; the traffic lights in
71 of 72, with the tie rule WP14's simulation must have used), Yurdakul and Naranjo Table 4 in 53 of 54, Demler et al.
exactly (1 rejection in 1,000). The two cells that do not reproduce are listed with their evidence in the suite and on
the docs pages. Every registered test with a p-value has a measured size; the normal test over five years is above the
bound, as WP14's own table shows, and its page and result notes state it.

The harness found, and the unit acted on: the Hosmer-Lemeshow default of G - 2 degrees of freedom over-rejects on
validation samples (11.7% at 5%), so the default is now G (`fitted=True` for G - 2); WP14's traffic-lights tables come
from a tie rule opposite to its printed mapping (`ties` option); the chi-square test over grades exceeds its level with
a grade expecting under one default; the ECB AUC test over-rejects when the development AUC is itself an estimate
(11.6% at 5% for equal samples), so `auc_initial_variance` adds the estimate's variance (5.4%, RV-317); the ECB concentration statistic is bounded and, with ten grades and an initial CV of
0.6, cannot reject at 5%, while the chi-square test of homogeneity on the counts
catches a 5% shift 94.5% of the time (`GradeFrequencies(development_obligors=...)`). Refuted: the PSI with bins at the development sample's deciles keeps the benchmark's size.

Open, for the product: Contraste case C22 runs every defect family over the tests built so far and shows the exact
size and power of the count tests live.
