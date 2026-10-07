# Design: U4 rating transitions

Requirements: [`requirements.md`](requirements.md). Research: Contraste dossier 13.

## Package layout

| Module | Holds |
|---|---|
| `riskvalidation.transitions.estimators` | `cohort(counts, withdrawals="exclude" or "state")`, `pooled_cohort(counts_by_period)`, `duration_generator(paths)`, `aalen_johansen(paths, s, t)` |
| `riskvalidation.transitions.embedding` | `log_series(P)`, `embedding_diagnostics(P)` (S, the three Theorem 3 conditions, stochastic monotonicity), `generator(P, method="diagonal" or "weighted" or "jlt")`, `l1_distance(P, Q, t=1)` |
| `riskvalidation.transitions.em` | `em_generator(counts_by_step, dt, q0=None, tol, max_iter)` returning the generator, the log-likelihood path and the expected jumps and holding times |
| `riskvalidation.transitions.intervals` | `pd_wald(defaults, firm_years)`, `pd_jeffreys(defaults, n)`, `pd_bootstrap(q, n_by_grade, horizon, n_boot, rng)` |
| `riskvalidation.transitions.mobility` | `svd_mobility(P)`, `trace_mobility(P)` |
| `riskvalidation.transitions.ttc` | `ttc_portfolio(T, origination)`, `project(T, w0, origination, years)` |
| `riskvalidation.validation.transitions` | the registered tests `rating.time_homogeneity`, `rating.markov_order`, `rating.matrix_reference`, `rating.momentum` |
| `riskvalidation.generators.paths` | `RatingPaths` (paths, snapshots, truth, verifier) |

## Conventions

- States are ordered best to worst, default last and absorbing; a withdrawal is a separate column, never a grade.
- A count matrix is rows of the beginning state, columns of the end state (plus withdrawals): CEREP's layout.
- Paths are a list of (times, states) per obligor with an end time (withdrawal or horizon) and a withdrawal flag, so
  every estimator that needs exits can tell censoring from default.
- Matrix exponentials and logarithms by SciPy (`expm`, `logm` as a cross-check of the series); every function returns
  plain arrays and dictionaries, JSON-safe through `jsonable`.

## The tests

| Test | Data | Statistic | Degrees of freedom | Reference |
|---|---|---|---|---|
| `rating.time_homogeneity` | count matrices for T periods | -2 log lambda (3.5); chi-square (3.8) in `extras` | (T - 1) m(m - 1) less cells with zero pooled probability | Anderson and Goodman 1957, section 3.2 |
| `rating.markov_order` | triplet counts n_ijk | -2 log lambda (3.12) | m(m - 1)^2 less empty middle-state rows | section 3.3 |
| `rating.matrix_reference` | a count matrix and a reference P0 | sum_i sum_j n_i (p^_ij - p0_ij)^2 / p0_ij (3.1) | m(m - 1) less zero reference cells | section 3.1 |
| `rating.momentum` | paths | partial-likelihood score or likelihood ratio for c in q_i exp(c Z) | 1 | dos Reis et al. 2020, section 4.1 (after Andersen et al. 1991) |

The momentum hazard is fitted by maximising the Cox partial likelihood stratified by the current grade (each grade its
own baseline), the covariate Z = 1 for an obligor whose last move into the grade was a downgrade (or an upgrade for
the upward test); ties broken by the exact time order of the generator's paths.

## The generator

`RatingPaths(q, obligors_by_grade, horizon, momentum=None, regimes=None, withdrawal_rate=0.0, informative=0.0,
snapshots=None)`. The Markov case draws each obligor's holding time from Exp(q_i) and its next state with the
probabilities q_ij / q_i; momentum adds dos Reis et al.'s self-exciting term, simulated by thinning (Ogata) with the
excitation spread evenly over the downgrade destinations; a regime list switches q at calendar times; withdrawals add
an exit with rate w (informative: w times (1 + k) in the grades above default for an obligor within its last year
before a planned default, drawn by first simulating the path without withdrawal). Truth: q, exp(tq) for the snapshot
steps, the PD by grade at 1 to 5 years, and, in the Markov case, the expected jumps and holding times the verifier
checks.

## Reproductions

Israel, Rosenthal and Wei (2001) section 4: the paper's PDF in the device data root (`RISKVALIDATION_DATA`), read and
checked against its print (the matrices' rows, the nine distances); the test skips without it (dossier 13, item 6 of
its open list).
