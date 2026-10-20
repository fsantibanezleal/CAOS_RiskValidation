# Stress testing: satellites, scenario paths, the IAPE capital projection, reverse stress and scenario models

`riskvalidation.stress` computes the satellites of a default or loss rate and their projection under a scenario, the handling of
scenario paths, the capital engine of the IAPE projection on a static balance sheet (the internal capital target, the Chilean
stack, the P2G proxy, the Annex 3 summary), the reverse stress and the VAR and growth-at-risk scenario models;
`riskvalidation.validation.stress` validates them. The engine computes and makes no compliance claim: the reverse stress statement
and the P2G proxy are Contraste's own, and every light threshold is a Contraste internal gate, not a regulatory requirement. The
supervisory replication of the Federal Reserve models (C19) is not part of this package (decision E1 of the unit sizing; backlog
item BL-081).

Conventions of every page:

- Arrays are `float64` NumPy arrays. Rates, ratios and probabilities are decimals; a path keeps the unit of its source (percent
  for the Federal Reserve and EBA paths). Money is a float in one currency (CLP millions in the Contraste bake) and the engine
  never converts it.
- Quarters are ISO dates of the quarter end (`2019-12-31`); a fit window is explicit (`start`, `end`), never implied by the data,
  and the engine hard-codes no sample end.
- A function with randomness takes an integer `seed` and uses `numpy.random.Generator(numpy.random.PCG64(seed))`; none of the
  functions of this package draws, so none takes a seed. Every function is pure and deterministic and never mutates its inputs.
- A function that cannot compute a value raises `ValueError` with the reason and the field named; a validation test returns the
  light `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value instead. Nothing returns a placeholder.
- Static balance sheet (decision E4; EBA Methodological Note 2025-01-20, addendum F6): paragraph 33 (p.18) assets and liabilities
  that mature or amortise within the horizon are replaced with similar instruments, no workout or cure of stage 3 assets and no
  capital measures after the reference date; paragraph 34 the same business mix and model; paragraph 90 (p.36) stage 3 exposures
  are not replaced, new stage 3 exposures move into the stock of stage 3 keeping the total constant, with no cures, charge-offs or
  write-offs within the three years; paragraph 91 the initial residual maturity is kept; paragraph 92 (p.37) a constant balance
  sheet for impairments; paragraph 24 (p.16) year-end starting figures and a three-year horizon.
- Every constant and table is transcribed from the dossier named on its page. An item the dossiers mark UNVERIFIED is marked
  UNVERIFIED here and is not silently decided.

| Page | Contents |
|---|---|
| [01. Satellites](01_satellites.md) | the probit, logit and identity links, the OLS fit with standard errors on an explicit window, the scenario projection, the conditional PD under a scenario |
| [02. Scenario paths](02_paths.md) | the transposition of a published path onto the last observed value, the reversion to the baseline, annual to quarterly, deviation paths, the vintage check, and the comparison with the C06 rule |
| [03. Capital](03_capital.md) | the loss-only CET1 path, the PPNR row, the internal capital target, the Chilean stack, the three-year plan, the P2G proxy, the Annex 3 summary, the EBA static balance sheet |
| [04. Reverse stress](04_reverse-stress.md) | the minimum Mahalanobis distance scenario, the closed form for a linear loss, SLSQP otherwise, the plausibility |
| [05. Scenario models](05_scenario-models.md) | the VAR and its conditional forecast, the Azzalini-Capitanio skewed t, growth-at-risk |
| [06. The validation tests](06_validation-tests.md) | the six tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.stress import capital` and
`from riskvalidation.validation import stress`; nothing is re-exported from the package root. Requirements `RV-1301` to `RV-1324` are
in `docs/design/features/u13-stress/requirements.md`, each naming its test. Growth-at-risk step 1 reads `statsmodels`, the optional
extra `stress`; every other function needs NumPy and SciPy only. Sources: dossier 05 of the Contraste initiative (section H),
dossier 04 (I.1 and I.2), dossier 01 (J.4), dossier 03 (B, G.4, H.2, M) and the U13 research addendum (F6, F7, F8); each page
cites the primary sources with their DOIs or URLs. Contract: `engine-stress-api` 1.0.0.
