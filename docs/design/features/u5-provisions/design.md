# Design: U5 provisions, the IFRS 9 ECL engine (U5-E1)

Requirements: [`requirements.md`](requirements.md). Research: Contraste dossier 02. Contract: `engine-provisions-api` 1.0.0
(Contraste `docs/contracts/engine/provisions-api.md`). The local-regulation modules (`local_cl`, `local_es`) are U5-E2's.

## Package layout

| Module | Holds |
|---|---|
| `riskvalidation.provisions.term_structure` | `TermStructure`, `CompetingRisks`, `from_hazard`, `from_marginal`, `from_cumulative`, `competing_risks`, `forward_pd`, `annualise_pd`, `ttc_to_pit`, `implied_z`, `weighted_pit_pd`, `gauss_hermite_pit_pd` |
| `riskvalidation.provisions.staging` | `StagingPolicy`, `StageResult`, `sicr_quantitative`, `allocate_stages`, `apply_cure_probation`, `collective_transfer` |
| `riskvalidation.provisions.ecl` | `discount_factors`, `ecl_scenario`, `probability_weighted_ecl`, `stage3_ecl`, `poci_allowance`, `simplified_matrix_ecl`, `commitment_ead`, `non_linearity_index`, `weight_sensitivity`, `AllowanceBreakdown`, `portfolio_allowance` |
| `riskvalidation.provisions.overlays` | `Overlay`, `overlay_lines`, `overlay_share`, `total_allowance` (and `check_unique`, `sum_amounts`) |
| `riskvalidation.provisions.reconciliation` | `RollForwardRow`, `allowance_rollforward`, `gca_rollforward`, `check_identities` (and `identity_residuals`, `gross_total`, `carried_by_stage`, `carried_from_lines`) |
| `riskvalidation.provisions._checks` | the shared input checks (open and closed unit intervals, months, weights), each naming its field |
| `riskvalidation.validation.ifrs9` | the nine tests of dossier 02 I.2 |

## Conventions

- Pure NumPy and SciPy; no pandas, lifelines or statsmodels. Every function is deterministic, never mutates an input and
  returns new arrays. Frozen dataclasses.
- PDs at an estimate site are open (0, 1); LGD, CCF, shares and weights closed [0, 1]; weights sum to one within 1e-9 and at
  least two scenarios are required (B5.5.42). A required input out of range raises `ValueError` naming the field.
- Months are strictly increasing integers (period ends). An empty portfolio is valid wherever a function sums over accounts.
- The identity tolerance is one definition: `1e-6 * max(1, G)`, G the total gross carrying amount (sum of absolute closings of the
  35I rows). A roll-forward residual is the difference itself.
- The IFRS 9 tests do not use the registry or the policy file: their light rules are the contract's, and the parameters travel
  in `policy_version`. A test that cannot be evaluated returns `not_evaluated` with NaN statistic and its reason in `notes`.

## Decisions taken where the contract is silent (each raised to the contract owner)

1. `competing_risks` accepts a prepayment hazard in [0, 1): the contract's invariant needs the zero limit (equal to the
   single-risk structure) although it types the field as open.
2. `apply_cure_probation` has no input for the triggers of the accounts it leaves unchanged. A required `now` argument
   (the `StageResult` of `allocate_stages`) carries them (decision C2 of the engine review), so that every unchanged
   Stage 2 or 3 account keeps a trigger inside the enumeration.
3. `reconciliation_identities` has no `overlays` argument, yet the stage closing of stage 1 includes the overlays. They are
   read from the `kind = overlay` lines of the waterfall (a null stage is stage 1). Its metric is the largest residual of the
   roll-forward identities; the stage closing against the stage ECL (a rounding comparison of two stated amounts, 0.0025 on the
   C06 fixture) is reported as `stage_ecl_gap_max` and judged on the same tolerance, so the golden value of 0.0 holds and a
   real gap still turns the light red.
4. `lifetime_pd_backtest` interprets `n_events` as the defaults observed up to each horizon, builds the Vasicek band from
   `n_at_risk` as the central interval of the default count (binomial given the systematic factor, two-sided with alpha/2 in
   each tail) and adds the red-level band to `extras`; the Brier score is computed from the aggregates.
5. `overlay_diagnostics` reads `allowance_by_stage` as the model allowance (overlays excluded); its total-level share is the
   overlay money attached to no segment, because the `Overlay` record has no total-level target.
6. The light parameters of the battery are keyword-only arguments after the contract's arguments, with the contract's values as
   defaults.
