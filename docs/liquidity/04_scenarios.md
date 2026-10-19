# 04. Scenarios: families and phases, the fast-run calibration, the anchor and the sensitivity grids

Module `riskvalidation.liquidity.scenarios`. Sources: dossier 05 section E.2 line 244 (BCBS Principle 10 and Table A.1), E.5 lines 276 to
278 (rungs 2 to 4); the U11 research addendum L1 (the LiST shocks are idiosyncratic only, annex slide 5; collateral calls, note paras 123
to 126). Contract `engine-liquidity-api` 1.0.0.

## The scenario

BCBS Principle 10 asks for stress tests of "a variety of short-term and protracted institution-specific and market-wide stress scenarios
(individually and in combination)". `compose_scenario(family, phase, parameter_set, segments, horizon)` takes a family (`idiosyncratic`,
`market-wide`, `combined`), a phase (`acute`, 1 to 5 days; `protracted`, up to 6 to 12 months) and a parameter set, and gives every
segment `(LiST class, opening balance)` the daily-rate path of its class, constant over the horizon:

$$q_{j,d}=q(\text{class}_j\mid\text{parameter set}),\qquad O_{j,d}=A_j\prod_{s<d}(1-q_{j,s})\,q_{j,d}.$$

- `list_scenario(severity)` returns the ECB LiST 2019 parameter set of `baseline`, `adverse` or `extreme`: exactly `runoff.LIST_PARAMETERS`,
  with no stable-retail proxy for uninsured corporate balances (the fastest sight class runs at its own rate, A1). The classes are read by
  id (`other-retail`) or as attributes (`stable_retail`).
- A class the parameter set does not hold raises; it is never mapped silently. A financial-customer sight class has rate 1.0 and empties
  the stock on day 1 (note para 111). The opening balances are not altered.
- The mapping of a variant to a family and phase is the named parameter `variant_scenarios`: the caller's. The family and phase label the
  scenario; the rates are those of the class.

## Fast-run calibration and the anchor

`fast_run_scenario(rows, rule)` gives the daily rate of the concentrated uninsured segment from observed runs, the Table A.1
`(outflow share, days)` pairs the caller selects. Each row's rate is `daily_rate_from_cumulative`; the fitted rate is the statistic the
rule names (the named parameter `fast_run_calibration_rule`):

$$\hat q=\operatorname{stat}\{\,1-(1-F_i)^{1/n_i}\,\},\qquad \operatorname{stat}\in\{\max,\ \operatorname{median},\ \operatorname{mean}\}.$$

When the caller marks the rows `scored=True`, a 2023 row of Table A.1 is refused with a message (decision Q4: the scored variant is
calibrated on pre-2023 rows, the 2023 rows are scored against). The result `FastRun` holds the fitted rate, the implied rates and the
class `concentrated-uninsured`, the only class a fast-run parameter set holds. `anchor_from_observation(share, days)` returns the observed
outflow, the days and the implied daily rate, with `forecast_claim` always `False`: an observation is never a forecast.

## The sensitivity grids

`uninsured_share_grid(insured, uninsured, shares)` re-splits the call-report deposits (DEPINS and DEPUNA) at each uninsured share, keeping
the total: `uninsured' = total x share`, `insured' = total - uninsured'`. `htm_loss_grid(reported_loss, multiples)` returns
`reported_loss x multiple` for each multiple (0 removes the loss, 1 is the reported loss). Both grids are increasing (dossier 05 line 278).

## Golden values

| Call | Result |
|---|---|
| `list_scenario("adverse")` | `stable_retail = 0.00103258`, `non_operational_non_financial_corporates = 0.00688541` |
| `list_scenario("extreme")` | `0.00161494` and `0.01065229` |
| `compose_scenario(..., Adverse, {"retail": ("other-retail", 100)}, 2)` | rates `[0.00373218, 0.00373218]`, outflows `[0.373218, 0.371825...]` (day 2 is `99.626782 x 0.00373218`) |
| `fast_run_scenario` on Icesave, Northern Rock, WaMu, IndyMac | implied `[0.002971, 0.054258, 0.010481, 0.005938]`; fitted `0.054258` (maximum), `0.0082095` (median), `0.018412` (mean) |
| `anchor_from_observation(0.85, 2)`; `(0.20, 4)` | implied rate `0.612702`; `0.054258` |
| `uninsured_share_grid(40, 60, [0.5, 0.9])` | `[(50, 50), (10, 90)]` |
| `htm_loss_grid(-15, [0, 1, 2])` | `[0, -15, -30]` |

The medians and means are recomputed in the tests with the Python standard library.

## Limits

- Collateral calls have no numeric parameter in the LiST note (addendum L1.5: none found in paras 119 to 127 and Tables 2.4, 3.2 and
  3.3; the results annex reports the observed size only): a scenario input the caller supplies, labelled. UNVERIFIED as a parameter.
- The calibration method of the fast run is not given by the dossiers: UNVERIFIED, the rule is a convention and the named parameter
  carries it. The committed lines are not in the FDIC financials read: none is modelled.
- The LiST shocks are idiosyncratic only; a market-wide or combined scenario is a label and the caller's parameter set, never a LiST
  figure. The LiST relevant-day calendar is the 2019 TARGET2 calendar (page 01).
- The 2023 observations anchor and score; they are not a forecast of any bank.

## Sources

- BCBS, Principles for sound liquidity risk management and supervision (2008, bcbs144): https://www.bis.org/publ/bcbs144.htm
- BCBS, Report on the 2023 banking turmoil (d555), Box A, Table A.1: https://www.bis.org/bcbs/publ/d555.htm
- ECB LiST 2019 methodological note, paras 111, 119 to 126:
  https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.ECB_sensitivity_analysis_of_liquidity_risk-stress_test_2019-methodological_note_20190206~6771e88926.en.pdf
