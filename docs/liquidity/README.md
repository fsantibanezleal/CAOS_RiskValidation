# Liquidity: run-off, counterbalancing capacity, ladders, scenarios, reverse stress, the depositor generator, the LCR and NSFR, and the validation tests (U11)

The liquidity part of `riskvalidation` (contract `engine-liquidity-api` 1.2.0, requirements in
[`docs/design/features/u11-liquidity/requirements.md`](../design/features/u11-liquidity/requirements.md)). The engine computes and states no
compliance claim: the LCR is a Basel formula on aggregates with stated assumptions, never the bank's reported or regulatory LCR. Rates are
decimals per relevant day, cumulative outflow is `1 - (1 - q)^n`, a survival period is "survives the horizon" with the horizon when the net
liquidity position never turns negative (never a number), and a missing input is refused with the field named, never filled. Money is a float in USD
thousands as the call report gives it. Modules are imported by path (`from riskvalidation.liquidity import runoff`); nothing is re-exported.

| Page | Module | What it holds |
|---|---|---|
| [01_runoff.md](01_runoff.md) | `riskvalidation.liquidity.runoff` | constant daily compounding and its inverse, contractual roll-off, `LIST_PARAMETERS`, `TABLE_A1`, `SEGMENTS` |
| [02_counterbalancing-capacity.md](02_counterbalancing-capacity.md) | `riskvalidation.liquidity.cbc` | the haircut rule, liquidity value, monetisation, the CBC stack, the HTM loss |
| [03_ladders.md](03_ladders.md) | `riskvalidation.liquidity.ladder` | contractual gaps, behavioural outflows, the NLP, the survival period, the cliff effect |
| [04_scenarios.md](04_scenarios.md) | `riskvalidation.liquidity.scenarios` | LiST parameter sets, families and phases, the fast run, the anchor, the sensitivity grids |
| [05_reverse-stress.md](05_reverse-stress.md) | `riskvalidation.liquidity.reverse` | the survival period against the run-off multiplier and the multiplier that reaches a target |
| [06_depositor-generator.md](06_depositor-generator.md) | `riskvalidation.liquidity.deposits` | the depositor generator with known truth |
| [07_lcr-and-nsfr.md](07_lcr-and-nsfr.md) | `riskvalidation.regulatory.lcr_nsfr` | HQLA, the LCR outflows, inflows, cap and ratio, the NSFR ASF, RSF and ratio |
| [08_validation-tests.md](08_validation-tests.md) | `riskvalidation.validation.liquidity` | the eleven `TestResult` functions |

Common input checks are in `riskvalidation.liquidity._checks`. The pages quote the sources they transcribe: the ECB LiST 2019
methodological note and results annex, BCBS d555 Table A.1, the Basel Framework LCR and NSF standards and the Federal Reserve discount window
margins. The licence classes of the BCBS d555 Table A.1 and of the ECB LiST parameters are readings of the notices, not legal advice; the BIS
notice covering the Basel Framework pages of `lcr_nsfr.py` is read by analogy and stays UNVERIFIED for those pages. The FDIC names no
call-report schedule or line for any field used: the fields are named by their FDIC names and titles; `DEPUNA` and `DEPINS` cover domestic
offices (denominator `DEPDOM`) while `DEP` includes foreign offices, and both denominators are stated where the uninsured share appears.
