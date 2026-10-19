# 01. Run-off: constant daily compounding, the transcribed LiST and Table A.1 parameters

Module `riskvalidation.liquidity.runoff`. Sources: dossier 05 section E.1 (line 237: the open-maturity deposits of the ECB LiST 2019
run off at a constant daily compounding rate on relevant days) and E.2 (BCBS d555 Table A.1); the U11 research addendum L1 (the ECB
methodological note of February 2019, Tables 3.2, 3.3, 2.2 and 2.3, paragraphs 94 to 126) and L2.1. The engine contract is
`engine-liquidity-api` 1.0.0 (Contraste `docs/contracts/engine/liquidity-api.md`).

## Equations

A deposit class loses a constant share `q` of the stock of the day before on each relevant day. After `n` relevant days the cumulative
outflow share is

$$F(n) = 1-(1-q)^{n},\qquad q = 1-(1-F)^{1/n}$$

and for a daily rate that changes with the stress state, `q_d` on day `d`,

$$F_d = 1-\prod_{s\le d}\,(1-q_s),\qquad \text{outflow}_d = \text{opening}\times\prod_{s<d}(1-q_s)\times q_d .$$

`cumulative_outflow_constant(q, n)` is the first form, `daily_rate_from_cumulative(F, n)` the second (its inverse, round trip to 1e-12),
`cumulative_outflow_path(q_d)` the path form. A rate of exactly 0 or 1 is legitimate for a segment (a financial deposit leaves in full on
day 1): `provisions.term_structure.from_hazard` has the same arithmetic but refuses both and indexes months, so it was considered and
not reused. A contractual class rolls off differently: `contractual_rolloff(maturing, rate)` returns `rate x maturing` on each day and the
remainder `(1 - rate) x maturing` stays past the horizon (note para 94); `financial_no_rollover(maturing)` returns the maturing flow
whole (note para 98, rate 1.0).

Symbols: `q` daily rate (a decimal per relevant day), `n` relevant days (an integer counted from 1), `F` cumulative outflow share in
`[0, 1]`, money in USD thousands as the FDIC call report gives it (the engine never converts).

## The transcribed parameter sets (module data)

`LIST_PARAMETERS[severity]` holds four blocks, transcribed from the ECB note (read 2026-10-18; paragraph numbers are the note's):

| Block | Source | Content |
|---|---|---|
| `open_maturity` | Table 3.3, para 107 | the daily compounding rate by class, on the stock of the day before, relevant days only |
| `contractual_outflow` | Table 3.2, para 94 | the roll-off rate applied to each flow when it matures, constant across maturities |
| `contractual_inflow` | Table 3.2, paras 101 to 105 | 1.0 for secured lending, loans to credit institutions and other financial customers, FX swaps, derivatives, own paper and other inflows; 0 for loans to retail, non-financial corporates and others |
| `drawdown` | para 121 | committed credit facilities 0.00202704 (Adverse) and 0.00257634 (Extreme); liquidity facilities 0.01443904 and 0.02176434, daily, first three months only |

The open-maturity daily rates (Baseline, Adverse, Extreme): stable retail sight 0, 0.00103258, 0.00161494; other retail 0, 0.00373218,
0.00440873; operational 0, 0.00514464, 0.00750509; non-operational non-financial corporates and others 0, 0.00688541, 0.01065229;
open repos, non-operational credit institutions and other financial customers, other transactions 1.0 in all three. The contractual
roll-off rates: stable retail 0, 0.18, 0.27; other retail 0, 0.39, 0.48; operational 0, 0.37, 0.50; non-operational non-financial
corporates and others 0, 0.52, 0.76; the other classes 1.0. The fastest sight class runs at its own rate: there is no stable-retail
proxy for uninsured corporate balances. `LIST_RELEVANT_DAYS = 125` and `LIST_FACILITY_DAYS = 63`.

`TABLE_A1` holds the nine observed runs of BCBS d555 Box A, Table A.1 (SVB 0.85 in 2 days, First Republic 0.57 in 90, SVB UK 0.30 in 1,
Credit Suisse 0.21 in 90, Signature 0.20 in 1, Icesave 0.20 in 75, Northern Rock 0.20 in 4, WaMu 0.10 in 10, IndyMac 0.08 in 14), with
the tenth row "LCR deposit outflow rates, 3% to 40% over 30 days" as `TABLE_A1_LCR_RANGE = (0.03, 0.40, 30)`; the LCR rate table itself is
in `regulatory.lcr_nsfr`. Footnote (a) of the table: First Republic's outflow excludes 30 bn USD of deposits placed by banks in Q1 2023,
SVB's includes the expected outflow of management for 10 March. `SEGMENTS` are the five segments of dossier 05 E.5: retail insured,
retail uninsured, operational, non-operational corporate, financial.

## Golden values

| Check | Value | Origin |
|---|---|---|
| `q = 0.00103258`, `n = 125` | `0.121148` (`0.12114837`) | dossier 05 line 237, the note's Table 2.2 last row 12.11 percent |
| `q = 0.00161494`, `n = 125` | `0.182930` | recomputed; the note's Table 2.3 prints 18.29 percent |
| `q = 0.00373218`, `n = 125` | `0.373368` | recomputed; Table 2.2 prints 37.34 percent |
| `q = 0.00688541`, `n = 2` | `0.013723` | recomputed; Table 2.2 second bucket prints 1.37 percent |
| `[0.01, 0.02, 0.03]` | `[0.01, 0.0298, 0.058906]` | `1 - 0.99 x 0.98 x 0.97` |
| SVB `0.85` in 2 days | daily rate `0.612702` | recomputed, not a dossier value |
| First Republic, Northern Rock, Icesave, WaMu, IndyMac, Credit Suisse | `0.009334`, `0.054258`, `0.002971`, `0.010481`, `0.005938`, `0.002616` | recomputed |
| facilities after 63 relevant days | 0.12 and 0.15 (credit), 0.60 and 0.75 (liquidity) | note para 122 |

The tests reproduce each value through a second route: a loop of daily survivals, and the percentages the ECB note prints in its own
Tables 2.2 and 2.3 (first, second and last bucket for every class).

## Conventions

Rates are decimals per relevant day, shares are decimals in `[0, 1]`, days are integers counted from 1. Every function is pure and
returns a new array; a scalar call returns a float.

## Limits

- The relevant days of the note are the 2019 TARGET2 calendar (125 in the six months; para 16 lists the closures). A ladder on a US
  business-day calendar is a stated convention, and every LiST cumulative figure says which calendar it is on. UNVERIFIED as a calendar
  choice (the contract's open item A2).
- The 52 and 76 percent of the contractual classes are outflow rates applied to each flow when it matures (para 94), not a six-month
  cumulative on the stock: the dossier's "over six months" is correct only for flows that mature inside the horizon.
- The LiST shocks are idiosyncratic only (annex slide 5). The Table A.1 rows are observations; the daily rate each implies is a
  recomputation, never a forecast.
- The licence classes of the BCBS d555 Table A.1 and of the ECB LiST parameters are readings of the notices (addendum L2.1 and L2.2),
  not legal advice.

## Sources

- ECB Banking Supervision, Sensitivity analysis of liquidity risk, stress test 2019, methodological note (2019-02-06):
  https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.ECB_sensitivity_analysis_of_liquidity_risk-stress_test_2019-methodological_note_20190206~6771e88926.en.pdf
- ECB, LiST 2019 final results annex (2019-10-07):
  https://www.bankingsupervision.europa.eu/press/pr/date/2019/html/ssm.pr191007_annex~537c259b6d.en.pdf
- BCBS, Report on the 2023 banking turmoil (d555), Box A, Table A.1: https://www.bis.org/bcbs/publ/d555.htm
