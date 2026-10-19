# 02. The counterbalancing capacity: haircuts, liquidity value, monetisation, the CBC stack and the HTM loss

Module `riskvalidation.liquidity.cbc`. Sources: dossier 05 section E.1 line 238 (ECB LiST 2019 methodological note, paras 73 to 75 and
Table 2.1), lines 263 and 480 (held-to-maturity unrealised losses reduce the effective CBC), line 276 (the monetisation schedule); the
U11 research addendum L1.6 (the exact order of para 74) and L4 (the Federal Reserve margins). Contract `engine-liquidity-api` 1.0.0.

## Equations

The counterbalancing capacity (CBC) is the stock of unencumbered liquid assets after haircuts: the funds obtainable from them. The
liquidity value of an asset is its market value less a haircut on the unencumbered amount (note para 63: unencumbered assets only; para
74: market value less a haircut):

$$LV = (MV - E)\,(1-h),\qquad 0\le E\le MV,\; 0\le h\le 1 ,$$

the amount usable on day `d` follows a step schedule `(day, cumulative share)` (the named parameter `monetisation_schedule`):

$$\text{usable}_d = LV\times s(d),\qquad s(d)=\text{share of the last point with day}\le d\ (0\text{ before the first}),$$

and the CBC of a class stack is `CBC(t)` the sum over classes. For a held-to-maturity (HTM) book with carrying value `C` and fair value `V`
at the same encumbrance:

$$\text{loss}=V-C,\quad CBC^{0}=(C-E)(1-h),\quad CBC^{1}=(V-E)(1-h),\quad \text{removed}=CBC^{0}-CBC^{1}\ (\ge 0\text{ when }V\le C).$$

## The haircut rule (note para 74), in order

1. Cash, reserves and committed facilities are at nominal value (haircut 0, para 73): label `nominal-value`.
2. A central-bank-eligible asset takes the central bank's haircut when it is given (the caller's, 1 minus the central bank margin, label
   `central-bank-framework`); when it cannot be determined, the LCR haircut if the asset is HQLA (label `lcr-haircut-hqla`, read from
   `regulatory.lcr_nsfr.HQLA_HAIRCUTS`, never a second table) and 0.50 otherwise (`fifty-percent-fallback`).
3. A marketable asset not eligible takes the equity table for equities, by issuer rating: AAA to A- 0.20, BBB+ to BBB- 0.35, lower or
   unrated 0.50 (label `ecb-equity-table`); for the rest 0.30 if HQLA under the LCR Delegated Regulation (`ecb-non-equity-hqla`) and 0.50
   otherwise (`fifty-percent-fallback`).

## Golden values

| Call | Result |
|---|---|
| equity `AAA-A-`, `BBB+-BBB-`, `lower-or-unrated` | `(0.20, ...)`, `(0.35, ...)`, `(0.50, ...)` label `ecb-equity-table` |
| non-equity HQLA, non-equity not HQLA | `(0.30, "ecb-non-equity-hqla")`, `(0.50, "fifty-percent-fallback")` |
| central-bank-eligible with haircut 0.04 | `(0.04, "central-bank-framework")` |
| eligible, undetermined, `level_2a`; not HQLA | `(0.15, "lcr-haircut-hqla")`; `(0.50, "fifty-percent-fallback")` |
| cash | `(0.0, "nominal-value")` |
| `liquidity_value(100, 0.2)`; `(100, 0.1, encumbered=20)` | `80`; `72` |
| `monetisation_capacity(100, [(1, .5), (2, .8), (3, 1)], 4)` | `[50, 80, 100, 100]` |
| `cbc_stack({"A": (100, 20, 0.1), "B": (50, 0, 0.5)}, [(1, 1)], 2)` | `A = [72, 72]`, `B = [25, 25]` |
| `cbc_with_htm_loss(100, 85, 20, 0.1)` | loss `-15`, without `72`, with `58.5`, removed `13.5` |

The nine haircut cases are also the cases of the validation test `cbc_haircut_rules`.

## Conventions

The Federal Reserve margins in force on 2023-03-10 (table M2 of addendum L4, effective 2022-03-14 to 2023-03-12) give the central bank's
haircut as 1 minus the margin on the fair market value, which is the basis note para 75 prescribes; the same securities at the Bank Term
Funding Program terms count at par with no haircut from 2023-03-12, a distinct, labelled counterfactual and not a restatement of the
2022-12-31 position. Loan margins are ranges and give no single haircut: a loan class needs a stated point inside the printed range.

## Limits

- The HTM unrealised loss is not reported by type: it is applied to the book as a whole or pro rata, a stated assumption (contract item
  A3), never a reported by-type loss. UNVERIFIED as a by-type figure.
- The 30 and 50 percent non-equity haircuts are two sub-bullets that follow Table 2.1 in the note with no header of their own; reading
  them as the non-equity rule is the dossier's reading and the only one that leaves no gap. The fallback "LCR haircut when HQLA, else
  50 percent" sits under the bullet for other central banks' standard credit operations and applies only where the bank cannot apply the
  central bank's haircut with full certainty.
- The monetisation schedule, the securities duration buckets and the encumbrance are conventions or call-report inputs, not parameters
  the dossiers fix. An encumbered amount above the market value is refused.

## Sources

- ECB LiST 2019 methodological note, paras 58 to 75 and Table 2.1:
  https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.ECB_sensitivity_analysis_of_liquidity_risk-stress_test_2019-methodological_note_20190206~6771e88926.en.pdf
- Federal Reserve Discount Window, historical margin tables: https://www.frbdiscountwindow.org/generalpages/historical_margins/margin_tables
- Federal Reserve Board, press release of 2023-03-12 (Bank Term Funding Program):
  https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312a.htm
- BCBS, Basel Framework LCR30 (the HQLA haircuts): https://www.bis.org/committees/bcbs/basel-framework/standard/lcr/
