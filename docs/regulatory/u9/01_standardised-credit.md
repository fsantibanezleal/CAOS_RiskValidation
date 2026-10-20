# 01. Standardised credit risk weights (CRE20)

Module `riskvalidation.regulatory.standardised`. Sources: dossier 01 section C (the CRE20 tables and paragraphs) and the contract
`engine-regulatory-api` 1.0.0 with the `cre20_grid` of the regulatory-calculators fixture. Primary text: BCBS CRE20, in force
2023-01-01, published 2025-06-10
https://www.bis.org/committees/bcbs/basel-framework/standard/cre/20/inforce/2023-01-01/published/2025-06-10. CRE21 governs the
use of external ratings and CRE22 credit risk mitigation; neither is implemented here.

## What a function returns

Every function returns a `RegulatoryParameter(value, paragraph, table, in_force)` (`RiskWeight` is an alias of the same class):

$$\text{parameter} = (v,\ \text{paragraph},\ \text{table},\ \text{in-force date}),\qquad v\in\mathbb{R}\ \text{a decimal}$$

Symbols: $v$ is a risk weight, a credit conversion factor (CCF), a phase-in factor or a multiplier, as a decimal (0.2 is 20
percent). The paragraph and table strings are those of the Basel Framework (for example `CRE20.7` and `CRE20 Table 1`); the in-force
date is that of the chapter version, `2023-01-01` for CRE20 (dossier 01 M.3 item 8: every displayed regulatory parameter carries its
paragraph and the version of the chapter). A key outside a table raises `ValueError` naming the table and the paragraph; the engine
never returns a placeholder weight.

## The tables

Rating buckets are the labels of the tables; mapping an external rating to a bucket is the caller's (CRE21).

| Function | Paragraph and table | Values |
|---|---|---|
| `sovereign_risk_weight(rating)` | CRE20.7 Table 1 | AAA to AA- 0, A+ to A- 0.2, BBB+ to BBB- 0.5, BB+ to B- 1.0, below B- 1.5, unrated 1.0 |
| `eca_risk_weight(score)` | CRE20.9 Table 2 | scores 0 and 1: 0; 2: 0.2; 3: 0.5; 4 to 6: 1.0; 7: 1.5 |
| `bank_risk_weight_ecra(rating, short_term)` | CRE20.18 Table 6 | base 0.2, 0.3, 0.5, 1.0, 1.5; short-term 0.2, 0.2, 0.2, 0.5, 1.5; `unrated` is the SCRA and is refused |
| `bank_risk_weight_scra(grade, short_term)` | CRE20.21 Table 7 | Grade A 0.4 (short-term 0.2), Grade B 0.75 (0.5), Grade C 1.5 (1.5) |
| `corporate_risk_weight(rating, ...)` | CRE20.43 Table 10 | 0.2, 0.5, 0.75, 1.0 (BB+ to BB-), 1.5 (below BB-), unrated 1.0 |
| the same, `investment_grade_unrated` | CRE20.44 and CRE20.46 | 0.65 (investment grade where ratings are not allowed) |
| the same, `sme` and `regulatory_retail_sme` | CRE20.47 | unrated SME 0.85; regulatory retail SME 0.75 |
| `retail_risk_weight(kind)` | CRE20.68 | regulatory 0.75, transactor 0.45, other 1.0 |
| `equity_risk_weight(kind)` | CRE20.57 to CRE20.60 | equity 2.5, speculative unlisted 4.0, legislated programmes 1.0, subordinated debt 1.5 |
| `credit_conversion_factor(kind)` | CRE20.94 to CRE20.101 | 1.0 for direct credit substitutes, repos, securities lending and forward purchases; 0.5 for NIFs and RUFs and transaction-related contingents; 0.4 commitments; 0.2 short-term self-liquidating trade letters of credit; 0.1 unconditionally cancellable |
| `defaulted_risk_weight(share, ...)` | CRE20.104 to CRE20.107 | 1.5 when the specific provision share is below 0.2, 1.0 from 0.2; 1.0 for defaulted residential real estate that is not cash-flow dependent |

The three corporate flags describe an unrated corporate: a flag on a rated bucket and two flags at once are refused. Legislated
programme equity is 100 percent only up to 10 percent of Total capital (CRE20.59); the cap is the caller's.

## Real estate by LTV

Residential, whole loan: the weight depends on the loan-to-value ratio band, and **the upper edge of a band belongs to it** (an LTV of
exactly 0.8 gives 0.3).

$$w(\text{LTV}) = \begin{cases} 0.2 & \text{LTV}\le 0.5\\ 0.25 & 0.5<\text{LTV}\le 0.6\\ 0.3 & 0.6<\text{LTV}\le 0.8\\ 0.4 & 0.8<\text{LTV}\le 0.9\\ 0.5 & 0.9<\text{LTV}\le 1.0\\ 0.7 & \text{LTV}>1.0\end{cases}\quad\text{(CRE20.82 Table 11)}$$

Cash-flow dependent residential (CRE20.84 Table 12): 0.3, 0.35, 0.45, 0.6, 0.75, 1.05 on the same bands. The loan-splitting alternative
(CRE20.83) is 0.2 for the part of the loan up to 55 percent of the property value; `residential_re_risk_weight(..., loan_splitting=True)`
returns that weight and the split of the loan is the caller's. Loan splitting on a cash-flow dependent exposure is refused.

Commercial real estate (`commercial_re_risk_weight(ltv, cash_flow_dependent, counterparty_weight)`): not cash-flow dependent, CRE20.85
Table 13, $w=\min(0.6,\ w_{cp})$ for LTV at most 0.6 and $w=w_{cp}$ above, with $w_{cp}$ the counterparty weight (required); cash-flow
dependent, CRE20.87 Table 14, 0.7 up to 0.6, 0.9 up to 0.8 and 1.1 above 0.8.

## Currency mismatch

$$w_{adj}=\min(1.5\,w,\ 1.5)\qquad\text{(CRE20.92)}$$

`apply_currency_mismatch(weight)` returns the adjusted weight for an unhedged retail or residential exposure to an individual; the
multiplier 1.5 and the cap 1.5 are the `multiplier` and `cap` rows of `lookup("currency-mismatch", key)`.

## The lookup

`lookup(exposure_class, key, **options)` dispatches to the functions above and returns exactly their parameter. The classes are
`sovereign`, `eca`, `bank-ecra`, `bank-scra`, `corporate`, `retail`, `equity`, `residential-re`, `commercial-re`, `defaulted`, `ccf` and
`currency-mismatch`; the key is the rating bucket, grade or kind, an ECA score or an LTV as a string, and the options are `short_term`,
`cash_flow_dependent` and `counterparty_weight`. An option a class does not use must stay at its default. Every one of the 79 cells of the
`cre20_grid` of the regulatory-calculators fixture is keyed as `lookup(exposure_class, key, short_term, cash_flow_dependent,
counterparty_weight)` and equals the cell's value, table, paragraph and in-force date (`tests/regulatory/test_reg_standardised.py`).

## Golden values

The tables above are the goldens (dossier 01 C, M.1): for example `sovereign_risk_weight("A+ to A-")` is 0.2 (CRE20.7),
`bank_risk_weight_ecra("BB+ to B-", short_term=True)` is 0.5 (CRE20.18), `residential_re_risk_weight(0.8)` is 0.3 and
`residential_re_risk_weight(0.8, cash_flow_dependent=True)` is 0.45, `apply_currency_mismatch(0.75)` is 1.125 and
`apply_currency_mismatch(1.2)` is 1.5.

## Conventions

The in-force date `2023-01-01` and the publication date 2025-06-10 are those of the CRE20 version transcribed. Weights, CCFs and LTVs are
decimals. The engine computes and states no compliance claim; whether a jurisdiction allows external ratings (ECRA) or requires the
SCRA, and its own risk weights, are not modelled.

## Limits

- A further CRE20 version in force 2028-01-01 (published 2025-06-10) is not compared (dossier 01 C).
- The mapping from an external rating to a bucket (CRE21) and credit risk mitigation (CRE22) are not implemented.
- The cap of 10 percent of Total capital on legislated programmes (CRE20.59) is the caller's.
- The refusals of a rated bucket with an unrated-corporate flag and of loan splitting on a cash-flow dependent exposure are the engine's
  reading of CRE20.44 to CRE20.47 and CRE20.83; the contract does not state them.
