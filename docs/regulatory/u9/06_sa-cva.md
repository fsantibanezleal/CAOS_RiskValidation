# 06. SA-CVA (MAR50.27 to MAR50.77)

Module `riskvalidation.regulatory.sa_cva` (unit U19, backlog item BL-076, requirements RV-2001 to RV-2005 in `docs/design/features/u19-frtb-cva/requirements.md`).
Primary text: the Basel Framework chapter MAR50, CVA risk, in force 2023-01-01, published 2020-07-08, read on bis.org on 2026-11-15:
https://www.bis.org/basel_framework/chapter/MAR/50.htm?inforce=20230101&published=20200708 (a paragraph `MAR50.x` is the anchor `#50.x`). The formulas of the
chapter are images on that page (`MAR50_20230101_<n>.svg`); the module follows their transcription. The Basel text holds no numeric example of the SA-CVA. The
engine computes and states no compliance claim: it reproduces the published formulas and tables.

## What the SA-CVA is

The SA-CVA is an adaptation of the standardised approach for market risk with a reduced granularity of risk factors, and without default risk and curvature
risk (MAR50.27). It is for banks the supervisor approves; the others use the BA-CVA ([page 04](04_ba-cva.md)). The capital is

$$K = K_{delta} + K_{vega}\quad\text{(MAR50.42)},\qquad \text{RWA} = 12.5\times K\quad\text{(MAR50.1)}$$

$K_{delta}$ is the simple sum of the delta requirements of six risk classes (interest rate, FX, counterparty credit spread, reference credit spread, equity,
commodity; MAR50.43) and $K_{vega}$ the simple sum over five (the same without counterparty credit spread: there is no vega requirement for counterparty credit
spread risk, MAR50.45 and MAR50.63). Both are computed on the whole CVA portfolio including the eligible hedges, and in the same manner (MAR50.46). There is no
curvature charge and no default risk charge in the SA-CVA (MAR50.27).

## The aggregation of one risk class (MAR50.51 to MAR50.53)

For each risk factor $k$ the weighted sensitivities are $WS_k^{CVA}=RW_k\,s_k^{CVA}$ and $WS_k^{Hdg}=RW_k\,s_k^{Hdg}$ and the net is
$WS_k=WS_k^{CVA}-WS_k^{Hdg}$ (the CVA is positive, MAR50.32 and footnote 5). Within a bucket $b$ and across buckets:

$$K_b=\sqrt{\max\Big(0,\ \sum_{k\in b}WS_k^2+\sum_{k\in b}\sum_{l\in b,\,l\ne k}\rho_{kl}\,WS_k\,WS_l+R\sum_{k\in b}\big(WS_k^{Hdg}\big)^2\Big)}$$

$$S_b=\max\Big(-K_b,\ \min\Big(\sum_{k\in b}WS_k,\ K_b\Big)\Big),\qquad K=m_{CVA}\sqrt{\max\Big(0,\ \sum_b K_b^2+\sum_b\sum_{c\ne b}\gamma_{bc}\,S_b\,S_c\Big)}$$

$R=0.01$ is the hedging disallowance parameter that prevents the recognition of a perfect hedge, and $m_{CVA}=1$ unless the supervisor raises it (MAR50.41). Three
points of the formulas that the tests hold:

- The hedging disallowance sits inside the square root: $K_b^2$ grows by exactly $R\sum(WS_k^{Hdg})^2$. A perfect hedge of a single factor leaves
  $\sqrt{R}\,|WS^{Hdg}|=0.1\,|WS^{Hdg}|$.
- $S_b$ is always capped at $K_b$ and floored at $-K_b$ (MAR50.53 (3)), not only when the radicand is negative as in MAR21.4.
- With no hedge $R$ plays no role. The radicand is floored at zero, so $K_b$ and $K$ are never negative.

Functions: `weighted_sensitivity(rw, s_cva, s_hdg)` returns the three weighted values; `bucket_k(ws_cva, ws_hdg, rho, r=R)`; `bucket_s(ws, k_b)`;
`risk_class_k(k_b, s_b, gamma, m_cva)`. An empty bucket has $K_b=0$; a non-finite input, an asymmetric or out-of-range correlation matrix and an $m_{CVA}$ below 1 are
refused naming the field.

## Risk classes, buckets, weights and correlations (MAR50.54 to MAR50.77)

All tables are module data in `riskvalidation.regulatory.sa_cva` as read-only `RegulatoryParameter` records or arrays naming their paragraph and table, pinned
cell by cell in `tests/regulatory/test_sa_cva.py::test_parameters_pinned_to_mar50` against the verbatim rows of the page, with the symmetry of Tables 4, 6 and 9 and
the bucket counts 8, 17, 13 and 11 of Tables 5, 8, 11 and 13. `sa_cva_parameters(risk_class, kind, specified_currency)` returns the parameters of one class.

| Class (code) | Buckets (MAR50) | Delta factor and weight | Vega factor and weight | Cross-bucket $\gamma_{bc}$ |
|---|---|---|---|---|
| Interest rate (`ir`) | one per currency (MAR50.54) | specified currencies: 1, 2, 5, 10, 30 years and inflation, 1.11%, 0.93%, 0.74%, 0.74%, 0.74%, 1.11% (Table 3), $\rho$ of Table 4; other currencies: the parallel yield curve and inflation, 1.58% each, $\rho=40\%$ (MAR50.57) | the interest rate volatilities and the inflation volatilities, 100%, $\rho=40\%$ (MAR50.58) | 0.5 for all currency pairs (MAR50.55) |
| FX (`fx`) | one per currency except the reporting currency (MAR50.59) | one factor, 11% (MAR50.61) | one factor, 100% (MAR50.62) | 0.6 (MAR50.60) |
| Counterparty credit spread (`ccs`) | 8 (Table 5; bucket 1 splits into 1a and 1b) | per name and tenor 0.5, 1, 3, 5, 10 years; weights by bucket and IG or HY and NR (Table 7); $\rho_{kl}=\rho_{tenor}\rho_{name}\rho_{quality}$ (MAR50.65) | none (MAR50.45, MAR50.63) | Table 6 |
| Reference credit spread (`rcs`) | 17 (Table 8) | one factor per bucket, weights of Table 10 | one factor, 100% (MAR50.69) | Table 9; halved between a bucket of 1 to 7 and one of 8 to 14 (MAR50.67 (2)) |
| Equity (`eq`) | 13 (Table 11) | one factor per bucket, weights of Table 12 | one factor, 78% for the large market capitalisation buckets and 100% for the others (MAR50.73) | 15% inside 1 to 10, 75% between 12 and 13, 45% between 12 or 13 and 1 to 10, 0% with 11 (MAR50.71) |
| Commodity (`com`) | 11 (Table 13) | one factor per bucket, weights of Table 14 | one factor, 100% (MAR50.77) | 20% inside 1 to 10, 0% with 11 (MAR50.75) |

The counterparty credit spread correlation (MAR50.65) is `ccs_rho(same_tenor, name_relation, same_quality, index=False)`: $\rho_{tenor}$ is 1 for the same tenor and
0.9 otherwise; $\rho_{name}$ is 1 for the same name, 0.9 for distinct but legally related names and 0.5 otherwise (for an index: 1 for the same index and series, 0.9 for the
same index of distinct series, 0.8 otherwise); $\rho_{quality}$ is 1 for the same credit quality and 0.8 otherwise. The sensitivities are defined by the shifts of
MAR50.56 to MAR50.77 (1 basis point divided by 0.0001, or 1 percent relative divided by 0.01) and are computed by the caller.

## The capital from the sensitivities

`sa_cva_capital(sensitivities, reporting_currency, m_cva=1.0)` takes `Sensitivity` records (risk class, kind, bucket, factor, `s_cva`, `s_hdg`; the counterparty credit
spread records also the name, the credit quality, the legal group and the index series), sums the records of the same risk factor (the net sensitivity of MAR50.51),
weights them, aggregates each risk class and returns `SaCvaCapital` with `k_delta`, `k_vega`, `capital = k_delta + k_vega`, `rwa = 12.5 capital` and the K, $K_b$ and
$S_b$ of every risk class. It refuses a counterparty credit spread vega, an unknown risk class or kind, a non-finite sensitivity, a bucket the class does not have, a factor
the currency class does not have (a "parallel" factor for a specified currency, a "1y" factor for another) and an FX bucket for the reporting currency, naming the offender.

## Readings taken where the text is silent

- A bucket with a single risk factor takes no correlation term. MAR50 states no within-bucket correlation for the reference credit spread, equity, commodity and FX
  buckets, whose delta and vega factor is one simultaneous shift, so $K_b=\sqrt{WS^2+R\,(WS^{Hdg})^2}$. The reading is the engine's (UNVERIFIED against the standard
  text beyond the chapter page) and is stated in the module docstring.
- The equity vega weight of 78% applies to the buckets marked large market capitalisation in Table 11 (1 to 8) and to the qualified index bucket 12, which Table 11
  labels "Large cap, advanced economies"; 100% applies to buckets 9, 10, 11 and 13. MAR50.73 says only "large market capitalisation buckets".
- The correlation tables are symmetric ($\gamma_{bc}=\gamma_{cb}$, $\rho_{kl}=\rho_{lk}$); the printed upper triangle is the source and the lower triangle is printed blank.
- A `Sensitivity` of bucket 1 of counterparty credit spread needs its sub-bucket, "1a" or "1b" (Table 7 has a weight for each); both belong to bucket 1 of Table 6.

## Stated limits

- The CVA sensitivities are the caller's. Their computation from exposure models, regulatory CVA, the prudent valuation standards and the eligibility of the bank
  (MAR50.30 to MAR50.36, CAP50) is the bank's; the engine takes the sensitivities as inputs and does not check that they were measured with the prescribed shifts.
- Index look-through is the caller's (MAR50.49, MAR50.50): the sensitivity of an index hedge to its constituents' risk factors, or its single sensitivity to a qualified index
  bucket under the option of MAR50.50 with the 75% sector rule, is computed outside and passed in.
- The eligibility of a hedge (MAR50.37 to MAR50.39) and the choice of the counterparty or reference credit spread class for a hedge (MAR50.44) are the caller's.
- Securitisation positions and the correlation trading portfolio are outside the SA-CVA as implemented here; so are curvature and default risk, which the SA-CVA does not have.
- The multiplier $m_{CVA}$ above 1 is a supervisory decision (MAR50.41) and an argument; the engine does not decide it.

## Tests and fixtures

`tests/regulatory/test_sa_cva.py`: `test_parameters_pinned_to_mar50` (RV-2001), `test_bucket_formula_derived_fixtures` (RV-2002), `test_cross_bucket_and_invariants`
(RV-2003), `test_capital_composition_and_refusals` (RV-2004), `test_stated_limits` (RV-2005). The fixtures marked DERIVED (the Basel text has no numbers) are
computed from the transcription and recomputed in the tests by an independent loop over the formula, never typed as answers: SA-A, one factor with a weighted sensitivity
of 100, gives $K_b=100$; SA-B, a perfect hedge of 100, gives $\sqrt{0.01\cdot100^2}=10$; SA-C, two factors at $\rho=0.91$ with weighted sensitivities 100 and 50, gives
146.9694, and with a hedge of 30 on the second 118.5285; SA-D, EUR sensitivities of 2,000,000 and 1,000,000 at one and two years, gives 30,904.485; SA-E, two buckets
with $K=(100,60)$, $S=(100,-60)$ and $\gamma=0.5$, gives 87.178.
