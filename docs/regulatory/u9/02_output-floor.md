# 02. The output floor (RBC20, RBC90)

Module `riskvalidation.regulatory.output_floor`. Sources: dossier 01 section C.1 and the contract `engine-regulatory-api` 1.0.0.
Primary texts: BCBS RBC20, in force 2023-01-01, published 2020-11-26
https://www.bis.org/committees/bcbs/basel-framework/standard/rbc/20/inforce/2023-01-01/published/2020-11-26, and RBC90, in force
2023-01-01, published 2020-03-27
https://www.bis.org/committees/bcbs/basel-framework/standard/rbc/90/inforce/2023-01-01/published/2020-03-27.

## The floor

$$RWA = \max\big(RWA_{\text{nominated}},\ f_t\cdot RWA_{\text{SA-only}}\big),\qquad f_t = 72.5\%\ \text{fully phased in (RBC20.4)}$$

With the optional national cap of RBC90.2, the increase from the floor is limited to a share of the pre-floor RWA during the phase-in:

$$RWA = \min\Big(\max\big(RWA_{\text{nominated}},\ f_t\,RWA_{\text{SA-only}}\big),\ RWA_{\text{nominated}}\,(1+c)\Big),\qquad c = 25\%$$

Symbols: $RWA_{\text{nominated}}$ the bank's RWA from its approved approaches (the pre-floor RWA); $RWA_{\text{SA-only}}$ the RWA of the
same exposures under the standardised approaches only; $f_t$ the phase-in factor of year $t$; $c$ the cap share. `floored_rwa(rwa_nominated,
rwa_sa_only, factor, cap_share=None)` returns the floored RWA; the result is never below `rwa_nominated`. RWA may be in one currency or in
index points; the engine converts nothing.

The floor base takes standardised approaches only (RBC20.11); the IRB approach, SEC-IRBA, the internal models approach for market risk
and the VaR models approach to CCR (IMM) are excluded (RBC20.12). Building `rwa_sa_only` from the standardised calculators (for example
`riskvalidation.regulatory.standardised`) is the caller's.

## The phase-in

`phase_in_factor(year)` returns a `RegulatoryParameter` of paragraph `RBC90.1`, table `RBC90.1 Table 1` and in-force date 2023-01-01:

| Year | 2023 | 2024 | 2025 | 2026 | 2027 | 2028 and later |
|---|---|---|---|---|---|---|
| Factor $f_t$ | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.725 |

A year before 2023 raises `ValueError`. These are the BCBS dates; actual jurisdictional dates differ (dossier 01 L), and the caller passes its
own factor to `floored_rwa` where they do.

## Golden values

RBC20.13 Table 1 (dossier 01 C.1 and M.1; both are fixture examples `rbc20-13-output-floor` and `rbc20-13-output-floor-capped`): pre-floor RWA
76, standardised-only RWA 140, factor 0.725.

- `floored_rwa(76, 140, 0.725)` is 101.5 (the floor binds: 140 x 0.725 = 101.5, above 76).
- `floored_rwa(76, 140, 0.725, cap_share=0.25)` is 95 (the increase is capped: 76 x 1.25 = 95).

Both are exact to the floating point of the products. With the factor of 2023 (0.50) the floor does not bind: 0.50 x 140 = 70 is below 76,
so the result is 76.

## Conventions

The result is `max(rwa_nominated, factor x rwa_sa_only)` without the cap, and the `min` of that and `rwa_nominated x (1 + cap_share)` with it. A
negative or non-finite RWA, factor or cap share is refused with the field named. Minimum ratios (CET1 4.5 percent, Tier 1 6 percent, Total 8
percent of RWA plus the 2.5 percent conservation buffer, RBC20.1) are not computed here.

## Limits

- The engine computes and states no compliance claim (dossier 01 M.4).
- The national schedules of the EU, UK and US are not modelled (dossier 01 L, items 6, 7 and 12 of its UNVERIFIED list); the BCBS table is.
- The RBC90.2 cap is optional and national; the engine applies it only when `cap_share` is given.
