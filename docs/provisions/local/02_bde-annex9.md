# Banco de Espana Circular 4/2017, Annex 9: coverage tables (Spain)

Module `riskvalidation.provisions.local_es`. Source: Contraste research dossier 03, sections K.1 to K.3, read on 2026-10-04
from Circular 4/2017 of 27 November (BOE-A-2017-14334, https://www.boe.es/eli/es/cir/2017/11/27/4; in force 2018-01-01) and
Circular 6/2021 of 22 December (BOE-A-2021-21666), which replaced the tables of points 140, 144 and 172 of Annex 9,
applicable from 2022-06-30. The tables below are asserted cell by cell against a fixture transcribed from the dossier
(`tests/fixtures/provisions_local/annex9_tables.json`). The engine computes; it makes no compliance claim.

## Categories and the stage map

Annex 9 ("Analisis y cobertura del riesgo de credito") sets, for entities that do not use internal methodologies for
collective estimates, the "soluciones alternativas": a coverage percentage of the uncovered amount by risk segment and
category. Spanish accounting for credit institutions applies the IFRS 9 expected-loss model, unlike the Chilean CNCB.

| Circular 4/2017 category (norma 29) | Allowance | IFRS 9 analogue | Engine value |
|---|---|---|---|
| Riesgo normal | 12-month expected credit losses | Stage 1 | `stage1` |
| Riesgo normal en vigilancia especial | lifetime expected credit losses | Stage 2 | `stage2` |
| Riesgo dudoso | expected credit losses, credit-impaired | Stage 3 | `stage3` |
| Riesgo fallido | none (derecognised) | Write-off | `written-off` |

Circular 6/2021 states the mapping for reporting: "la categoria de fase 1 se correspondera con la de riesgo normal, la de
fase 2 con la de riesgo normal en vigilancia especial y la de fase 3 con la de riesgo dudoso".
`annex9_category_to_stage(category)` is total and injective on the four categories and raises on any other.

## Coverage on the uncovered amount

Coverage applies to the "importe no cubierto con garantias eficaces":

$$
\text{Cobertura}_j = c_{s,k}\cdot \max\!\left(0,\ EAD_j - \sum_g \text{ValorAjustado}_g\right)
$$

with $c_{s,k}$ the percentage for risk segment $s$ and category or arrears bucket $k$. `annex9_uncovered_amount(ead,
adjusted_guarantee_values)` is the max term and `annex9_coverage(segment, category, ead, adjusted_guarantee_values,
days_past_due)` the product. The guarantee values are the caller's adjusted values; the haircut table for effective
guarantees is not transcribed.

## Point 144: normal and normal in special watch

`annex9_normal_coverage(segment, category)`, percent of the uncovered amount, applicable from 2022-06-30 (Circular 6/2021).

| Segment | Riesgo normal | Normal en vigilancia especial |
|---|---|---|
| A.1.1 | 1.9 | 30.0 |
| A.1.2 | 2.0 | 18.8 |
| A.1.3 | 0.6 | 9.6 |
| A.2.1 | 0.6 | 9.6 |
| A.2.2 | 1.1 | 17.8 |
| A.2.3 | 1.4 | 13.9 |
| B.1.1 | 0.7 | 18.0 |
| B.1.2 | 0.7 | 18.0 |
| B.1.3 | 0.7 | 18.0 |
| B.2 | 1.8 | 20.2 |
| B.2.1 | 1.0 | 11.6 |
| B.3 | 1.8 | 20.2 |

Segments: A.1.1 construction and real-estate development including land; A.1.2 civil works; A.1.3 other specialised
financing; A.2.1 large companies; A.2.2 SMEs; A.2.3 sole proprietors; B.1.1 primary residence with LTV at most 80%;
B.1.2 primary residence with LTV above 80%; B.1.3 residence other than the primary one; B.2 consumer credit; B.2.1 of which
credit cards; B.3 other purposes.

## Point 140: doubtful by reason of arrears

`annex9_doubtful_coverage(segment, days_past_due)`, percent of the uncovered amount by the age of the oldest past-due amount.
Columns: more than 90 days to 6 months; 6 to 9; 9 to 12; 12 to 15; 15 to 18; 18 to 21; more than 21 months.

| Segment | >90d-6m | 6-9m | 9-12m | 12-15m | 15-18m | 18-21m | >21m |
|---|---|---|---|---|---|---|---|
| A.1.1 | 70 | 75 | 85 | 90 | 95 | 100 | 100 |
| A.1.2 | 55 | 65 | 70 | 80 | 95 | 100 | 100 |
| A.1.3 | 55 | 65 | 75 | 90 | 95 | 100 | 100 |
| A.2.1 | 55 | 65 | 75 | 90 | 95 | 100 | 100 |
| A.2.2 | 65 | 70 | 75 | 85 | 90 | 95 | 100 |
| A.2.3 | 35 | 45 | 60 | 65 | 80 | 95 | 100 |
| B.1.1 | 45 | 50 | 65 | 70 | 85 | 95 | 100 |
| B.1.2 | 45 | 50 | 65 | 70 | 85 | 95 | 100 |
| B.1.3 | 45 | 50 | 65 | 70 | 85 | 95 | 100 |
| B.2 | 55 | 65 | 80 | 85 | 95 | 100 | 100 |
| B.2.1 | 55 | 65 | 80 | 85 | 95 | 100 | 100 |
| B.3 | 55 | 65 | 80 | 85 | 95 | 100 | 100 |

The dossier prints one row for B.1.1, B.1.2 and B.1.3 (housing) and one for B.2 with B.2.1 (cards); B.3 has the same values as
B.2. The point counts months; the engine converts them once at 30.4375 days per month, rounded to the nearest day (Contraste's
convention, which a bank's own day count overrides in the caller), giving the day buckets (90, 183], (183, 274], (274, 365],
(365, 457], (457, 548], (548, 639] and above 639, upper bounds inclusive. An age of 90 days or fewer is not a doubtful
account by arrears and raises.

A doubtful account for reasons other than arrears ("riesgo dudoso por razones distintas de la morosidad") has no transcribed
coverage: `annex9_coverage` raises when `days_past_due` is missing, and the caller reports the account as not evaluated rather
than with a placeholder number.

## Limits

- The engine computes; it makes no compliance claim. Annex 9 is a supervisory alternative to internal methodologies, which an
  entity may choose or compare against by segment (point 65 as restated in Circular 1/2025).
- Point numbers (144, 140) are as of Circular 6/2021. Whether they shifted after Circular 1/2025 of 19 December (BOE-A-2025-26847,
  country-risk coverage from 2026-06-30) is UNVERIFIED in the dossier.
- Not transcribed (to be fetched from the Banco de Espana consolidated text before use): the collateral haircut table for
  effective guarantees, the treatment of doubtful risk for reasons other than arrears, the country-risk tables revised by
  Circular 1/2025, and point 172 (the discount on foreclosed real estate: finished housing 25%; offices, premises and
  multi-purpose warehouses 30%; other finished buildings 32%; urban and developable land 35%; other real estate 40%), which the
  module does not use.
