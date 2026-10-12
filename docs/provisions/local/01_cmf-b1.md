# CMF Chapter B-1: provisions for credit risk (Chile)

Module `riskvalidation.provisions.local_cl`. Source: Contraste research dossier 03, sections D.1 to D.6, read on 2026-10-04
from the CMF "Compendio de Normas Contables para Bancos" (CNCB), current consolidated PDF
(https://www.cmfchile.cl/portal/normativa/624/articles-29177_doc_pdf.pdf), base Circular N 2.243 of 2019-12-20 with later
circulars up to N 2.346 of 2024-03-06 in the page footers. Every table below is the engine's table, cell by cell, and each is
asserted against a fixture transcribed from the dossier (`tests/fixtures/provisions_local/b1_tables.json`). The engine
computes; it makes no compliance claim.

## What B-1 is and what it is not

In the local statutory accounts of a Chilean bank, the provision on loans and contingent credits is the Chapter B-1
expected loss, not IFRS 9 impairment. CNCB Chapter A-2 N 5 says that Chapter 5.5 (impairment) of IFRS 9 "no sera aplicado
respecto de las colocaciones ... ni sobre los Creditos contingentes, ya que los criterios para estos temas se definen en los
Capitulos B-1 a B-3" (footer Circular N 2.243). The CMF's stated reason is prudential. IFRS 9 impairment does apply to other
financial assets (debt instruments at amortised cost, repo receivables, instruments at fair value through other
comprehensive income). The CMF's August 2026 proposal keeps B-1 and adds internal models under one regime; no CMF document
schedules IFRS 9 impairment for loans. This module therefore computes B-1 only and no IFRS 9 expected credit loss; the two
regimes are kept apart in the engine and are compared side by side only in the documents that call both.

B-1 has two model families: N 2, individual assessment of debtors, and N 3, group assessment. Group assessment applies to
residential mortgages, consumer loans, student loans and the commercial exposures that meet both conditions of Circular N 2.313
(aggregate exposure to the counterparty below 20,000 UF, and each aggregate exposure not above 0.2% of the total group pool);
other commercial exposures are assessed individually.

The provision factor of the standard methods is

$$
PE = PI \cdot PDI
$$

All parameters are decimals in the engine: 1.0916 per cent is 0.010916. The tables below print per cent.

## Individual assessment (N 2.1, "Hoja 3")

States: Normal (A1 to A6), Subestandar (B1 to B4), Incumplimiento (C1 to C6, below). `b1_individual_category(category)`
returns `(PI, PDI, PE)`; any other category raises `KeyError`.

| Category | PI (%) | PDI (%) | PE (%) |
|---|---|---|---|
| A1 | 0.04 | 90.0 | 0.03600 |
| A2 | 0.10 | 82.5 | 0.08250 |
| A3 | 0.25 | 87.5 | 0.21875 |
| A4 | 2.00 | 87.5 | 1.75000 |
| A5 | 4.75 | 90.0 | 4.27500 |
| A6 | 10.00 | 90.0 | 9.00000 |
| B1 | 15.00 | 92.5 | 13.87500 |
| B2 | 22.00 | 92.5 | 20.35000 |
| B3 | 33.00 | 97.5 | 32.17500 |
| B4 | 45.00 | 97.5 | 43.87500 |

A1 requires at least an AA rating on the Ley 18.045 national scale (or international equivalent). The provision for a Normal
or Substandard debtor (N 2.1.3), `b1_individual_provision(eap, ea, category, guarantor_category)`:

$$
\text{Provision}_{\text{debtor}} = (EAP - EA)\cdot PI\cdot PDI + EA\cdot PI_{\text{aval}}\cdot PDI_{\text{aval}}
$$

$EAP$ is the exposure subject to provisions (loans plus contingent credits, less financial or real collateral at realisable
value) and $EA$ the guaranteed exposure; guarantor substitution never deducts a guarantee from the exposure. The State, CORFO
or FOGAPE as guarantor are assigned category A1; the other equivalences are in B-1 N 4.1 a.

### The 0.50% minimum

"El banco debe mantener un porcentaje de provision minimo de 0,50% sobre las colocaciones y creditos contingentes de la
Cartera Normal", at individual, local-consolidated and global-consolidated levels. `b1_minimum_normal_check(normal_exposure,
normal_provision)` returns the minimum amount $0.005 \times$ exposure, the top-up $\max(0, \text{minimum} - \text{provision})$
and whether it binds. Its scope is the individually assessed Normal portfolio (A1 to A6), loans plus contingent credits, and
nothing else: it is not applied to the group-assessed portfolios (their provision is the group factor), to the Substandard
portfolio or to the default portfolio, and no other function of the module applies it.

## Default portfolio (N 2.2, "Hoja 8")

Debtors with payments 90 days or more past due, forced restructuring or evident signs of default; all their credits and
100% of their contingent credits. The expected loss rate is $TPE = (E - R)/E$ (exposure minus the recoverable amounts from
collateral and, if justified, the present value of collection recoveries), and the provision is $E \cdot PP$.
`b1_default_provision(exposure, recoverable)` rounds TPE to 1e-9 before the bucket comparison; upper bounds are inclusive.

| Category | Expected loss range | Provision PP (%) |
|---|---|---|
| C1 | up to 3% | 2 |
| C2 | above 3% up to 20% | 10 |
| C3 | above 20% up to 30% | 25 |
| C4 | above 30% up to 50% | 40 |
| C5 | above 50% up to 80% | 65 |
| C6 | above 80% | 90 |

Cure conditions to leave default (all required): no obligation more than 30 days past due; no new refinancing to pay
obligations; at least one payment includes principal; two payments if partial payments in periods under six months; four
consecutive monthly instalments; no unpaid direct debts in the CMF consolidated debtor information. The engine does not
evaluate cures.

## Group assessment and the max rule (N 3)

Banks hold two internal methods for retail (direct expected-loss rate per homogeneous group; segment PI and recovery rate),
and the provision is "el mayor valor obtenido entre el respectivo metodo estandar y el metodo interno", unless the internal
method was evaluated and not objected by the CMF (Annex N 1), in which case the provision follows the internal method:

$$
\text{Prov}_{\text{group}} = \max\left(\text{Prov}^{ME},\ \text{Prov}^{MI}\right) \quad\text{unless MI has CMF "no objeccion"}
$$

`b1_group_max_rule(standard, internal, internal_not_objected)` returns `(applied, method)` with method `max`, `internal` or
`standard` (no internal method). The August 2026 CMF report states that no institution yet uses internal methods for
provisions, so in practice the provision is the maximum.

The group default definition (N 3.2) is 90 or more days past due, a loan granted to keep current an operation more than
60 days past due, or forced restructuring or partial debt forgiveness. Guarantor treatment in group portfolios (N 4.1),
`b1_group_provision(group_exposure, pe_group, guaranteed_share, pe_guarantor)`:

$$
\text{Prov} = EG\,(1 - EA/100)\,PE_{\text{group}} + EG\,(EA/100)\,PE_{\text{aval}}
$$

## Standard methods for group portfolios (N 3.1)

### Residential mortgages (3.1.1, Circular N 2.346 "Hoja 11"); also residential leasing

`b1_mortgage_pe(pvg, dpd_bucket)`. PVG is the outstanding principal over the mortgage collateral value (appraisal in UF at
origination). All values in per cent; PE is the transcribed cell, equal to PI x PDI within the table's own rounding
(5e-7 as a decimal).

| PVG bucket | Param | 0 days | 1-29 | 30-59 | 60-89 | Default |
|---|---|---|---|---|---|---|
| PVG <= 40% | PI | 1.0916 | 21.3407 | 46.0536 | 75.1614 | 100 |
| PVG <= 40% | PDI | 0.0225 | 0.0441 | 0.0482 | 0.0482 | 0.0537 |
| PVG <= 40% | PE | 0.0002 | 0.0094 | 0.0222 | 0.0362 | 0.0537 |
| 40% < PVG <= 80% | PI | 1.9158 | 27.4332 | 52.0824 | 78.9511 | 100 |
| 40% < PVG <= 80% | PDI | 2.1955 | 2.8233 | 2.9192 | 2.9192 | 3.0413 |
| 40% < PVG <= 80% | PE | 0.0421 | 0.7745 | 1.5204 | 2.3047 | 3.0413 |
| 80% < PVG <= 90% | PI | 2.5150 | 27.9300 | 52.5800 | 79.6952 | 100 |
| 80% < PVG <= 90% | PDI | 21.5527 | 21.6600 | 21.9200 | 22.1331 | 22.2310 |
| 80% < PVG <= 90% | PE | 0.5421 | 6.0496 | 11.5255 | 17.6390 | 22.2310 |
| PVG > 90% | PI | 2.7400 | 28.4300 | 53.0800 | 80.3677 | 100 |
| PVG > 90% | PDI | 27.2000 | 29.0300 | 29.5900 | 30.1558 | 30.2436 |
| PVG > 90% | PE | 0.7453 | 8.2532 | 15.7064 | 24.2355 | 30.2436 |

State-subsidy loans with state auction insurance carry a mitigation factor MP (100 for PVG <= 80%; 95 for V <= 1,000 UF and
96 for 1,000 < V <= 2,000 UF at 80% < PVG <= 90%; 84 and 89 for PVG > 90%); the engine does not apply it.

### Commercial leasing (3.1.2 a)

`b1_leasing_pe(pvb, dpd_bucket, real_estate)`; PVB is the present value of the lease over the asset value. PE is PI x PDI.

| Days past due | PI real estate (%) | PI non real estate (%) |
|---|---|---|
| 0 | 0.79 | 1.61 |
| 1-29 | 7.94 | 12.02 |
| 30-59 | 28.76 | 40.88 |
| 60-89 | 58.76 | 69.38 |
| Default | 100.00 | 100.00 |

| PVB bucket | PDI real estate (%) | PDI non real estate (%) |
|---|---|---|
| PVB <= 40% | 0.05 | 18.2 |
| 40% < PVB <= 50% | 0.05 | 57.00 |
| 50% < PVB <= 80% | 5.10 | 68.40 |
| 80% < PVB <= 90% | 23.20 | 75.10 |
| PVB > 90% | 36.20 | 78.90 |

### Student loans (3.1.2 b)

`b1_student_loan_pe(scheme, dpd_bucket)`, scheme CAE (Ley 20.027) or CORFO and other. When payment is due ("exigible"):

| Scheme | 0 | 1-29 | 30-59 | 60-89 | Default | PDI |
|---|---|---|---|---|---|---|
| CAE PI (%) | 5.2 | 37.2 | 59.0 | 72.8 | 100 | 70.9 |
| CORFO PI (%) | 2.9 | 15.0 | 43.4 | 71.9 | 100 | 70.9 |

When not yet due (`not-due`): PI CAE 41.6 and CORFO 16.5; PDI CAE 50.3 and CORFO 45.8.

### Generic commercial loans and factoring (3.1.2 c)

`b1_commercial_pe(dpd_bucket, ptvg, factoring_with_recourse)`; PTVG is the obligations over the value of real guarantees,
`None` for no guarantee.

| Days past due | PI, guarantee, PTVG <= 100% (%) | PI, guarantee, PTVG > 100% (%) | PI, no guarantee (%) |
|---|---|---|---|
| 0 | 1.86 | 2.68 | 4.91 |
| 1-29 | 11.60 | 13.45 | 22.93 |
| 30-59 | 25.33 | 26.92 | 45.30 |
| 60-89 | 41.31 | 41.31 | 61.63 |
| Default | 100.00 | 100.00 | 100.00 |

| Guarantee | PTVG bucket | PDI generic or factoring without recourse (%) | PDI factoring with recourse to the assignor (%) |
|---|---|---|---|
| With | PTVG <= 60% | 5.0 | 3.2 |
| With | 60% < PTVG <= 75% | 20.3 | 12.8 |
| With | 75% < PTVG <= 90% | 32.2 | 20.3 |
| With | 90% < PTVG | 43.0 | 27.1 |
| Without | none | 56.9 | 35.9 |

### Consumer loans (3.1.3, Circular N 2.346 of 2024-03-06)

`b1_consumer_pe(max_dpd_in_bank, has_mortgage_in_system, arrears_30_in_system_3m, product)`. Introduced by Circular N 2.346,
applicable from the January 2025 accounting close (Chapter E N 8), applied uniformly to all consumer loans and contingents of
the debtor with the bank and its Chilean subsidiaries. PI depends on the debtor's maximum days past due in the bank, whether
it had a debt 30 or more days past due anywhere in the system in the previous three months, and whether it holds a residential
mortgage in the system; defaulted debtors (90 days or more) have PI 100%.

| Max days past due in bank | With mortgage, no 30+ system arrears (%) | With mortgage, 30+ system arrears (%) | No mortgage, no 30+ system arrears (%) | No mortgage, 30+ system arrears (%) |
|---|---|---|---|---|
| 0 to 7 | 3.3 | 14.6 | 6.6 | 19.8 |
| 8 to 30 | 20.4 | 41.6 | 30.6 | 48.5 |
| 31 to 60 | 50.2 | 63.0 | 65.1 | 66.3 |
| 61 to 89 | 62.6 | 81.7 | 72.3 | 86.9 |

| PDI | Leasing and auto loans (%) | Instalment loans (%) | Cards, credit lines and other consumer (%) |
|---|---|---|---|
| With residential mortgage in system | 33.2 | 47.7 | 49.5 |
| Without residential mortgage in system | 33.2 | 56.6 | 60.3 |

## Additional provisions (N 9)

Banks may hold "provisiones adicionales" against unpredictable economic fluctuations: counter-cyclical, always general
provisions on commercial, mortgage or consumer portfolios, "y en ningun caso podran ser utilizadas para compensar
deficiencias de los modelos utilizados por el banco"; the board sets a policy with minimum and maximum limits and they are
presented as a liability. `b1_additional_provisions(amount, policy_min, policy_max)` validates the amount against the limits
and returns it as its own line. Nothing in the module adds it to a B-1 total that is compared across regimes, so a comparison
between B-1, IFRS 9 and another regime excludes it and shows it beside the total.

## Limits

- The engine computes; it makes no compliance claim. The tables are statutory parameters as transcribed from the dossier,
  read from the CMF PDF page images on 2026-10-04; a bank's own filing is the authority.
- The CMF proposal of August 2026 (new RAN Chapter 21-9 on internal methodologies, a recalibrated PI scale, collateral moved
  from exposure deduction to an LGD effect, effective January 2031 for the individual model) is a proposal "en tramite",
  not in force, and is not implemented. Circular N 2.373 of 2026-07-14 (collateral as risk mitigants, RAN Chapter 21-10) is
  also outside this module.
- Not modelled: the individual-assessment guarantor equivalences of N 4.1 a by external rating (the caller passes the
  guarantor's category), the MP mitigation factor for subsidised mortgages, the cure conditions, the split between individual
  and group assessment (decided at consolidated level once a year), and internal methods.
- Chilean loan provisions are not computed as IFRS 9 anywhere in this module.
