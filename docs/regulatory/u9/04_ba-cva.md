# 04. BA-CVA (MAR50)

Module `riskvalidation.regulatory.ba_cva`. Sources: dossier 01 section E.1 and M.1, the research addendum R8 (no verified numeric example) and the contract
`engine-regulatory-api` 1.0.0. Primary text: BCBS MAR50, in force 2023-01-01, published 2020-07-08
https://www.bis.org/committees/bcbs/basel-framework/standard/mar/50/inforce/2023-01-01/published/2020-07-08.

## Materiality threshold

A bank with aggregate non-centrally-cleared derivative notional of EUR 100 billion or less may set CVA capital equal to 100 percent of its CCR capital,
without hedge recognition (MAR50.9). `materiality_threshold_applies(aggregate_notional_eur)` is true at or below 1e11 and false above.

## The reduced version (MAR50.14 to MAR50.16)

$$\text{capital}=DS\cdot K_{reduced},\quad DS=0.65,\qquad K_{reduced}=\sqrt{\Big(\rho\sum_c SCVA_c\Big)^2+(1-\rho^2)\sum_c SCVA_c^2},\quad\rho=50\%$$

$$SCVA_c=\frac1\alpha\,RW_c\sum_{NS}M_{NS}\,EAD_{NS}\,DF_{NS},\quad\alpha=1.4,\qquad DF_{NS}=\frac{1-e^{-0.05\,M_{NS}}}{0.05\,M_{NS}}\ \text{(1 for banks using the IMM)}$$

Symbols: $c$ a counterparty, $NS$ a netting set of the counterparty, $M_{NS}$ its effective maturity in years, $EAD_{NS}$ its exposure at default in one
currency, $RW_c$ the counterparty risk weight (below), $DF_{NS}$ the supervisory discount factor, $\rho$ the correlation between the counterparties' credit
spreads, $DS$ the discount scalar. Functions: `discount_factor(m, imm_bank)`, `scva(risk_weight, netting_sets, imm_bank, alpha)` (netting sets as
`(M, EAD)` pairs), `k_reduced(scvas, rho)` and `capital(k, ds)`.

## Counterparty risk weights (MAR50.16 Table 1)

`sector_risk_weight(sector, investment_grade)` returns a `RegulatoryParameter` (paragraph `MAR50.16`, table `MAR50.16 Table 1`, in force 2023-01-01). The
module data `SECTOR_RISK_WEIGHTS` holds the eight sector rows, compared cell by cell with the fixture's table:

| Sector label | Investment grade | High yield or not rated |
|---|---|---|
| `sovereigns-central-banks-mdbs` (sovereigns, central banks, MDBs) | 0.5% | 2.0% |
| `local-government-public` (local government, government-backed non-financials, education, public administration) | 1.0% | 4.0% |
| `financials` | 5.0% | 12.0% |
| `basic-materials-energy-industrials` (basic materials, energy, industrials, agriculture, manufacturing, mining) | 3.0% | 7.0% |
| `consumer-transport-administrative` (consumer goods and services, transport, administrative) | 3.0% | 8.5% |
| `technology-telecoms` | 2.0% | 5.5% |
| `health-care-utilities-professional` (health care, utilities, professional) | 1.5% | 5.0% |
| `other` | 5.0% | 12.0% |

## The full version (MAR50.20 to MAR50.25)

$$K_{full}=\beta K_{reduced}+(1-\beta)K_{hedged},\quad\beta=0.25,\qquad \text{capital}=0.65\,K_{full}$$

$$K_{hedged}=\sqrt{\Big(\rho\sum_c(SCVA_c-SNH_c)-IH\Big)^2+(1-\rho^2)\sum_c(SCVA_c-SNH_c)^2+\sum_c HMA_c}$$

with $SNH_c=\sum_{h\in c}r_{hc}RW_hM_h^{SN}B_h^{SN}DF_h^{SN}$, $IH=\sum_iRW_iM_i^{ind}B_i^{ind}DF_i^{ind}$ and $HMA_c=\sum_{h\in c}(1-r_{hc}^2)(RW_hM_h^{SN}B_h^{SN}DF_h^{SN})^2$
(MAR50.21 to MAR50.25). `k_hedged(scva_minus_snh, ih, hma, rho)` takes the per-counterparty $SCVA_c-SNH_c$, the index hedge and one $HMA_c$ per counterparty, and
the radicand is floored at zero; `k_full(k_reduced_value, k_hedged_value, beta)`.

## Golden values and invariants

There is no verified official numeric example (addendum R8): the tests rest on the invariants of dossier 01 M.1 and on goldens recomputed from the formulas.

- `k_reduced([10, 20], rho=1)` is 30, the sum of the SCVA (fixture entry `ba-cva-rho-one`); one counterparty gives its own SCVA; $K_{reduced}$ is at most the sum
  of the SCVA for $\rho\le1$.
- `k_full(30, 0)` is 7.5 (fixture entry `ba-cva-hedges-floor`): hedges never take $K_{full}$ below $\beta K_{reduced}$; with no hedges ($SNH=0$, $IH=0$, $HMA=0$)
  `k_hedged` equals `k_reduced`.
- `discount_factor(1)` is 0.975412 and `discount_factor(5)` is 0.884797; `scva(0.05, [(5, 100)])` is $0.05\times5\times100\times0.884797/1.4=15.799944$, linear
  in the EAD and in the risk weight.
- `capital(100)` is 65.
- The BA-CVA block of the C12 documents is reproduced from the SA-CCR EAD of the book: a financials counterparty, investment grade (0.05), $M=10$ years,
  $DF=0.786939$, so capital is $0.65\times SCVA$ (for example 31,440.35 for the MPoR 10 day variant).

## Conventions

Money is in one currency the caller states; times are years; the risk weights, $\rho$, $\beta$ and $DS$ are decimals. A missing or invalid input is refused with the
field named.

## Limits

- UNVERIFIED (addendum R8): no official BA-CVA numeric example exists in the sources read. The Reserve Bank of India draft directions of 2026 announce worked
  examples in an Annex I whose PDF is behind a human-verification page; its figures (INR 70.35 crore and INR 41.32 crore, from a secondary summary) are not golden
  values. Reading Annex I is backlog item BL-080.
- Not implemented: the computation of $SNH_c$, $IH$ and $HMA_c$ from hedge instruments (the caller passes the terms), the MAR50.18 eligible-hedge rule (single-name
  CDS, single-name contingent CDS, index CDS), the cap on the use of an index hedge, and SA-CVA (MAR50.27 onward, decision D1 of the U9 sizing).
- The engine computes and states no compliance claim.
