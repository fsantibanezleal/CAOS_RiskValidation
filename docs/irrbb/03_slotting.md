# 03. Slotting

Module `riskvalidation.irrbb.slotting`. Sources: dossier 01 section H.4 (SRP31.94 to SRP31.129; BCBS SRP31
https://www.bis.org/basel_framework/chapter/SRP/31.htm), dossier 03 section F.2 (RAN 21-13 Annex 1 repeats the buckets, the
multipliers and the caps) and dossier 05 section D.4 (SMM, CPR and PSA).

## The 19 buckets

SRP31.96 Table 3 (all midpoints verified in the page text of the standard, dossier 01 H.4; the same list in dossier 03 F.2):

| Bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Range | overnight | O/N to 1M | 1M to 3M | 3M to 6M | 6M to 9M | 9M to 1Y | 1Y to 1.5Y | 1.5Y to 2Y | 2Y to 3Y | 3Y to 4Y | 4Y to 5Y | 5Y to 6Y | 6Y to 7Y | 7Y to 8Y | 8Y to 9Y | 9Y to 10Y | 10Y to 15Y | 15Y to 20Y | over 20Y |
| Midpoint (years) | 0.0028 | 0.0417 | 0.1667 | 0.375 | 0.625 | 0.875 | 1.25 | 1.75 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 | 12.5 | 17.5 | 25.0 |

`bucket_grid()` also returns the edges `0, 1/360, 1/12, 3/12, 6/12, 9/12, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, infinity`,
derived from the range labels. `bucket_of(t)` uses the intervals $(\text{lower},\text{upper}]$, time 0 in bucket 1 and every time
beyond 20 years in bucket 19. Golden: `bucket_of` of 0, 0.5, 0.75, 20, 20.5, 25 is 1, 4, 5, 18, 19, 19. `slot_cash_flows(times,
amounts)` adds each amount (assets positive, liabilities negative) to its bucket; the total is preserved and the slotting is
additive. Golden: times `[0.5, 2.5, 9.5]`, amounts `[100, -30, -70]` fill buckets 4, 9 and 16.

Limits: the upper edge of the overnight bucket as $1/360$ year is UNVERIFIED (the sources give the label "overnight" and the
midpoint 0.0028 only; the interval $(0,1/360]$ has midpoint 0.0014, and the midpoint of bucket 2 is that of $[0,1/12]$, not of
$(1/360,1/12]$).

## Non-maturity deposits (SRP31.107 to SRP31.112, Table 4)

| Category (BCBS) | Category (Annex 1) | Cap on the core share | Cap on the average maturity of the core |
|---|---|---|---|
| retail-transactional | retail-operational | 0.9 | 5.0 years |
| retail-non-transactional | retail-non-operational | 0.7 | 4.5 years |
| wholesale | wholesale | 0.5 | 4.0 years |

Stable versus non-stable is judged from 10 years of volume history in SRP31, at least 5 years of data in Annex 1; the core is the
stable part unlikely to reprice and the non-core part is slotted overnight (bucket 1). `nmd_split` puts $(1-s)B$ in bucket 1 and the
core $sB$ in the two buckets around the core maturity $m$ so that its weighted average maturity is $m$:

$$w_{lo}=\frac{t_{hi}-m}{t_{hi}-t_{lo}},\qquad w_{hi}=1-w_{lo},\qquad t_{lo}\le m<t_{hi}$$

Symbols: $B$ the balance, $s$ the core share, $m$ the core maturity, $t_{lo},t_{hi}$ the midpoints of the adjacent buckets that
bracket $m$. Golden: $B=100$, `retail-transactional`, $s=0.9$, $m=5.0$ gives 10 in bucket 1 and 45 in each of buckets 11 (4.5 y) and
12 (5.5 y).

**The caps are applied and reported.** `nmd_split` returns an `NmdSlotting` with the request, the cap and the used value of both the
core share and the core maturity (`requested_core_share`, `core_share_cap`, `core_share_used`, `requested_core_maturity_years`,
`core_maturity_cap_years`, `core_maturity_used_years`) and `cap_applied`. With `enforce_caps=True` (the standardised treatment) the
used value is $\min(\text{request},\text{cap})$ and `cap_applied` is true when either the share or the maturity was cut; a core share
or maturity above its cap is never passed through silently. With `enforce_caps=False` (an internal what-if) the request is used and
`cap_applied` is false; a document never carries that output when a request exceeds a cap. Golden: $B=100$, `wholesale`, $s=0.6$,
$m=4.0$ with `enforce_caps=True` gives `core_share_cap = 0.5`, `core_share_used = 0.5`, `cap_applied = True` and 50 in bucket 1; with
`enforce_caps=False` it gives `core_share_used = 0.6`, `cap_applied = False` and 40 in bucket 1. The Annex 1 names
`retail-operational` and `retail-non-operational` are accepted aliases. A maturity below the first midpoint or above the last
goes to that bucket alone.

## Prepayment, SMM, CPR and PSA

Prepayment (SRP31.119 Table 5, RAN 21-13 Annex 1 repeats it): $CPR_i=\min(1,\gamma_i\,CPR_0)$ with $\gamma_i=0.8,\,1.2,\,0.8,\,1.2,\,
0.8,\,1.2$ for the six shocks in the order of `SHOCKS`; the cash flows are $CF^p_i(k)=CF^s_i(k)+CPR_i\,N^p_i(k-1)$ (SRP31.121).
Term-deposit early redemption (SRP31.125 Table 6): $TDRR_i=\min(1,u_i\,TDRR_0)$ with $u_i=1.2,\,0.8,\,0.8,\,1.2,\,1.2,\,0.8$;
$CF^p_i(1)=TD\cdot TDRR_i$ (SRP31.126). Golden: `prepayment_cpr("parallel-up", 0.05) = 0.04`, `("parallel-down", 0.05) = 0.06`,
`("parallel-down", 0.9) = 1.0`; `term_deposit_redemption_rate("parallel-up", 0.05) = 0.06`, `("parallel-down", 0.05) = 0.04`.

$$SMM=1-(1-CPR)^{1/12},\qquad CPR=1-(1-SMM)^{12}$$

Golden: `smm_from_cpr(0.06) = 0.005143`, `cpr_from_smm(0.005143) = 0.060000` to `1e-6`; the two are inverse to `1e-12`.

PSA benchmark (dossier 05 D.4): 100 percent PSA is a CPR of 0.2 percent in month 1, rising 0.2 percent a month to 6 percent at
month 30 and flat thereafter; $x$ percent PSA scales the curve:

$$CPR_m=\frac{x}{100}\,\min(0.002\,m,\ 0.06)$$

Golden: 100 percent PSA gives `0.002` at month 1, `0.06` at month 30 and month 60; 150 percent PSA gives `0.06` at month 20 and
`0.09` at month 40.

## Sources

- BCBS SRP31.96, SRP31.107 to SRP31.112, SRP31.119 to SRP31.126 (dossier 01 H.4).
- CMF RAN 21-13 Annex 1 (dossier 03 F.2), https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf.
- Fuster, Lucca and Vickery (2022), Mortgage-backed securities, FRBNY Staff Report 1001,
  https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1001.pdf (dossier 05 D.4).

## Limits

- UNVERIFIED: the definition of the outstanding notional $N^p(k-1)$ in SRP31.121 (the dossier gives the formula without it; the
  cash flows $CF^p$ are the caller's slotting).
- UNVERIFIED: the treatment of the remainder of a redeemed term deposit (the dossier states the bucket-1 flow only; reading the
  rest as scaled by $1-TDRR$ in its contractual bucket is the case fixtures' assumption).
- UNVERIFIED: the SIFMA primary text of the PSA definition (consistent across several secondary sources); the upper edge of the
  overnight bucket (above).
- The engine does not calibrate a CPR: no Chilean mortgage prepayment source was checked (dossier 05 D.4).
