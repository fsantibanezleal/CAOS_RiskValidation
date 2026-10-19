# 07. The LCR and the NSFR calculators

Module `riskvalidation.regulatory.lcr_nsfr`. Sources: dossier 01 section G.1 and G.2 (BCBS Basel Framework LCR20 published 2022-12-08,
LCR30 2020-06-05, LCR40 2023-03-30, LCR99 2019-12-15; NSF20, NSF30 and NSF99 Tables 1 to 3; all in force 2019-12-15) and M.1 to M.3
(constructed tests, the paragraph and in-force date of every parameter). Contract `engine-liquidity-api` 1.0.0, decision Q1.

The LCR computed here is a Basel formula applied to aggregates with stated assumptions: it is never the bank's reported or regulatory LCR,
and that label is the caller's, required in the document. No official LCR or NSFR worked example exists (dossier 01 lines 1045 to 1046), so
none is claimed: the golden values are constructed. The engine computes and states no compliance claim. The parameters are transcribed,
never the text of the framework; the BIS notice that covers the framework pages is read by analogy and stays UNVERIFIED for those pages.

## Equations

$$LCR=\frac{HQLA}{NCO_{30d}}\ge 100\%\ (\text{LCR20.4}),\qquad NCO=O-\min(I,\,0.75\,O)\ (\text{LCR40.1}),$$

$$HQLA=L1+L2A+L2B-Adj_{15}-Adj_{40}\ (\text{LCR30.38}),$$

$$Adj_{15}=\max\!\Big(aL2B-\tfrac{15}{85}(aL1+aL2A),\ aL2B-\tfrac{15}{60}\,aL1,\ 0\Big),\quad
Adj_{40}=\max\!\Big(aL2A+aL2B-Adj_{15}-\tfrac23\,aL1,\ 0\Big)\ (\text{LCR30.39}),$$

with `aL` the adjusted amounts after unwinding short-term SFTs and collateral swaps maturing within 30 days (LCR30.37), which default to the
unadjusted ones. Level 1 is unlimited; Level 2 is at most 40 percent of HQLA (LCR30.31) and Level 2B at most 15 percent (LCR30.33), both
after haircuts (LCR30.34); adjusted Level 2 is at most two-thirds of adjusted Level 1 (LCR30.35). `l2b_share = (L2B - Adj15) / HQLA` and
`l2_share = (L2A + L2B - Adj15 - Adj40) / HQLA`, both 0 for an empty stock.

$$NSFR=\frac{ASF}{RSF}\ge 100\%\ (\text{NSF20.2}),\qquad ASF=\sum_j f^{ASF}_jL_j,\qquad
RSF=\sum_i f^{RSF}_iA_i+\sum_k f^{RSF}_kOBS_k+r\,D ,$$

with `D` the gross derivative liabilities and `r` the add-on rate of NSF30.8 ("5% to 20% of derivative liabilities"): not a fixed factor, a
caller-supplied rate within `[0.05, 0.20]`, never a default (the in-force value is the supervisor's); `nsfr_rsf` raises `ValueError`
naming `derivative_liabilities_addon` for a missing or out-of-range rate when `D > 0`. The same holds for the trade finance contingent
outflows of the LCR (0 to 5 percent at national discretion): a rate in `rates`, under the key `trade_finance`, never a default.

## The parameter tables (module data)

`LCR_OUTFLOW_RATES` (30 keys: retail stable with insurance criteria 0.03, retail stable 0.05, less stable 0.10, retail term over 30 days 0,
small business 0.05 and 0.10, operational 0.25 with insured portion 0.05, cooperative network 0.25, non-financial corporates 0.40 and 0.20
fully insured, other legal entities 1.00, secured funding with a central bank or Level 1 0, Level 2A 0.15, sovereign, MDB or PSE
counterparty 0.25, eligible Level 2B RMBS 0.25, other Level 2B 0.50, other secured 1.00, non-Level-1 collateral valuation 0.20, the undrawn
committed facilities by counterparty, net derivative and other contractual outflows 1.00), `LCR_INFLOW_RATES` (11 keys), `HQLA_HAIRCUTS` (6
keys: Level 1 0, Level 2A 0.15, qualifying RMBS 0.25, corporate debt, common equity and sovereign BBB 0.50), `NSFR_ASF_FACTORS` (9 keys) and
`NSFR_RSF_FACTORS` (19 keys, with `undrawn_facilities_obs` 0.05 of the undrawn amount). Each key has a reference string naming its
paragraph or table (`*_REFERENCES`) and the in-force date `IN_FORCE = "2019-12-15"` is a module constant. The categories are inputs the
caller maps to (the C16 mapping table, decision Q13); the engine does not map customers.

## Golden values (constructed, dossier 01 M.1)

| Call | Result |
|---|---|
| `hqla_stock(100, 20, 30)` | `adj_15 = 8.823529`, `adj_40 = 0`, `hqla = 141.176471`, `l2b_share = 0.15` |
| `hqla_stock(100, 80, 0)` | `adj_15 = 0`, `adj_40 = 13.333333`, `hqla = 166.666667`, `l2_share = 0.40` |
| `hqla_stock(100, 0, 0)` | `hqla = 100` |
| `lcr_outflows({"retail_stable": 1000})` | `total = 50` |
| `lcr_outflows` on less stable 100, non-financial corporate 100, other legal entity 10 | `10 + 40 + 10 = 60` |
| `lcr_inflows({"retail_and_nonfinancial_wholesale": 100})`; `{"secured_lending_level2a": 100}` | `50`; `15` |
| `net_cash_outflows(100, 90)`; `(100, 50)` | counted 75, net 25, cap binds; counted 50, net 50 |
| `lcr_ratio(100, 50)`; `lcr_ratio(100, 0)` | `2.0`; raises |
| `nsfr_asf({"regulatory_capital": 10, "stable_retail_deposits": 100})` | `10 + 95 = 105` |
| `nsfr_rsf({"unencumbered_level_2b": 100})`, with OBS 100, with derivatives 200 at 0.10 | `50`; `55`; `75`; a rate of 0.25 raises |
| `nsfr_ratio(105, 50)` | `2.1` |

The tests recompute `HQLA` from the formulas with the caps checked on a grid of balance sheets, and every table key against a copy of the
dossier table.

## Limits

- A ratio below 1.0 under a stress scenario is not a breach (dossier 01 J.3); the 100 percent minimum applies in normal times.
- `lcr_ratio` and `nsfr_ratio` refuse a non-positive denominator: the ratio is undefined, never infinite.
- The NSFR tables are the selection the dossier prints, not the full NSF30 text; a category outside the table raises `KeyError` naming it,
  and the caller supplies the supervisor's factor through `factors` for a national option.
- The BCBS parameters are the Basel standard; a national implementation (the CMF in Chile, the US agencies for a call-report bank) can
  differ, and the engine states none.

## Sources

- BCBS Basel Framework, LCR standard (LCR20, LCR30, LCR40, LCR99): https://www.bis.org/committees/bcbs/basel-framework/standard/lcr/
- BCBS Basel Framework, NSF99 Tables 1 to 3:
  https://www.bis.org/committees/bcbs/basel-framework/standard/nsf/99/inforce/2019-12-15/published/2019-12-15
- Dossier 01, section G.1 and G.2, and M.1 to M.3 (the Contraste initiative, 2026-10-04).
