# 05. Floors and supervisory CCFs

Module `riskvalidation.lgd_ead.floors`. Sources: the Basel III final chapter CRE32 (in force 2023-01-01; the Basel II floors are
not implemented), read on the paragraph text on 2026-11-11 from the BIS chapter PDFs
(https://www.bis.org/committees/bcbs/basel-framework/81366/chapter.pdf, version effective 1 January 2022, and
https://www.bis.org/committees/bcbs/basel-framework/81431/chapter.pdf, version effective 1 November 2028: the floor paragraphs are
identical), and Regulation (EU) 2024/1623 (CRR3, http://data.europa.eu/eli/reg/2024/1623/oj, applying from 2025-01-01), Articles
161(4) to (7) and 164(4) to (6), read in the Official Journal text.

## LGD floors (A-IRB)

`lgd_floor(portfolio, collateral, regime="basel3")`; sovereign exposures are exempt (CRE32.16) and are not passed to it. A combination
without a cell (a corporate `qrre`, a corporate `mortgage`, a Basel `public-sector`) raises. CRE32.16: "The LGD for each corporate
exposure that is used as input into the risk weight formula and the calculation of expected loss must not be less than the parameter
floors indicated in the table below (the floors do not apply to the LGD for exposures in the sovereign asset class)". CRE32.58 gives the
retail table and CRE32.59 fixes the residential mortgage floor.

| portfolio | collateral | Basel III (`basel3`) | reference | CRR3 (`crr3`) | reference |
|---|---|---|---|---|---|
| corporate | unsecured | 0.25 | CRE32.16 | 0.25 | Article 161(4) |
| corporate | financial | 0.00 | CRE32.16 | 0.00 | Article 161(4) |
| corporate | receivables | 0.10 | CRE32.16 | 0.10 | Article 161(4) |
| corporate | real-estate | 0.10 | CRE32.16 | 0.10 | Article 161(4) |
| corporate | other-physical | 0.15 | CRE32.16 | 0.15 | Article 161(4) |
| public-sector | unsecured | none | | 0.05 | Article 161(5) |
| retail | mortgage | 0.05 | CRE32.58, fixed by CRE32.59 | 0.05 | Article 164(4), fixed by Article 164(4a)(d) |
| retail | qrre | 0.50 | CRE32.58 | 0.50 | Article 164(4) |
| retail | unsecured | 0.30 | CRE32.58 | 0.30 | Article 164(4) |
| retail | financial | 0.00 | CRE32.58 | 0.00 | Article 164(4) |
| retail | receivables | 0.10 | CRE32.58 | 0.10 | Article 164(4) |
| retail | real-estate | 0.10 | CRE32.58 | 0.10 | Article 164(4) |
| retail | other-physical | 0.15 | CRE32.58 | 0.15 | Article 164(4) |

`LGD_FLOOR_CELLS` and `LGD_FLOOR_CELLS_CRR3` hold the values and `LGD_FLOOR_SOURCES[regime][(portfolio, collateral)]` the paragraph of each. The
order of the retail secured cells (financial 0%, receivables 10%, real estate 10%, other physical 15%) is **confirmed** against the CRE32.58 table
(the 2026-11-11 reading settled the question the earlier version of this page left UNVERIFIED). The corporate floors of CRR3 are in Article 161(4)
(not 164(4)); the 5% cell of Article 161(5) is for exposures to regional governments, local authorities and public sector entities
(Article 147(2)(aa)(i) and (ii)) and has no Basel counterpart. The shared cells carry the same numbers in both texts. QRRE has no secured cell and
a mortgage no unsecured cell ("Not applicable" in both tables).

## The partly secured floor

CRE32.17 (and CRR3 Article 161(6), the same form): "The LGD floor for a partially secured exposure is calculated as a weighted average of the
unsecured LGD floor for the unsecured portion and the secured LGD floor for the secured portion",

$$\mathrm{Floor}=\mathrm{LGD}_{U,\text{floor}}\,\frac{E_U}{E(1+H_E)}+\mathrm{LGD}_{S,\text{floor}}\,\frac{E_S}{E(1+H_E)},$$

with, from CRE32.10, $E$ the current value of the exposure, $H_E$ the haircut on it where securities are lent or posted, $E_S$ the value of the
collateral received after its haircuts (CRE32.11), capped at $E(1+H_E)$, and $E_U=E(1+H_E)-E_S$. `partly_secured_lgd_floor(unsecured_floor,
secured_floor, e, e_s, h_e=0.0)` computes it. Hand cases: $0.30\cdot0.4+0.10\cdot0.6=0.18$ for $E=100$, $E_S=60$ with a 30% unsecured and a 10%
secured floor; $E=100$, $H_E=0.10$, $E_S=44$ gives $E_U=66$ and $0.30\cdot66/110+0.10\cdot44/110=0.22$. $E_S=0$ gives the unsecured floor and
$E_S\ge E(1+H_E)$ (fully secured, CRE32.17 first sentence; the cap) the secured floor. The retail rule is CRE32.59: partially secured "other retail"
exposures take the same formula; the residential mortgage floor "is fixed at 5%, irrespective of the level of collateral provided by the property"
(Article 164(4a)(d) says the same).

## The floor of one exposure

`resolve_lgd_floor(asset_class, lgd, regime, spec, defaulted=False)` returns an `LGDFloorApplied` (`applied`, `portfolio`, `collateral`, `lgd_input`,
`floor`, `lgd_used`, `binding`, `secured_share`, `source`, `reason`); `spec` is an `LGDFloorInput(collateral="unsecured", e=1.0, e_s=0.0,
h_e=0.0, portfolio=None)`. It is what `regulatory.irb.capital_requirement(..., lgd_floor=spec)` calls.

- Where the floors apply: the LGD input of the risk-weight and expected-loss formulas, for own-estimate LGD (CRE32.15; CRR3 Articles 161(4) and
  164(4) "where own estimates of LGD are used"), of a non-defaulted exposure, not sovereign. A caller using the supervisory LGD of the foundation
  approach does not pass the option.
- `corporate`, `sme` and `hvcre` take the corporate row; `residential_mortgage`, `qrre` and `other_retail` the retail rows;
  `LGDFloorInput(portfolio="public-sector")` (CRR3 only, asset class `corporate`) the Article 161(5) cell.
- A sovereign exposure is returned unfloored with the source (CRE32.16; under CRR3 the Articles name only corporate and retail classes). A defaulted
  exposure is returned unfloored with the source (CRE31.7 and CRE31.20; CRR3 Article 158(5)): the floor paragraphs name the non-defaulted formulas
  (Articles 153(1)(iii), 154(1)(ii)) and do not mention defaulted exposures, and whether the floor binds the default-time LGD (CRE36.85) was not read:
  **UNVERIFIED**, so none is applied and the page says so.
- `bank` and `financial_institution` raise: CRE32.16 and Article 161(4) say "corporate" and how banks and financial institutions are classed in them was
  not read (**UNVERIFIED**). Any regime but `basel3` and `crr3` raises.
- Exactly at a floor the LGD is kept (`binding` False); above every floor nothing changes; a floor of 0 (financial collateral) binds nowhere.

Differences between CRR3 and Basel that are **not** implemented (UNVERIFIED or out of scope): the haircuts $H_c$ and $H_{fx}$ of Article 230 against
CRE32.10 to CRE32.12 (the Article 230 values were not read: the caller supplies $E_S$ after its haircuts); the national higher floors of Article 164(6)
and (7) with their six-month transition; the mapping of retail purchased receivables (no row in Table 1); a transitional phase-in of the floors (not
searched); Article 161(7) (the own unsecured LGD as $\mathrm{LGD}_U$ when recoveries on the collateral cannot be modelled).

Goldens (`tests/regulatory/test_irb_lgd_floors.py`, `tests/lgd_ead/test_lgd_floors_partly_secured_crr3.py`): a corporate LGD of 10% takes 25% and its risk
weight is the printed CRE99 column of LGD 40% times 25/40 (K is linear in the LGD); a QRRE LGD of 30% takes 50% and reproduces the printed QRRE 50% column;
a mortgage LGD of 2% takes 5%; the partly secured other-retail blend above gives 0.18; the CRR3 public sector cell gives 0.05.

## Applying a floor

`apply_lgd_floor(lgd, floor)` and `apply_ead_floor(ead, floor)` return a `FloorBite`: `applied = max(estimate, floor)`,
`binding` (true exactly where the floor raised the estimate), `share_binding`, `mean_before`, `mean_after` and `uplift`
(never negative). A floor of 0 binds nowhere. Golden: `lgd = [0.20, 0.30, 0.60]`, floor 0.25 gives `[0.25, 0.30, 0.60]`,
binding `[True, False, False]`, share 1/3, means 0.366667 and 0.383333, uplift 0.016667; EAD `[650, 700]` against floors
`[680, 680]` gives `[680, 700]`, share 0.5, means 675 and 690, uplift 15.

## Supervisory CCFs and the EAD floor

`supervisory_ccf(item)`: the F-IRB CCFs equal the standardised ones (CRE32.33; CRE20.94 to CRE20.101; dossier 01 A.5 and B.2).

| item | CCF | reference |
|---|---|---|
| `direct-credit-substitutes` | 1.0 | CRE20.94 to CRE20.101 |
| `nifs-rufs-transaction-related-contingents` | 0.5 | CRE20.94 to CRE20.101 |
| `commitments` | 0.4 | CRE20.98 |
| `short-term-self-liquidating-trade-letters-of-credit` | 0.2 | CRE20.94 to CRE20.101 |
| `unconditionally-cancellable` | 0.1 | CRE20.100 |

Basel II's 75% for commitments (old CRE32.33) is not implemented. The EAD floor (own estimates) is, in the dossier's words,
"on-balance amount + 50% of off-balance exposure using the SA CCF" (CRE32.36 corporate, CRE32.64 retail; sovereigns exempt):

$$\mathrm{EAD}_{\text{floor}}=E+0.5\cdot\mathrm{CCF}_{SA}\,\max(L-E,\,0),$$

Symbols: $E$ drawn, $L$ limit, $\mathrm{CCF}_{SA}$ the standardised CCF of the facility. Golden: $(600, 1000, 0.40)$ gives 680.
The reading that the 50% applies to the SA-CCF-weighted off-balance exposure follows the dossier wording and is **UNVERIFIED**
against the paragraph text. The floor lies between the drawn amount and the larger of the drawn amount and the limit for a CCF
and an off-balance share in [0, 1], and is nondecreasing in the CCF and the limit. An overdrawn line (drawn above the limit)
has no off-balance amount and its floor is the drawn amount.

## Limits

The engine reproduces the published cells and states no compliance claim: a floor applies to a regulatory capital calculation under the
framework the institution is subject to. The consolidated CRR was not read (the amending Regulation (EU) 2024/1623 was); the CCFs and the EAD
floor below were transcribed from dossier 01 and not re-read on the paragraph text.
