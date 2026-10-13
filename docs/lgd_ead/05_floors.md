# 05. Floors and supervisory CCFs

Module `riskvalidation.lgd_ead.floors`. Source: dossier 01 section A.5 (CRE32, Basel III final, in force 2023-01-01; the Basel
II floors are not implemented). Every cell is transcribed from the dossier table; the paragraph text of the consolidated
framework (https://www.bis.org/basel_framework/chapter/CRE/32.htm) was not re-read.

## LGD floors (A-IRB)

`lgd_floor(portfolio, collateral)`; sovereign exposures are exempt (CRE32.16) and are not passed to it. A combination without a
cell (a corporate `qrre`, a corporate `mortgage`) raises.

| portfolio | collateral | floor | reference |
|---|---|---|---|
| corporate | unsecured | 0.25 | CRE32.16 |
| corporate | financial | 0.00 | CRE32.16 |
| corporate | receivables | 0.10 | CRE32.16 |
| corporate | real-estate | 0.10 | CRE32.16 |
| corporate | other-physical | 0.15 | CRE32.16 |
| retail | mortgage | 0.05 | CRE32.58 to CRE32.59 |
| retail | qrre | 0.50 | CRE32.58 to CRE32.59 |
| retail | unsecured | 0.30 | CRE32.58 to CRE32.59 |
| retail | financial | 0.00 | CRE32.58 to CRE32.59 |
| retail | receivables | 0.10 | CRE32.58 to CRE32.59 |
| retail | real-estate | 0.10 | CRE32.58 to CRE32.59 |
| retail | other-physical | 0.15 | CRE32.58 to CRE32.59 |

The dossier lists the retail secured floors as "0% / 10% / 10% / 15% by collateral" without naming the order; the order
financial, receivables, real estate, other physical is read from the corporate row and is **UNVERIFIED**; the dossier cites
CRE32.58 to CRE32.59 jointly, so the split of the seven retail cells between the two paragraphs is not asserted.

## The partly secured floor

CRE32.17 (dossier 01 A.5): the floor of a partly secured exposure is not a cell. The caller applies `lgd_floor` to each part
and combines them with the formula,

$$\mathrm{Floor}=\mathrm{LGD}_{U,\text{floor}}\,\frac{E_U}{E(1+H_E)}+\mathrm{LGD}_{S,\text{floor}}\,\frac{E_S}{E(1+H_E)},$$

Symbols: $E_U$ the unsecured part and $E_S$ the secured part of the exposure, with $E_S$ capped at $E(1+H_E)$ and
$E_U=E(1+H_E)-E_S$, $H_E$ the exposure haircut. Example: corporate unsecured 0.25 on 40% of the exposure and receivables 0.10
on 60% gives $0.25\cdot0.4+0.10\cdot0.6=0.16$. The engine has no function for the blend (the contract leaves it to the
caller); the formula is the one printed in dossier 01.

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

The engine reproduces the dossier cells and states no compliance claim: a floor applies to a regulatory capital calculation
under the framework the institution is subject to, and the CRR3 text for the EU was not compared cell by cell here.
