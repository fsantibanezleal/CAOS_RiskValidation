# 05. Annex 1: CLP and UF under RAN 21-13

Module `riskvalidation.irrbb.eve_nii` (the `cmf_pair`, `annex1_*`, `indexation_charge` and `prioritised_bank` functions). Source:
dossier 03 sections F.1 and F.2, RAN 21-13 Annex 1 "Determinacion de la exposicion al riesgo de mercado en el libro de banca"
(Circular 2.270 of 2020-09-11 and 2.365 of 2025-07-08), pages "Anexo 1 Hoja 1" to "Hoja 12" read from page images,
https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf. BCCh methodology note on the UF:
https://si3.bcentral.cl/estadisticas/Principal1/Metodologias/EMF/UF.pdf.

## What Annex 1 defines

Two standard measures by significant currency (more than 5 percent of banking-book assets or liabilities): $\Delta NII$ (constant
balance, repricing flows up to one year, two parallel shocks) and $\Delta EVE$ (run-off balance, all repricing flows, six shocks).
The flows in indexed local currency are treated as a single currency, called UR (CLF, the UF-indexed CLP): "los flujos en moneda
local reajustable son tratados como una sola moneda, que denominaremos UR". The shock sizes are in `shock_table("cmf-annex1")`
(page 02). A UF amount is a CLF amount that the caller converts to the reporting currency with the UF value of the date.

## The CLP and UF pair and the 50 percent offset

For a shock $i$ the CLP and CLF deltas are combined by `cmf_pair(delta_clp, delta_clf, offset=True, share=0.5)`:

$$\Delta_{i,(CLP+CLF)}=\begin{cases}\max\{\Delta_{CLP};0\}+\max\{\Delta_{CLF};0\} & \text{same sign, either zero, or no offset}\\
\max\{0.5\,\Delta_{CLP}+\Delta_{CLF};\,0\} & \text{opposite signs and } \Delta_{CLP}\le0\\
\max\{\Delta_{CLP}+0.5\,\Delta_{CLF};\,0\} & \text{opposite signs and } \Delta_{CLP}>0\end{cases}$$

Symbols: $\Delta_{CLP},\Delta_{CLF}$ the loss-positive deltas of the two legs in the reporting currency, 0.5 the offsetting share. The
50 percent applies to the gain leg only, on both sign branches. The dossier's "else" branch gives the same value as the same-sign
branch when a leg is zero, so the engine sends zero legs to the first branch. With `offset=False` the positive parts add and the
result is never less than with the offset.

Golden (the formula of dossier 03 F.2 evaluated by hand): (10, 20) gives 30; (-10, -20) gives 0; (30, -20) gives 20; (-20, 30)
gives 20; (10, -40) gives 0; (-40, 10) gives 0; (0, 25) gives 25; (25, 0) gives 25.

## Delta NII and delta EVE of the bank

$$\Delta NII=\max_{i\in\{1,2\}}\Big\{\sum_{c\notin\{CLF,CLP\}}\max\{\Delta NII_{i,c};0\}+\Delta NII_{i,(CLP+CLF)}\Big\}+2.5\%\cdot|PN_{UR}|$$

$$\Delta EVE=\max_{i\in\{1,\dots,6\}}\Big\{\sum_{c\notin\{CLP,CLF\}}\max\{\Delta EVE_{i,c};0\}+\Delta EVE_{i,(CLP+CLF)}\Big\}$$

Symbols: $PN_{UR}$ the net (asset or liability) UF position when the functional currency is CLP, `other_currencies` the (shocks,
currencies) matrix of the significant currencies other than CLP and CLF. `annex1_delta_nii(nii_clp, nii_ur, pn_ur, offset,
other_currencies)` returns the used pairs for parallel-up then parallel-down, `max_pair` (the maximum before the charge), the
charge, the total and the worst shock (`None` when the maximum is zero). `annex1_delta_eve` is the same maximum over the six
shocks with no indexation charge. The charge is added once, after the maximum.

The term $2.5\%\cdot|PN_{UR}|$ is the standardised indexation (reajustabilidad) charge, an implied 2.5 percent inflation surprise:
`indexation_charge(net_uf_position, rate=0.025) = rate * abs(net_uf_position)`, independent of the sign of the position. A CPI
surprise scenario is a different quantity ($-PN_{UR}\cdot\text{surprise}/100$, loss-positive) carried by the case document, not by
this charge. The indexation risk exists because the UF is a one-month-lagged CPI accrual (dossier 03 F.1): NII in Chile is the
interest margin plus the indexation ("reajustes") margin.

Golden: `annex1_delta_nii([30, -10], [-20, 5], -200)`: pairs `[20, 0]`, `max_pair 20`, charge $0.025\cdot200=5$, total `25`, worst shock
parallel-up. `annex1_delta_eve([10, -20, 0, 5, -30, 40], [20, 30, -10, -2, 50, -15])`: pairs `[30, 20, 0, 4, 35, 32.5]`, `35`, worst
shock short-up. `indexation_charge(-200) = indexation_charge(200) = 5`, `indexation_charge(0) = 0`.

## The prioritised-bank tests

A bank is prioritised if any of: (a) $\Delta NII>5\%$ of Tier 1; (b) $\Delta EVE>15\%$ of Tier 1; (c) $\Delta NII$ over the 12-month
cumulative net interest and indexation margin $>18\%$. `prioritised_bank(delta_nii, delta_eve, tier1, margin_12m)` returns the
three tests in this order (`nii-tier1`, `eve-tier1`, `nii-margin`), each with its measure, base, ratio, threshold and a strict
`breached`. Golden: $\Delta NII=6$, $\Delta EVE=10$, Tier 1 100, margin 40: ratios `0.06` (breached), `0.10`, `0.15`; prioritised.
Then a detailed review follows and capital may be required under art. 66 quinquies; internal IRRBB models need no supervisory
approval but are examined in the IAPE review (dossier 03 F.2).

## Limits

- The Hoja 9 table was read from page images (dossier 03 F.2); the post-shock floor reading for UR (`-2%`) is UNVERIFIED (page 02).
- Board limits ($\Delta NII$ as a percentage of the net interest and indexation margin, $\Delta EVE$ as a percentage of Tier 1) are set
  annually by the bank and are inputs, not constants of the engine.
- The Annex 1 treatment of options (treatment V, with a 25 percent relative implied-volatility increase) is not implemented:
  `automatic_option_addon` takes the changes in fair value as given.
- The engine states no compliance claim about a bank's position.
