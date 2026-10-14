# 04. EVE and NII

Module `riskvalidation.irrbb.eve_nii`. Sources: dossier 05 sections D.1 and D.2, dossier 01 sections H.1, H.4 and H.5 (BCBS
SRP31.129, SRP31.82 to SRP31.83, DIS70.4 and DIS70.5), dossier 03 section F.2 for the NII form. A positive delta EVE or delta NII
is a **loss**; discounting is continuously compounded; the engine never converts currencies.

## Repricing gap and the earnings approximation

With $RSA_k$ and $RSL_k$ the rate-sensitive assets and liabilities slotted by next repricing date (dossier 05 D.1):

$$G_k=RSA_k-RSL_k,\qquad \Delta NII\approx\sum_{k:\,t_k<1}G_k\,\Delta r\,(1-t_k)\ \ (\text{gain-positive in the dossier})$$

`delta_nii_gap(gap, midpoints, shift_bp)` returns the **loss-positive** form $-\sum_{t_k<1}G_k\,\Delta r\,(1-t_k)$ with
$\Delta r=\mathrm{shift\_bp}/10^4$ (the textbook earnings-gap approximation on a constant balance sheet; it ignores basis risk,
optionality and new business, which is why the dynamic simulation of page 07 exists). Golden: $G_3=100$ at $t=0.1667$, shift
$+100$ bp: $-100\cdot0.01\cdot(1-0.1667)=-0.8333$ (an asset-sensitive book gains when rates rise). `repricing_gap` is antisymmetric in
its arguments and sums to total assets minus total liabilities.

## Duration gap

$$DGAP=D_A-\frac{L}{A}\,D_L,\qquad \Delta E\approx-DGAP\cdot A\cdot\Delta y$$

Symbols: $A,L$ the present values of assets and liabilities at the continuously compounded rates, $D_A,D_L$ their value-weighted
times (with continuous compounding the modified duration equals the Macaulay duration). `duration_gap` returns the first-order
`equity_change_100bp_up` $=-DGAP\cdot A\cdot0.01$ (textbook first-order approximation, dossier 05 D.2, no single primary source); a
book without assets raises, and with no liabilities the gap is $D_A$. Golden: an asset flow 100 at 2.0 y and a liability flow 80
at 1.0 y at zero rates give $A=100$, $L=80$, $D_A=2$, $D_L=1$, $DGAP=1.2$, equity change $-1.2$.

## The standardised EVE (SRP31.129)

$$DF_{i,c}(t_k)=e^{-R_{i,c}(t_k)\,t_k},\qquad EVE_{i,c}=\sum_{k=1}^{K}CF_{i,c}(k)\,DF_{i,c}(t_k)$$

$$\Delta EVE_{i,c}=\sum_{k}CF_{0,c}(k)DF_{0,c}(t_k)-\sum_{k}CF_{i,c}(k)DF_{i,c}(t_k)+KAO_{i,c}$$

$$\text{EVE risk measure}=\max_{i\in\{1,\dots,6\}}\Big\{\max\Big(0;\ \sum_{c:\,\Delta EVE_{i,c}>0}\Delta EVE_{i,c}\Big)\Big\}$$

Symbols: $CF$ the net notional repricing cash flow of bucket $k$ (assets positive, liabilities negative), $R_{i,c}$ the zero rate of
currency $c$ under shock $i$ ($i=0$ the base), $KAO$ the automatic-option add-on, only currencies with material exposure (more than 5
percent of banking-book assets or liabilities) enter. `eve`, `delta_eve` and `eve_risk_measure` implement the three lines; the EVE is
additive over items (the sum-of-parts identity the validation test checks), and a gain in one currency never offsets a loss in
another. `automatic_option_addon` is $KAO=\sum\Delta FVAO^{sold}-\sum\Delta FVAO^{bought}$ (SRP31.127). Run-off balance sheet for
EVE, constant balance sheet for NII (DIS70.4 and DIS70.5).

Golden (the three-bucket sheet): net flows $+100$ in bucket 4 (0.375 y), $-30$ in bucket 9 (2.5 y), $-70$ in bucket 16 (9.5 y) on a flat
2 percent curve: $EVE=12.828783$ (zero at zero rates). Delta EVE under the USD 2024 sizes 200/300/225 bp, no floor, flows
unchanged: parallel-up `-10.666989`, parallel-down `12.828783`, steepener `-8.944982`, flattener `6.076550`, short-up `-1.625807`,
short-down `1.702071`. `eve_risk_measure` of the rows `[[3, -1], [2, 4], [-5, -2]]` completed by three rows of zeros (six shocks) is `6` (the per-shock positive sums are 3, 6, 0, 0, 0, 0); a matrix without exactly six rows is refused, so a transposed (currencies, shocks) matrix cannot give another number silently.
`automatic_option_addon([3, 2], [1]) = 4`.

## Delta NII under a rate shock

$$\Delta NII_{i,c}=\sum_{k:\,t_k<1}CF_{c}(k)\,\big(r_{0,c}(t_k)-r_{i,c}(t_k)\big)\,(1-t_k)$$

(RAN 21-13 Annex 1 Hoja 10; dossier 03 F.2: the first six buckets; the same form serves SRP31.90's two parallel scenarios on the
constant balance sheet.) Loss-positive. Golden: $CF=+100$ in bucket 5 (0.625 y), $r_0=0.03$, $r_i=0.06$:
$100\cdot(0.03-0.06)\cdot(1-0.625)=-1.125$.

## The outlier test

$$\text{breached}\iff \frac{\text{measure}}{\text{Tier 1}}>\text{threshold},\qquad \text{threshold}=0.15$$

SRP31.82 to SRP31.83: at least one test comparing the maximum delta EVE under the six shocks with 15 percent of Tier 1; additional
tests must be at least as stringent. The inequality is strict and a nonpositive measure never breaches. Golden: measure 16 on Tier
1 100 breaches (ratio 0.16); measure 15 does not (ratio 0.15). `tier1 > 0` or `ValueError`.

## Sources

- BCBS SRP31.82 to SRP31.83, SRP31.127, SRP31.129 and DIS70.4 to DIS70.5, https://www.bis.org/basel_framework/chapter/SRP/31.htm
  and https://www.bis.org/basel_framework/chapter/DIS/70.htm (dossier 01 H.1, H.4, H.5).
- BCBS d368 (2016), https://www.bis.org/bcbs/publ/d368.htm (dossier 05 D.2).
- CMF RAN 21-13 Annex 1 Hoja 10 (dossier 03 F.2).

## Limits

The duration-gap equity change is first order and has no single primary source. The EU NII outlier test of Commission Delegated
Regulation (EU) 2024/856 Article 5 (a decline of one-year NII by more than 5 percent of Tier 1) is not implemented here; the CRD
article number of the EU EVE test is UNVERIFIED (dossier 01 H.5). The engine takes the cash flows, rates and add-ons as given:
slotting the balance sheet and valuing the options are the caller's.
