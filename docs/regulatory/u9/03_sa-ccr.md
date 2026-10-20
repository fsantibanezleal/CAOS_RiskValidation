# 03. SA-CCR (CRE52)

Module `riskvalidation.regulatory.sa_ccr`. Sources: dossier 01 section E.2 (formulas and golden values), M.1 and M.3 item 2, the contract
`engine-regulatory-api` 1.0.0 and the BIS worked examples CRE99.20 to CRE99.97 (USD thousands). Primary text: BCBS CRE52, in force 2023-01-01,
published 2020-06-05
https://www.bis.org/committees/bcbs/basel-framework/standard/cre/52/inforce/2023-01-01/published/2020-06-05. The CRE99 trade inputs were
transcribed by U9-S from the BIS Basel Framework page CRE99 (in force 2023-01-01, fetched 2026-10-19, https://www.bis.org/basel_framework/chapter/CRE/99.htm)
into the `trades` of the five examples of the regulatory-calculators fixture, paragraph numbers kept.

## The exposure at default

$$EAD = \alpha\,(RC + PFE),\quad \alpha = 1.4\quad(\text{CRE52.1}),\qquad PFE = \text{multiplier}\cdot AddOn^{\text{aggregate}}\quad(\text{CRE52.20})$$

$$RC_{\text{unmargined}}=\max\{V-C,\,0\}\ (\text{CRE52.10}),\qquad RC_{\text{margined}}=\max\{V-C,\ TH+MTA-NICA,\ 0\}\ (\text{CRE52.18})$$

$$\text{multiplier}=\min\Big\{1,\ \text{Floor}+(1-\text{Floor})\exp\Big(\frac{V-C}{2\,(1-\text{Floor})\,AddOn^{\text{aggregate}}}\Big)\Big\},\quad \text{Floor}=5\%\ (\text{CRE52.23})$$

$AddOn^{\text{aggregate}}=\sum_a AddOn^{(a)}$ over interest rate, FX, credit, equity and commodity (CRE52.24 to CRE52.25).

Symbols: $V$ the netting set value, $C$ the net collateral, $TH$ the threshold, $MTA$ the minimum transfer amount, $NICA$ the net independent collateral
amount, all in one currency the caller states. `replacement_cost(v, c, th, mta, nica, margined)`, `multiplier(v_minus_c, addon_aggregate, floor)` and
`netting_set_ead(v, c, addon_by_class, margined, th, mta, nica, alpha)` return $RC$, the multiplier and an `SaCcrResult(rc, addon_aggregate,
multiplier, pfe, ead, alpha)`. The multiplier is 1 for $V-C\ge 0$ and, as the contract states, for a zero add-on.

## Duration, delta, maturity factor

$$SD=\frac{e^{-0.05\,S}-e^{-0.05\,E}}{0.05}\quad(\text{CRE52.34}),\qquad MF^{\text{unmargined}}=\sqrt{\min\{M,1\}},\quad MF^{\text{margined}}=\tfrac32\sqrt{\tfrac{MPoR}{250}}\quad(\text{CRE52.48, CRE52.52})$$

Symbols: $S$ and $E$ the start and end of the trade in years (a started trade has $S=0$, the caller's), $M$ the residual maturity in years, MPoR in
business days with 250 to the year. The MPoR floor of 10 business days (CRE52.50) is the caller's: `maturity_factor` does not raise a shorter MPoR.

Supervisory delta (CRE52.38 to CRE52.41): $\pm 1$ for a linear trade (a payer swap +1, a receiver -1; for credit, protection bought +1 and sold -1, the signs the printed
add-ons 106, -280 and 168 of CRE99 netting set 2 imply); options by bought or sold call or put, with $d=\dfrac{\ln(P/K)+\tfrac12\sigma^2T}{\sigma\sqrt T}$:

$$\Delta_{\text{bought call}}=\Phi(d),\quad \Delta_{\text{sold call}}=-\Phi(d),\quad \Delta_{\text{bought put}}=-\Phi(-d),\quad \Delta_{\text{sold put}}=\Phi(-d)$$

CDO tranches: $\pm\dfrac{15}{(1+14A)(1+14D)}$ with $0\le A<D\le 1$ (long positive, short negative). `supervisory_delta_option`, `supervisory_delta_tranche`.

## Interest rate (CRE52.57)

Maturity buckets by the end date: B1 below 1 year, B2 from 1 to 5 years inclusive, B3 above 5 (`maturity_bucket`). Per trade the adjusted notional is
$notional\times SD$ and the effective notional $\delta\times\text{adjusted notional}\times MF$. Per hedging set (a currency), with $D^{B1},D^{B2},D^{B3}$ the
sums of the effective notionals per bucket:

$$EN_{HS}=\sqrt{(D^{B1})^2+(D^{B2})^2+(D^{B3})^2+1.4\,D^{B1}D^{B2}+1.4\,D^{B2}D^{B3}+0.6\,D^{B1}D^{B3}}\ \ \text{or}\ \ |D^{B1}|+|D^{B2}|+|D^{B3}|\ \text{(no offset)}$$

$$AddOn_{HS}=0.005\,EN_{HS},\qquad AddOn^{IR}=\sum_{HS}AddOn_{HS}$$

`ir_addon(trades, mpor_business_days, offset)` takes `IrTrade(hedging_set, start, end, notional, delta)` and returns an `IrAddOn` whose trades carry the
duration, adjusted notional, maturity factor, bucket and effective notional, whose hedging sets carry `d_b1`, `d_b2`, `d_b3`, the effective notional, the
supervisory factor and the add-on, and whose `addon` is the sum: every intermediate of the `sa_ccr` block of the `c12-outputs` document (the four swaps of
the C12 book, margined with MPoR 10 and 20 days and unmargined) is a field, and the tests reproduce that block field by field. `ir_effective_notional` is the
formula alone; the offset value never exceeds the no-offset value. Two opposite swaps of equal terms net to an add-on of 0; hedging sets of two currencies
sum.

## FX, credit, equity, commodity

FX (CRE52.59): $AddOn^{FX}=\sum_{\text{pairs}}0.04\,|EN_{\text{pair}}|$ (`fx_addon`, the signed effective notional per currency pair in the reporting
currency). Credit and equity (CRE52.61, CRE52.66) and commodity per hedging set (CRE52.70), with one correlation per entity:

$$AddOn=\Big[\Big(\sum_e\rho_e\,A_e\Big)^2+\sum_e(1-\rho_e^2)\,A_e^2\Big]^{1/2},\qquad A_e=SF_e\,EN_e$$

`entity_aggregate(addons, rhos)` is this formula with one $\rho_e$ per entity (the same length, else `ValueError`), because credit and equity aggregate single
names (0.5) and indices (0.8) in one formula (CRE99 netting set 2 uses 0.5 and 0.8). `credit_addon`, `equity_addon` and `commodity_addon` apply the CRE52.72
Table 2 factors and correlations; `commodity_addon` sums the entity aggregate over the hedging sets.
`credit_addon_terms(entities)` and `equity_addon_terms(entities)` return `AddonTerms(systematic, idiosyncratic, addon)`: the systematic term $(\sum_e\rho_e A_e)^2$,
the idiosyncratic term $\sum_e(1-\rho_e^2)A_e^2$ and the add-on $\sqrt{\text{systematic}+\text{idiosyncratic}}$; `credit_addon` and `equity_addon` return the `addon` field of
these, and `entity_aggregate_terms(addons, rhos)` is the same split for explicit add-ons (so no consumer recomputes the two terms).
The trade effective notionals that feed them are `credit_trade_effective_notional(delta, notional, start_years, end_years, mpor_business_days=None)`
$=\delta\,N\,SD(S,E)\,MF(E)$ (CRE52.34, CRE52.52) and `commodity_trade_effective_notional(delta, adjusted_notional, maturity_years, mpor_business_days=None)`
$=\delta\,N_{adj}\,MF(M)$ (CRE52.52); they refuse what `supervisory_duration` and `maturity_factor` refuse, and a non-finite delta or notional. The factors and correlations are module data
`SUPERVISORY_PARAMETERS` (a `RegulatoryParameter` per cell with paragraph `CRE52.72`, in force 2023-01-01), compared cell by cell with the fixture's table:
IR 0.5 percent (option volatility 50 percent); FX 4.0 percent (15 percent); credit single name AAA 0.38, AA 0.38, A 0.42, BBB 0.54, BB 1.06, B 1.6, CCC
6.0 percent with correlation 50 percent and volatility 100 percent; credit index IG 0.38 and SG 1.06 percent with correlation 80 percent and volatility 80
percent; equity single name 32 percent (50 percent, 120 percent) and index 20 percent (80 percent, 75 percent); commodity electricity 40 percent (40
percent, 150 percent) and oil and gas, metals, agricultural, other 18 percent (40 percent, 70 percent).

## Golden values: the five CRE99 netting sets

The trade-level path starts from the trades of the BIS page and carries every intermediate unrounded (CRE99.20). `round(ead)` equals the published integer
exactly (dossier 01 M.3 item 2); every intermediate below is a fixture value, reproduced to 1e-9 relative.

| Netting set | Trades (USD thousands) | Intermediates (unrounded) | EAD | Published |
|---|---|---|---|---|
| 1, IR swaps and a swaption, unmargined | payer 10,000 for 10 years, receiver 10,000 for 4 years, EUR bought put swaption 5,000 on a 10 year swap starting in 1 year | RC 60; $EN_{USD}$ 59,269.96; $EN_{EUR}$ 10,082.91; AddOn 296.3498 + 50.4146 = 346.764386; swaption delta -0.2693952 | 569.470141 | 569 (delta -0.2694) |
| 2, credit, unmargined | CDS Firm A AA 10,000 for 3 years (buyer), Firm B BBB 10,000 for 6 years (seller), CDX.IG 10,000 for 5 years (buyer) | RC 0; add-ons 105.86194, -279.91632, 168.11140; systematic term 2,252.635; idiosyncratic term 77,344.043; AddOn 282.128832; multiplier 0.965208 | 381.238319 | 381 |
| 3, commodity forwards, unmargined | WTI long 10,000 for 0.75 years, Brent short 20,000 for 2 years, silver long 10,000 for 5 years | RC 20; energy 2,041.154273; metals 1,800; commodity 3,841.154273; multiplier 1 | 5,405.615982 | 5,406 |
| 4, sets 1 and 2 combined | the trades of sets 1 and 2 | RC 40; AddOn 346.764386 + 282.128832 = 628.893218; multiplier 1 | 936.450506 | 936 |
| 5, margined IR and commodity | the trades of sets 1 and 3, $V=80$, $C=200$, $TH=0$, $MTA=5$, $NICA=150$, MPoR 14 days | RC $\max(80-200,\ 0+5-150,\ 0)=0$; $MF=1.5\sqrt{14/250}=0.354965$; IR 123.089147; commodity 1,277.873233; aggregate 1,400.962380; multiplier 0.958123 | 1,879.212632 | 1,879 |

Printed intermediates are rounded (347, 282, 629, 1,401 and so on): the path from them holds to one unit of the published EAD. The EAD of set 4 shows why the
golden uses the unrounded add-ons: the printed rounded add-ons give $1.4\times 669=936.6$, which the printed value 936 does not follow. The multiplier of set
2 with the unrounded add-on is 0.965208 (printed 0.965); of set 5, 0.958123 (printed 0.958).

Further goldens: `supervisory_duration(0, 5)` 4.423984, `(0, 1)` 0.975412, `(1, 5)` 3.448573; `maturity_factor(0.25)` 0.5, `maturity_factor(1, 14)` 0.354965;
`ir_effective_notional(100, -100, 0)` 77.459667; `tranche(0, 0.1, long)` 6.25; `replacement_cost` 60, 0, 20 and 40 for sets 1, 5, 3 and 4.

## Conventions

Money is in one currency the caller states; the engine never converts currencies. Arrays and scalars are `float`; every function is pure and deterministic
and does not mutate its inputs. A missing or invalid input is refused with the field named.

## Limits

- UNVERIFIED (contract): the equations of the BIS CRE99 page are images and were not transcribed; the formulas are those of CRE52 and recompute every printed
  intermediate and the five EADs exactly after rounding. The swaption inputs (price 6 percent, strike 5 percent, volatility 50 percent, expiry 1 year) give the
  delta -0.2693952, printed -0.2694.
- Not implemented: the CRE52.40 shift for negative rates, FX and commodity options beyond the supervisory delta, basis and volatility transactions, the
  CRE52.50 MPoR floors by margining frequency (the caller passes the MPoR), cleared transactions (CRE54), and the sign rules of FX trades beyond the signed
  effective notional the caller passes.
- The maturity factor of an interest-rate trade uses its end date; for a swaption the contract's `IrTrade` carries no separate exercise date.
- The engine computes and states no compliance claim.
