# 04. Collateral, the margin period of risk and initial margin

Module `riskvalidation.ccr.collateral`. Sources: the U9 research addendum R3 (ORE `Docs/UserGuide/methods/xva.tex`, section "Collateral (Variation
Margin) Model" and "Margin Period of Risk", tag 1.8.17.0, lines 324 to 428; CRE53.23 to CRE53.26 from
`https://www.bis.org/basel_framework/chapter/CRE/53.htm?inforce=20230101&published=20200605`) and R2 (ORE `xva.tex`, "Dynamic Initial Margin and MVA",
"Regression Approach", lines 137 to 196; `dimregressioncalculator.cpp` and `nettedexposurecalculator.cpp` at the same tag); Hull and White, "CVA and wrong
way risk", Financial Analysts Journal 68(5):58-69, 2012, equations (2) to (4).

## Credit support amount and delivery amount

With the netting-set value $NPV(t_m)$, the independent amount $IA$ (positive when received) and the thresholds $Th_{rec}$ and $Th_{pay}$ (ORE `eq:CSA`):

$$\mathrm{CSA}(t_m)=\begin{cases}\max\big(0,\ NPV+IA-Th_{rec}\big),&NPV+IA\ge0\\ \min\big(0,\ NPV+IA+Th_{pay}\big),&NPV+IA<0\end{cases}$$

`credit_support_amount(npv, independent_amount, threshold_receive, threshold_pay)`. With $NPV=0$, $Th_{rec}=0$ and $IA>0$ the amount equals $IA$: the independent
amount is an offset to the threshold. Goldens: $NPV=100$, $IA=0$, $Th_{rec}=0$ gives 100; $Th_{rec}=30$ gives 70; $NPV=-50$, $Th_{pay}=20$ gives $-30$; $NPV=0$,
$IA=5$ gives 5. A threshold may be infinite (no margin is called on that side).

With the current collateral $C(t_m)$, the outstanding margin calls $M(t_m)$ and the minimum transfer amount $MTA$ (ORE `eq:DA`):

$$\Delta=\mathrm{CSA}-C-M,\qquad D=\begin{cases}\Delta,&|\Delta|\ge MTA\\0,&|\Delta|<MTA\end{cases}$$

`delivery_amount(csa, collateral, outstanding_calls, mta)`. Goldens: CSA 100, $C=90$, $M=0$, $MTA=10$ gives 10; $C=92$ gives 0 (the shortfall 8 is below the
MTA); $MTA=0$ gives $\Delta$.

## The margin period of risk

ORE's close-out approach (preferred): when a loss is realised at the default time $t_d$, the last time collateral could be received is $t_d-\tau$, so the
collateral at $t_d$ is $\mathrm{CSA}(t_d-\tau)$. An auxiliary close-out grid is offset from the main grid by the MPoR: the main grid values feed the balance
(filtered by the MTA and the threshold); the close-out grid values $V(t)$ give the residual exposure (Hull and White equation 4)

$$E_C(t)=\max\big(V(t)-C(t-c),\,0\big)$$

`collateralised_exposure(value_closeout, collateral)`, positive "even if minimum transfer amounts and thresholds are zero" because of the lag, and tending to
zero with the MPoR when the threshold is zero. Golden (the ORE lagged toy example): value 10 and collateral 8 give a residual 2.

`variation_margin_exposure(values_main, values_closeout, independent_amount, threshold_receive, threshold_pay, mta)` returns `(exposure, collateral_balance)`
(P, K): the balance follows `credit_support_amount` and `delivery_amount` on the main grid, date by date from a zero balance with no outstanding calls (each call
settles at its date, the lag being the close-out grid), and the exposure is `collateralised_exposure(values_closeout, balance)`. A constant value with zero
threshold and MTA gives exposure 0. The exposure is never above the uncollateralised exposure plus the collateral posted by the bank. ORE notes that for an
uncollateralised netting set the default value is taken on the main grid; here a netting set with no CSA (both thresholds infinite) returns
$\max(\text{values\_main},0)$ and a zero balance.

The close-out value is the **sticky date** one: the market evolves from the default date to the close-out date and the portfolio does not age, so a swap with no
flow in the window needs no exposure spike (ORE's "more aggressive" option; the "actual date" option, with the flows of the window and its spikes, is "more
conservative" and is not implemented). The legacy lagged approach of ORE is not implemented either.

CRE53 floors: `mpor_floor(repo_only=False, n_trades=0, illiquid=False, disputes_over_two=False)` returns business days. CRE53.24: 5 for netting sets of only
repo-style transactions and 10 for all others (daily re-margining and mark-to-market); 20 when the number of trades exceeds 5000 at any point during a quarter,
or the collateral is illiquid, or an OTC derivative cannot be easily replaced (the `illiquid` flag covers the last two). CRE53.25: more than two margin call
disputes over the previous two quarters that lasted longer than the applicable MPoR force an MPoR at least double the supervisory floor (the function doubles the
floor that applies). Goldens: `mpor_floor()` is 10, `repo_only=True` is 5, `n_trades=5001` is 20, `disputes_over_two=True` is 20.
`mpor_with_remargining(floor, remargining_days)` is CRE53.26, $F+N-1$ for re-margining every $N$ days: $(10,1)$ gives 10 and $(10,5)$ gives 14.

## The dynamic initial margin

ORE's regression approach (addendum R2). With the netting set value $NPV$, the MPoR $\Delta$ and the cash flows $F$ of the period:

$$X(t)=NPV(t+\Delta)+F(t,t+\Delta)-NPV(t),\qquad V(t)=E_t[X^2]-E_t^2[X],\quad S=\sqrt{V},\quad \mathrm{DIM}(t)=\alpha\,S(t)$$

$E_t[X]$ is the sample mean over the paths; $E_t[X^2]$ is the regression of $X^2$ across the Monte Carlo samples on a polynomial $c_0+c_1x+\dots+c_nx^n$ of a market
regressor $x$ ("in the spirit of the Longstaff Schwartz method"; the default regressor is the netting set value, as in the ORE code); order 0 is the sample variance
(one DIM per time step). The confidence factor is $\alpha=\Phi^{-1}(\text{quantile})$ and the MPoR scaling of `dimregressioncalculator.cpp` is
$\sqrt{\text{horizonCalendarDays}/\text{mporCalendarDays}}$, both in **calendar** days (10 business days are 14 calendar days, 20 are 28):

$$\mathrm{DIM}=\Phi^{-1}(\text{quantile})\,S\,\sqrt{\frac{\text{horizon\_calendar\_days}}{\text{mpor\_calendar\_days}}}$$

`dynamic_initial_margin(values_now, values_mpor, flows, quantile, horizon_calendar_days, mpor_calendar_days, order=2, regressor=None)` returns (P,) money, zero
where $S$ is zero and non-negative (so the quantile must be at least 0.5). Golden: order 0 and $X$ samples `[-2, -1, 1, 2]` (zero flows, `values_now` zero),
quantile 0.99, horizon equal to MPoR give $S=\sqrt{2.5}=1.581139$ and $\mathrm{DIM}=2.326348\times1.581139=3.678279$ (the guide's rounded factor 2.33 gives
3.684053). A quadratic conditional variance, $X=(0.02+0.01s)z$ with the regressor $s$, is recovered by order 2 to within 5 percent on average over 40,000 paths.

How the DIM enters the exposure (`nettedexposurecalculator.cpp`): `apply_initial_margin(exposure_before_im, dim)` is

$$\max(\text{exposure}-\mathrm{DIM},\ 0)$$

at most the exposure and unchanged for $\mathrm{DIM}=0$. IM applies only together with variation margin ("don't apply initial margin without VM"): the caller passes
the VM exposure. A negative DIM is refused.

## Limits

- **The DIM is a regression proxy, not SIMM.** The ISDA SIMM methodology (version 2.8, 2026) carries a notice (U.S. Pat. No. 10,515,410) that it "cannot be used,
  revised or distributed except solely for the purpose of a market participant's own commercial transactions or as otherwise provided for by ISDA in a written
  licensing agreement"; SIMM risk weights and formulas are not reimplemented or published here. A written answer from ISDA legal is the only authority (UNVERIFIED
  beyond the two ISDA pages read). The ORE guide itself says that up to release 12 it focussed on a relatively simple regression approach and that the regression
  needs benchmarking against dynamic parametric VaR.
- The Andersen, Pykhtin and Sokol (2017) timelines ("Rethinking the margin period of risk", J. Credit Risk 13(1), DOI 10.21314/JCR.2016.218) are not modelled: UNVERIFIED
  as a source (the SSRN download was blocked and the journal copy is paywalled; only the abstract was read). Remedies, suspension rights and the time to exercise them
  are not part of the exposure.
- The default of ORE's `horizonCalendarDays` was not read (UNVERIFIED): the horizon and the MPoR are named parameters in calendar days (14 and 14 for the
  10-business-day variant of the C12 case).
- Settlement is immediate at each main-grid date and outstanding margin calls are zero; the lag is the close-out grid. A first-MPoR adjustment (ORE's
  `firstMporCollateralAdjustment`) is not implemented.
