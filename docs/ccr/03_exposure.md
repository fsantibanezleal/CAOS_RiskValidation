# 03. Exposure measures

Module `riskvalidation.ccr.exposure`. Sources: dossier 05 C.1 (definitions verified in the ORE Methodology, `simulation.tex`, section "Exposures"),
the U9 research addendum R9 (CRE53.11 to CRE53.20 from `https://www.bis.org/basel_framework/chapter/CRE/53.htm?inforce=20230101&published=20200605`,
formulas read from the BIS SVG images) and R4 (the derived statement of the analytic EE). Paths are rows, grid points columns.

## EE, ENE and the Monte Carlo error

For a netting set with value $V(t)$ and numeraire $N(t)$ (ORE, dossier 05 C.1):

$$\mathrm{EE}(t)=E\!\left[\frac{\max(V(t),0)}{N(t)}\right],\qquad \mathrm{ENE}(t)=E\!\left[\frac{\max(-V(t),0)}{N(t)}\right]$$

`expected_exposure(values, numeraire=None)` returns `(ee, mc_se)` each (K,): the sample mean of $\max(v,0)/N$ and its standard error
$\text{std}/\sqrt P$ (sample standard deviation, `ddof=1`). `numeraire=None` is the undiscounted Basel style, $EE_B(t)=E[\max(V(t),0)]$;
`hw1f.bank_account` gives the numeraire of simulated paths. ENE is `expected_exposure(-values)`. Invariants: $EE\ge0$; $EE-ENE=E[V]$ to 1e-9; the
discounted EE is at most the undiscounted one when the numeraire is at least 1 (positive rates). Golden: the values `[[-1, 2], [3, -4]]` give EE
`[1.5, 1.0]` and ENE `[0.5, 2.0]`.

## The effective measures (CRE53.12, CRE53.13)

$$\text{Effective EE}_{t_k}=\max\big(\text{Effective EE}_{t_{k-1}},\,EE_{t_k}\big)\quad(\text{equation 2}),\qquad
\text{Effective EPE}=\sum_{t_k\le\min(1\,\text{year},\ \text{maturity})}\text{Effective EE}_{t_k}\,\Delta t_k\quad(\text{equation 3})$$

with $\text{Effective EE}_{t_0}$ the current exposure and $\Delta t_k=t_k-t_{k-1}$, $t_0=0$. `effective_ee(ee, current_exposure=0.0)` is the running maximum
(non-decreasing and at least `ee`); golden `[1, 3, 2]` gives `[1, 3, 3]`. `effective_epe(grid, eff_ee, horizon=1.0)` is the sum over the points at or
before the horizon, normalised by the covered time so that a constant profile returns its value; golden grid `[0.25, 0.5, 1.0]` with effective EE
`[1, 3, 3]` gives $(1\times0.25+3\times0.25+3\times0.5)/1=2.5$. If every contract matures within the year, the average is over the time to maturity.
`epe(grid, ee, horizon=1.0)` is the plain time-weighted average of EE (dossier 05 C.1 $EPE_B$): golden EE `[1, 3, 2]` gives $0.25+0.75+1.0=2.0$, and it
is at most the effective EPE of the same EE. EEPE in dossier 05 is the framework's "Effective EPE". The regulatory EAD is $\alpha\times$ Effective EPE with
$\alpha=1.4$ (CRE53.11, CRE53.14); the engine states the measures and applies no alpha.

## PFE

$$\mathrm{PFE}_\alpha(t)=\Big(\inf\{x:F_t(x)\ge\alpha\}\Big)^+$$

`pfe(values, quantile, seed=0, n_bootstrap=200)` returns `(pfe, mc_se)`: the positive part of the inverse empirical quantile (`numpy.quantile` with
`method="inverted_cdf"`), golden the values $1,\dots,100$ at 0.95 giving 95. The standard error is the order-statistic one,
$\sqrt{q(1-q)/P}/\text{density}$, estimated by the standard deviation of the quantile over `n_bootstrap` resamples of the paths drawn with
`default_rng(seed)`, so the same inputs and seed give the same `mc_se`; the test compares it with the analytic value for a standard normal within 30
percent. PFE is non-negative and non-decreasing in the quantile. It is at least EE for a quantile high enough on a symmetric value (for a normal value
from about 0.66; at exactly 0.5 the PFE of a symmetric value is 0 while EE is positive, so the contract's "at least 0.5" is read as a quantile above
that).

## Netting

$$\max\Big(\sum_s v_s,0\Big)\le\sum_s\max(v_s,0)$$

`net_values(values_by_trade)` sums the (S, P, K) values over trades to (P, K): netting is on the netting-set sum. The netted EE is at most the sum of
the single-trade EE for any input, the difference being the netting benefit; `netting_benefit(ee_by_trade, ee_netted)` is $\sum_sEE_s-EE_{netted}$,
non-negative when the EEs come from the same paths. Golden: trades $+1$ and $-1$ on one path give a netted value 0 and a netted EE 0 while the sum of the
single-trade EE is 1; `[[1, 2], [1, 2]]` against `[1.5, 3.0]` gives `[0.5, 1.0]`.

## The effective maturity (CRE53.20)

$$M=\frac{\sum_{t_k\le1}\text{Effective EE}_k\,\Delta t_k\,df_k+\sum_{t_k>1}EE_k\,\Delta t_k\,df_k}{\sum_{t_k\le1}\text{Effective EE}_k\,\Delta t_k\,df_k}$$

`effective_maturity(grid, ee, eff_ee, df, cap=5.0)` with $df_k$ the risk-free discount factors, capped at 5 years. When every contract matures within one
year the second sum is empty and $M=1$ exactly; $M\ge1$ always. It raises when the effective EE is zero over the first year (the ratio is undefined).

## Ladder rung 2: the analytic EE of a single swap

`analytic_ee_single_swap(swap, grid, a, sigma, p0, forward)` returns the undiscounted EE (K,). Derived statement of the addendum R4 (own derivation from the
definition of exposure, not transcribed from a paper): with no collateral and $V(t)$ the value at $t$ of the remaining swap, $E^N[V(t)^+/N(t)]$ is by
risk-neutral pricing the time-0 price of a European option paying $V(t)^+$ at $t$: the payer swaption for a payer swap, the receiver swaption for a receiver
swap, expiring at $t$ on the remaining swap. Under the $t$-forward measure (numeraire $P(0,t)$) the undiscounted

$$EE_B(t)=\frac{\mathrm{Swaption}(0;t)}{P(0,t)}$$

from `hw1f.swaption_jamshidian` with the swap's notional, the flows at or before $t$ excluded (sticky date). The identity holds on the swap's reset dates only:
between resets the remaining swap has a fixed current coupon and is not a forward-starting swap, so a grid point that is neither a reset date nor at or after
the maturity raises `ValueError`. At and after the maturity the EE is 0. The comparison with the Monte Carlo `expected_exposure` of `swap_values_on_grid`
(test `ccr.ee_analytic_vs_simulated`) is made at the reset dates, within the Monte Carlo error: the discounted simulated EE equals $P(0,t)$ times this
profile (40,000 paths in the test). It is valid for the risk-neutral measure only, a check of the engine's EE and not a real-world exposure.

## Ladder rung 1

`rung1_ead(v, c, addon_aggregate, margined=False, th=0.0, mta=0.0, nica=0.0)` is SA-CCR's $1.4\,(RC+PFE)$ (decision D3): a thin wrapper of
`riskvalidation.regulatory.sa_ccr.netting_set_ead` that only ties the ladder numbering to a public name; it equals its `.ead` exactly. The aggregate add-on
is the sum of the five asset classes (CRE52.24), so the wrapper carries it whole in one class. The module is resolved when the function is called; it ships with
the regulatory unit (U9-E1), so on a tree without it the function raises `ModuleNotFoundError`. The CRE99 netting set 5 golden (V - C = -120, aggregate add-on
1,400.96238, EAD 1,879) is in the test and runs once that module is present.

## Limits

- The measures state no compliance claim and no alpha; the regulatory EAD is the caller's. CRE53.12 footnote 1: the expectations should in theory be
  taken under the actual probability distribution, and supervisors will not mandate which forecasting distribution to employ; the simulated paths are
  risk-neutral, so the profiles are those of the model's measure.
- The analytic EE is a check on the engine's EE, valid for the risk-neutral measure and on the reset dates of a single swap.
- The one-year horizon of the effective maturity is fixed by CRE53.20; the stressed EEPE of CRE53.7 (a calibration of three years that includes a stress
  period) is a choice of the caller's calibration window, not a function here.
