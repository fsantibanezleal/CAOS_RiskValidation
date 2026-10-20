# 02. Swaps

Module `riskvalidation.ccr.swaps`. Sources: dossier 05 (the counterparty ladder, rung 3), the U9 research addendum R3 (the sticky-date close-out)
and R4 (the Hull-White bond prices of page 01). A `Swap` (frozen) holds `trade_id`, `notional` $N$, `fixed_rate`, `is_payer` (the bank pays the
fixed rate), `start`, `maturity` and `fixed_frequency`. It refuses a maturity not above the start, a negative start and a non-positive notional.
The floating leg resets on the single curve of the case (no spread).

## Dates

`cash_flow_times(swap)` returns the fixed payment times $T_i=\text{start}+i/\text{frequency}$ up to the maturity, the maturity last (a last period
shorter than $1/\text{frequency}$ ends at the maturity); golden: start 0, maturity 2, frequency 2 gives `[0.5, 1.0, 1.5, 2.0]`. `reset_times(swap)`
is the start and every payment time but the last. Cash flows are by date, so a close-out window sees the flows it contains.

## The value at a date

`swap_value(swap, t, r, a, sigma, p0, forward, last_fixing=None)` returns (P,) money to the bank, positive an asset, from the short rates `r` at
$t$; only the flows after $t$ are valued (the sticky date of the close-out approach). The fixed leg is $\sum_i c_i\,P(t,T_i)$ over the payments
after $t$, with $c_i=N\cdot\text{fixed\_rate}\cdot\tau_i$. The floating leg: at a reset date $T_j$ the coupon $N\tau L_j$ with

$$L_j=\frac{1/P(T_j,T_{j+1})-1}{\tau}$$

is fixed and paid at $T_{j+1}$.

- With $t$ on a reset date, or before the first one, the floating leg is $N\big(P(t,T_0')-P(t,T_n)\big)$ with $T_0'$ the next reset at or after $t$
  ($P(t,t)=1$), from the affine bond prices; `last_fixing` is not used.
- With $t$ strictly between two resets $T_j<t<T_{j+1}$ the current coupon is already fixed, and the floating leg is
  $N\tau L_j\,P(t,T_{j+1})+N\big(P(t,T_{j+1})-P(t,T_n)\big)$. It needs the path's fixing, `last_fixing` $=L_j$ (P,); its absence raises `ValueError`
  naming `last_fixing`: the valuation is never silently approximated.

A payer value is minus the receiver value of the same terms; no flow after $t$ gives 0. Goldens: on a flat curve $P(0,T)=e^{-0.03T}$ a receiver at
its par rate has value 0 at $t=0$ for any $a$ and $\sigma$ (to 1e-12, on two curves); with $\sigma=0$ and a one-period swap $[0,0.5]$ at $t=0.25$ the
floating coupon is $N(e^{0.015}-1)$ paid at 0.5 and the floating leg is $N(e^{0.0075}-e^{-0.0075})=2\sinh(0.0075)\,N=0.01500014\,N$ (tolerance 5e-9). The
test also values a two-period swap through the flows discounted by hand.

`par_fixed_rate(maturity, frequency, p0, start=0.0)` is

$$\frac{P(0,\text{start})-P(0,\text{maturity})}{\sum_i \tau_i\,P(0,T_i)}$$

and the value of a swap at that rate is 0. Golden: a flat continuous 0.03, start 0, maturity 1, frequency 1 gives $(1-e^{-0.03})/e^{-0.03}=0.030455$.

## Values along the paths

`swap_values_on_grid(swaps, grid, short_rate, a, sigma, p0, forward)` returns (S, P, K) money per swap, path and grid point; it equals `swap_value` at
each grid point, and the sum over swaps is the netting-set value (netting is on the sum, not on the sum of positive parts). The fixing at the last
reset is read from the same path, so `grid` must hold every reset date the valuation reads: `simulation_grid(grid, swaps)` returns the union of a
report grid and every reset date not after its last point, the grid to hand to `hw1f.simulate_short_rate`; a reset date missing from `grid` raises
`ValueError`. The result is on the grid it is given.

A check of the between-reset values that does not depend on any formula of this page: the discounted value of the swap is not a martingale by itself,
because a flow paid at a date leaves the portfolio, but the discounted mean equals minus the present value, today, of the flows paid at or before the
date. The test verifies this at every grid point of a par swap within four Monte Carlo standard errors.

## Limits

- Interest rate swaps on one curve with a fixed and a floating leg of the same frequency; no spread, no stub on the floating leg, no
  cross-currency, no amortisation. The Andersen, Pykhtin and Sokol timelines are not modelled.
- The close-out value is the sticky-date one (the market evolves from the default date to the close-out date, the portfolio does not age): a swap with
  no flow in the window needs no spike. ORE's actual-date alternative, with its exposure spikes, is not implemented.
