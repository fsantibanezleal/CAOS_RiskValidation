# 07. Simulation

Module `riskvalidation.irrbb.simulation`. Sources: dossier 05 sections D.5 and D.6 (the dynamic NII simulation, EaR and EVaR),
C.4 (Hull-White one-factor) and dossier 01 H.1 (DIS70.5, the constant balance sheet). Every function with randomness takes a `seed`
and uses `numpy.random.default_rng(seed)`.

## The dynamic NII projection

`simulate_nii(balances, sides, rate0, spread, beta, reprice_months, short_rate_path, growth_annual)` projects the monthly NII of a
book over 12 to 36 months (the path length):

$$NII_m=\frac{1}{12}\sum_i s_i\,B_i\,(1+g)^{m/12}\,r_{i,m}$$

Symbols: $s_i=+1$ for an asset and $-1$ for a liability, $B_i$ the balance, $g$ the annual growth of the balance sheet (0 is the
constant balance sheet: run-off replaced by identical new business), $r_{i,m}$ the rate of item $i$ in month $m=0..M-1$ (the 0-based index). A reset rate
is $\mathrm{spread}_i+\beta_i\,r_j$ (an asset indexed to the market has $\beta=1$; an NMD deposit has the deposit beta). The reset
rule (the month $m$ uses `short_rate_path[m]`): `reprice_months = 0` resets every month, so the rate of month $m$ is
$\mathrm{spread}+\beta\,r_m$; $k\ge1$ keeps `rate0` for the months $m<k$ and, from $m=k$ on, uses
$\mathrm{spread}+\beta\,r_{\lfloor m/k\rfloor k}$, so the item resets at $m=k,2k,\dots$ and the rate set at a reset applies until the
next one. The whole
balance of an item resets together; a staggered ladder is a set of items. Total NII is the sum of the months.

Golden: one asset (balance 100, side +1, rate0 0.05, spread 0.02, beta 1, reprice 0) and one liability (100, -1, 0.015, 0, 0.5, 0),
a flat path of `0.03` over 12 months: total `100 * 0.05 - 100 * 0.015 = 3.5`; the same book with the path at `0.04` in every month:
`100 * 0.06 - 100 * 0.02 = 4.0`.

## Hull-White one-factor scenarios

$$dr=(\theta(t)-a\,r)\,dt+\sigma\,dW,\qquad \theta(t)=\frac{\partial f(0,t)}{\partial t}+a\,f(0,t)+\frac{\sigma^2}{2a}\left(1-e^{-2at}\right)$$

with $\theta(t)$ chosen to fit today's discount curve exactly (dossier 05 C.4; Hull and White 1990). `hw1f_scenarios` simulates the
equivalent decomposition $r(t)=x(t)+\alpha(t)$, with $dx=-a\,x\,dt+\sigma\,dW$, $x(0)=0$ (simulated exactly on the monthly grid,
$x_{k+1}=x_ke^{-a\Delta t}+\sigma\sqrt{(1-e^{-2a\Delta t})/(2a)}\,z_k$) and

$$\alpha(t)=f(0,t)+\frac{\sigma^2}{2a^2}\left(1-e^{-at}\right)^2$$

(Brigo and Mercurio; UNVERIFIED edition and page, as in the dossier). The identity $\alpha'+a\alpha=\theta$ holds, and the test checks
it numerically. $f(0,t)=R(t)+t\,R'(t)$ is the instantaneous forward of the initial zero curve (linear in tenor, flat extrapolation).
Symbols: $a$ the mean reversion per year, $\sigma$ the annual volatility, $z_k$ standard normal. The paths are antithetic (the second
half mirrors the first); with $\sigma=0$ every path is the initial forward curve; the average discount factor of the paths reprices the
initial curve within Monte Carlo error (the test checks it with a trapezoid on the monthly grid). Golden: a flat curve `0.03` with
`sigma = 0` gives every path equal to `0.03` at every step.

`dns_scenarios(factor_history, maturities_months, horizon_months, n_paths, seed)` maps the AR(1) factor paths of
`forecast_dns_factors` through the Diebold-Li loadings (page 01) to (paths, months, maturities) yields; it uses only the history it is
given (the C14 bake passes the factors up to the valuation date). Golden: a constant history gives every path equal to the
`dns_yield` of that row.

## EaR and EVaR

$$EaR=NII_0-Q_{1-\ell}(NII),\qquad EVaR=EVE_0-Q_{1-\ell}(EVE)$$

Loss-positive, with $\ell$ the confidence level (default 0.99), $Q$ the linearly interpolated quantile (`numpy.quantile`) of the
simulated NII and EVE paths, and $NII_0$, $EVE_0$ the base values. Monotone in the level; all-equal paths give base minus path; at
least 2 paths. Golden: `nii_paths = 1..100`, base 60, level 0.99: the first percentile is `1.99`, so `ear = 58.01`.

`ear_evar_mc_error(nii_paths, eve_paths, level, n_batches=20)` returns an `EarEvarMcError` (`level`, `nii_quantile_se`,
`eve_quantile_se`, `ear_se`, `evar_se`): the Monte Carlo standard error, in money, of the $1-\ell$ quantile of the NII paths and of
the EVE paths, which is the Monte Carlo error of the EaR and of the EVaR at the same level (the base values are constants, so
`ear_se = nii_quantile_se`). The case document carries an engine number, not a hand-rolled one. Batch means: the first
$B\lfloor P/B\rfloor$ paths, in the order they were simulated, are cut into $B$ contiguous batches, the $1-\ell$ quantile (linear
interpolation) is taken in each batch, and

$$SE=\frac{\operatorname{sd}(Q_1,\dots,Q_B)}{\sqrt B}$$

with $Q_b$ the batch quantiles. It needs $B\ge2$ and $P\ge2B$, is deterministic and linear in the scale of the paths. The estimate
itself has a relative noise of about $1/\sqrt{2(B-1)}$.

Golden (exact): $B=2$, $\ell=0.5$, paths $1..20,101..120$: the batch medians are $10.5$ and $110.5$, so $SE=(100/\sqrt2)/\sqrt2=50$.
Closed form for the check of the estimator: for independent $N(0,\sigma^2)$ paths the asymptotic standard error of the empirical
$p$-quantile is $\sigma\sqrt{p(1-p)/P}\,/\,\varphi(z_p)$ ($z_p$ the normal quantile), $0.037332$ at $\ell=0.99$ and $0.021132$ at
$\ell=0.95$ for $\sigma=1$, $P=10^4$; the tests compare the estimator with it on seeded samples of $10^5$ paths and 20 batches, and for
Exp(1) paths (density $1-q$ at the quantile).

## Sources

- Hull and White (1990), Rev. Financial Studies 3(4):573-592, doi:10.1093/rfs/3.4.573.
- Brigo and Mercurio, *Interest Rate Models* (Springer), the $\alpha(t)$ form of the Hull-White model (edition and page UNVERIFIED).
- Diebold and Li (2006), J. Econometrics 130(2):337-364, doi:10.1016/j.jeconom.2005.03.005.
- DIS70.5 (dossier 01 H.1): the constant balance sheet,
  https://www.bis.org/basel_framework/chapter/DIS/70.htm.

## Limits

- The reset rule treats the whole balance of an item as repricing together; the new business is priced at the current reset rate.
- The Hull-White parameters $a$ and $\sigma$ are inputs: the engine does not calibrate them to swaption volatilities.
- The Monte Carlo error is a batch-means estimate over contiguous batches; a bake reports the number of paths, the seed and the
  number of batches with it. Contiguous batches of an antithetic layout (the mirrored half after the first half) put the mirror
  images in different batches, which is harmless for a symmetric distribution and conservative for a skewed one.
- Antithetic pairs halve the Monte Carlo variance for a monotone payoff only; the bake reports the Monte Carlo error with the
  number of paths and the seed.
- The Kalkbrener and Willing stochastic NMD SDEs are not implemented (dossier 05 D.3, UNVERIFIED detail).
