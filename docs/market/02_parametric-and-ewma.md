# 02. Parametric VaR and ES, Cornish-Fisher and EWMA

Module `riskvalidation.market.parametric`. Sources: dossier 05 A.2 rows 4 to 6. Return convention throughout.

## Delta-normal, rung 3

$$\mathrm{VaR}_\alpha=\mu+\Phi^{-1}(\alpha)\,\sigma,\qquad \mathrm{ES}_\alpha=\mu-\sigma\,\frac{\varphi(\Phi^{-1}(\alpha))}{\alpha}$$

`delta_normal_var_es(mu, sigma, alpha)`: `mu` and `sigma > 0` the portfolio mean and volatility over the horizon (decimals). The
result is affine in `mu` and linear in `sigma`, with `es <= var`. Golden: `(0, 1, 0.01)` gives `(-2.326348, -2.665214)` and
`(0, 1, 0.025)` gives `(-1.959964, -2.337803)`. Source: RiskMetrics Technical Document, 4th ed., J.P. Morgan/Reuters, 1996-12-17
(https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a). Limit: it fails on optionality and fat tails; it is
the transparent baseline.

## Cornish-Fisher VaR (decision D-6)

$$z_{CF}=z+\frac{(z^2-1)S}{6}+\frac{(z^3-3z)K_e}{24}-\frac{(2z^3-5z)S^2}{36},\qquad z=\Phi^{-1}(\alpha),\qquad \mathrm{VaR}=\mu+z_{CF}\,\sigma$$

`cornish_fisher_var(mu, sigma, skew, excess_kurtosis, alpha)` returns a frozen `CornishFisher(var, z_cf, valid)`. With `S = Ke =
0` it equals the delta-normal VaR. The expansion is non-monotone for large skewness and kurtosis and the dossier requires the
validity domain to be reported; the dossier does not define it, so **decision D-6** (the contract's): `valid` is true when the
derivative of the map $z\mapsto z_{CF}(z)$,

$$1+\frac{2zS}{6}+\frac{(3z^2-3)K_e}{24}-\frac{(6z^2-5)S^2}{36},$$

is positive at every point of $[\Phi^{-1}(\alpha),0]$ (a grid of 201 points). A document reports `valid = false` rather than hiding
the forecast. Golden: `alpha = 0.01`, `S = -0.5`, `Ke = 1.0` gives `z_cf = -2.833709` and `valid = True`; `S = -3`, `Ke = 20` gives
`z_cf = -5.821010` and `valid = False`. Source: Cornish and Fisher (1938), doi:10.2307/1400905; applied to VaR by Zangari
(RiskMetrics Monitor, 1996), whose origin the dossier marks UNVERIFIED. Limit: the monotonicity check is on the tail
$[\Phi^{-1}(\alpha),0]$ only; it does not certify the tail beyond the quantile.

## EWMA, rung 4

$$\sigma^2_{t+1\mid t}=\lambda\,\sigma^2_{t\mid t-1}+(1-\lambda)\,r_t^2,\qquad K=\frac{\ln\epsilon}{\ln\lambda}$$

`ewma_variance(returns, lam, initial_variance)` returns the `T+1` one-step variances: element 0 the initial variance, element
`t+1` the forecast for day `t+1` after observing `r_t`; all positive when `initial_variance > 0` (`ValueError` otherwise). The
zero-mean assumption is part of the method. Golden: `returns = [0.02]`, `lam = 0.94`, `initial_variance = 0.0001` gives
`[0.0001, 0.000118]`. `ewma_effective_days(lam, tolerance)` is `ln(tolerance) / ln(lam)`: golden `(0.94, 0.01)` gives 74.4 and
`(0.97, 0.01)` gives 151.2 (74.427 and 151.191; the dossier states 74 and 151 days at 1 percent, the integer parts), increasing in
`lam`. `ewma_var_es(returns, lam, alpha, initial_variance)` is `delta_normal_var_es(0, sqrt(variance), alpha)` with the last EWMA
variance: golden `returns = [0.02]`, `lam = 0.94`, `alpha = 0.01`, `initial_variance = 0.0001` gives `(-0.025271, -0.028952)`
(`sigma = 0.010863`). Source: RiskMetrics TD 1996 section 5.3.2 and Table 5.9 ("the decay factor for the daily data set is 0.94,
and the decay factor for the monthly data set is 0.97"). Limit: the normal quantile under-covers at 99 percent; the initial
variance is the caller's choice and is part of the estimation window, never of the forecast day.
