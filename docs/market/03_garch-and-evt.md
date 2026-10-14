# 03. GARCH, GJR, EGARCH and conditional EVT

Modules `riskvalidation.market.garch` and `riskvalidation.market.evt`. Sources: dossier 05 A.2 rows 7 to 10. `arch` (the
optional extra `market`, `pip install "riskvalidation[market]"`) is imported inside the functions that need it, with a clear
error when it is absent; the engine core stays NumPy and SciPy.

## GARCH family, rung 5

$$\sigma_t^2=\omega+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2\qquad\text{(GARCH, Bollerslev 1986, doi:10.1016/0304-4076(86)90063-1)}$$

$$\sigma_t^2=\omega+(\alpha_1+\gamma\,\mathbb 1\{\varepsilon_{t-1}<0\})\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2\qquad\text{(GJR, doi:10.1111/j.1540-6261.1993.tb05128.x)}$$

$$\ln\sigma_t^2=\omega+\beta\ln\sigma_{t-1}^2+\theta z_{t-1}+\gamma(|z_{t-1}|-E|z_{t-1}|)\qquad\text{(EGARCH, Nelson 1991, doi:10.2307/2938260)}$$

`fit_garch(returns, kind, innovation, mean)` fits `kind` in `garch`, `gjr`, `egarch` with `innovation` in `student-t`, `skew-t`,
`normal` and `mean` in `zero`, `constant` by quasi-maximum likelihood through `arch.arch_model`, on returns of at least 250 days
(`ValueError` otherwise), deterministic (the optimiser and its start are those of `arch`). The data reach `arch` in percent;
the parameters come back in decimals: `mu` in decimals, `omega` in decimals squared (GARCH, GJR) and for EGARCH the log-variance
constant shifted by `ln(1e4)(1 - beta)`; the log-likelihood is that of the decimal returns (`ll_pct + N ln 100`). The names are
those of `arch`: `mu`, `omega`, `alpha[1]`, `gamma[1]` (the leverage term of GJR; for EGARCH `arch` calls the sign term
`gamma[1]`, the `theta` above, and the magnitude term `alpha[1]`, the `gamma` above), `beta[1]`, `nu`, `lambda` (skew-t). A
failed optimisation, `omega <= 0`, a negative coefficient, `alpha + beta >= 1` (GARCH), `alpha + gamma/2 + beta >= 1` (GJR),
`|beta| >= 1` (EGARCH) or `nu <= 2` gives `converged = False`; the parameters are never invented.

`garch_var_es_forecast(fit, returns, alpha)` runs the fixed filter (`arch_model(...).fix(params)`) through the series the fit
used and forecasts the next day:

$$\mathrm{VaR}=\mu+\sigma_{t+1}q_\alpha,\qquad \mathrm{ES}=\mu+\sigma_{t+1}\,\frac1\alpha\int_0^\alpha q_u\,du$$

with $q$ the quantile function of the unit-variance innovation. Normal: the result is `delta_normal_var_es(mu, sigma, alpha)`
(golden `mu = 0`, `sigma = 0.01`, `alpha = 0.01` gives `(-0.023263, -0.026652)`). Student-t with `nu > 2`:
$q_\alpha=t_\nu^{-1}(\alpha)\sqrt{(\nu-2)/\nu}$ and the closed form
$\mathrm{ES}_q=-\frac{f_\nu(t_\alpha)}{\alpha}\,\frac{\nu+t_\alpha^2}{\nu-1}\sqrt{\frac{\nu-2}{\nu}}$, checked against the quadrature of the
definition. Skew-t (Hansen): `arch`'s quantile function with the quadrature of `conventions.var_es_from_distribution`.
`garch_filter(fit, returns)` returns the per-day mean, volatility and standardised residuals `z_t = (r_t - mu_t)/sigma_t` and the
next-day `mu` and `sigma` for the filtered HS and the EVT.

Limits: quasi-maximum likelihood with a heavy tail needs a long window (the hyperparameters, window and distribution, are fixed
once on 1995-01-01 to 2003-12-31, decision D1 of the U8 sizing, never tuned on a test window); GARCH misses the leverage of
equities (use GJR); the EGARCH here has the magnitude and sign terms of `arch`'s parameterisation.

## Conditional EVT, rung 7

$$\hat z_q=u+\frac{\hat\beta}{\hat\xi}\left[\left(\frac{n}{N_u}(1-q)\right)^{-\hat\xi}-1\right],\qquad
\widehat{\mathrm{ES}}_q=\frac{\hat z_q}{1-\hat\xi}+\frac{\hat\beta-\hat\xi u}{1-\hat\xi}\quad(\hat\xi<1)$$

`fit_gpd(z_loss, threshold_fraction)`: `u` is the empirical `(1 - fraction)` quantile of the standardised loss residuals, `N_u`
counts the residuals strictly above it, and the excesses are fitted by maximum likelihood with `scipy.stats.genpareto` (location
fixed at 0); `ValueError` if `N_u < 10`. `evt_var_es(fit, q)` returns `(z_q, es_q)` in standardised loss units; `es_q` is `None`
when `xi >= 1` (the caller records the day in `es_undefined_days`, never a number); `xi = 0` uses the exponential limit
`z_q = u - beta ln(n(1-q)/N_u)`, `es_q = z_q + beta`. Golden: `u = 1`, `beta = 1`, `xi = 0.5`, `n = 1000`, `N_u = 100`, `q = 0.99`
gives `z_q = 5.324555` and `es_q = 11.649111`.

`conditional_evt_forecast(returns, alpha, threshold_fraction, kind)` fits the GARCH filter (`kind`, normal innovations, constant
mean: the pseudo-maximum likelihood of McNeil and Frey; the AR(1) mean term of their paper is not used, a decision of this unit
recorded on the report), fits the GPD to $-z_t$, and forms, in the return convention, `var = mu_{t+1} - sigma_{t+1} z_q` and
`es = mu_{t+1} - sigma_{t+1} es_q` (`None` exactly when `xi >= 1`). `threshold_stability(z_loss, fractions)` returns one
`(fraction, xi, n_exceed)` per fraction ascending, the table the C11 models document keeps per window. Source: McNeil and Frey,
J. Empirical Finance 7(3-4):271-300, 2000, doi:10.1016/S0927-5398(00)00012-8. Limits: the threshold choice is the main model
risk (the original paper uses about the top 10 percent of residuals, UNVERIFIED here); `xi` is unstable in short windows; the
ES formula holds for `xi < 1`.
