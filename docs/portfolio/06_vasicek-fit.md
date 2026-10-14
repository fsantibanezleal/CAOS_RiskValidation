# 06. The Vasicek fit

Module `riskvalidation.portfolio.asrf_fit`. Source: dossier 04 C.4 (Vasicek, Gordy ASRF and the conditional PD). Vasicek's notes are
reprinted in Vasicek (2015), Finance, Economics and Mathematics, Wiley, doi:10.1002/9781119186229.ch17 to ch19 (the 2002 note,
"Loan portfolio value", Risk, December 2002, is ch19); Gordy (2003), doi:10.1016/S1042-9573(03)00040-8.

## The model

In the large homogeneous portfolio the yearly loss rate is $p(Z)$ with $Z\sim N(0,1)$ and $p(z)=\Phi\big((G(\mathrm{PD})-\sqrt\rho z)/\sqrt{1-\rho}\big)$, with the
density `engines.vasicek.lhp_pdf`, the distribution function `lhp_cdf` and the quantile `lhp_quantile`. A series of yearly rates $r_1,\dots,r_T$ is fitted
for $(\mathrm{PD},\rho)$. A PD, a correlation and a rate are open at every estimate site: a value outside (0, 1) raises `ValueError`. A NaN rate is a
year the table lacks: it is excluded, counted in `n_missing` and never interpolated. At least 3 usable years are needed.

## Estimators

**Maximum likelihood (`fit_ml`).** $(\widehat{\mathrm{PD}},\widehat\rho)=\arg\max\sum_t\ln f(r_t;\mathrm{PD},\rho)$ with $f$ = `lhp_pdf` (`log_likelihood`), by Nelder-Mead on the
logit scale from a fixed start with a fixed tolerance and one restart (deterministic). `converged` is false when the optimiser stops at its
cap or at a bound ($\rho$ or PD outside [0.0001, 0.9999]). A series with zero spread raises: the correlation is not identified.

**Moments (`fit_moments`).** $\widehat{\mathrm{PD}}=\bar r$ and $\widehat\rho$ solves

$$\mathbb E[L^2]=\Pr(\text{two obligors default})=\Phi_2\big(G(\mathrm{PD}),G(\mathrm{PD});\rho\big)=\mathbb E[p(Z)^2]=\overline{r^2},$$

the second moment of the large-portfolio default fraction being the joint default probability of two names. The joint probability is increasing in
$\rho$ (from $\mathrm{PD}^2$ to $\mathrm{PD}$), so the root is unique; when $\overline{r^2}\le\mathrm{PD}^2$ there is no excess dispersion and the function raises, never returning 0.
The log-likelihood is evaluated at the moment estimates for comparison.

**Profile and interval.** `profile_likelihood(rates, rho_grid)` returns, at each correlation of a strictly increasing grid, the PD that maximises the likelihood
(a coarse grid on the logit scale and a bounded refinement) and the maximised log-likelihood. `profile_interval(rates, level)` is the likelihood-ratio interval:
the parameter values whose profile log-likelihood is at least $\max-\chi^2_1(\text{level})/2$ (for 0.95, $3.841459/2=1.920729$), for the PD and for $\rho$;
an endpoint that reaches the parameter space is returned as the bound (0.0001 or 0.9999), not as an error.

**Bootstrap (`bootstrap_interval`).** A parametric bootstrap: fit the estimator, draw `n_boot` (at least 100) series of the same length with
$z_t\sim N(0,1)$ and $r_t=p(z_t)$ (`conditional_pd`), refit each, and return the percentiles $(1\mp\text{level})/2$ of the refits. A refit that raises is dropped and
counted (`n_dropped` on the result); a maximum-likelihood refit that stops at a bound is kept. Reproducible for a seed, using
`numpy.random.Generator(numpy.random.PCG64(seed))`. The interval is not required to contain the estimate for a skewed refit distribution.

**Implied factors (`implied_factors`).** The Z-factor of the credit cycle (a derivation from the inverse of the large-portfolio relation, not a quoted formula):

$$\hat z_t=\frac{G(\mathrm{PD})-\sqrt{1-\rho}\,G(r_t)}{\sqrt\rho},$$

decreasing in the rate (a high loss rate is a low factor), equal to `engines.vasicek.implied_factor`, NaN where the rate is NaN.

## Goldens

- `log_likelihood` of the rates (0.01, 0.02, 0.05) at PD 0.02, $\rho$ 0.15: densities 34.661219, 18.004184 and 3.517955, log-likelihood 7.694105.
- `fit_ml` on the quasi-sample $r_t=\mathtt{lhp\_quantile}((t-0.5)/T,\,0.02,\,0.15)$, $T=2000$: PD 0.019995 and $\rho$ 0.149917, recovering (0.02, 0.15) to $10^{-3}$.
- `fit_moments` on (0.01, 0.02, 0.03): PD 0.02, $\overline{r^2}=0.000467$, $\rho=0.026880$.
- `implied_factors((0.05,), 0.02, 0.15)` = -1.387222, and `conditional_pd(0.02, 0.15, -1.387222)` = 0.05.

## Limits

A short series (22 years in the product) identifies the correlation poorly: the profile interval is wide and may reach the bound, and
the maximum-likelihood estimate can sit at a bound (then `converged` is false). Moments are less efficient than maximum likelihood and need
excess dispersion. The model is the single-factor large homogeneous portfolio: a series of loss rates of a portfolio with granularity or several
factors is fitted as if it were one. The size and power of these estimates on a known truth are measured by the simulator and judged by
`correlation_size_power` (page 07).
