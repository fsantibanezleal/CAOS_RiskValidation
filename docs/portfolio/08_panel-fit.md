# 08. The default correlation from a panel by rating grade

Module `riskvalidation.portfolio.panel_fit`. The data are two $T\times G$ integer matrices: the defaults $D_{tg}$ of grade $g$ in year $t$ and the
cohort $N_{tg}$, the obligors of the grade at the start of the year. Page 06 fits one series of loss rates; this page fits the panel of a rating
system, with a common asset correlation and one threshold per grade, and returns the **default correlation** of each grade, which is what the
Schuermann and Hanson adjusted number of observations (`transitions.intervals.effective_n`, equation 3.4) takes. Sources, read on 2026-11-11:
Gordy and Heitfield (2002), "Estimating Default Correlations from Short Panels of Credit Rating Performance Data", Federal Reserve Board
(https://eml.berkeley.edu/~mcfadden/e242_f03/heitfield.pdf); Frey and McNeil (2003), "Dependent defaults in models of portfolio credit risk", Journal of
Risk 6(1), 59-92, doi:10.21314/jor.2003.089 (read as the 16 June 2003 preprint, https://www.wu.ac.at/fileadmin/wu/d/i/statmath/Publications/Publications_Frey/Journal_Papers/frey-mcneil-JoR.pdf);
Schuermann and Hanson, "Estimating probabilities of default", FRBNY Staff Report 190 (https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr190.pdf),
equation (3.4); Gordy (2000), "A comparative anatomy of credit risk models", Journal of Banking and Finance 24(1-2), 119-149,
doi:10.1016/S0378-4266(99)00054-0, read as the working paper FEDS 1998-47 (https://www.federalreserve.gov/pubs/feds/1998/199847/199847pap.pdf).

## The model and the likelihood (`fit_panel_asrf`)

An obligor of grade $g$ defaults when its asset value falls below the threshold $\gamma_g$, the asset value being $wX+\sqrt{1-w^2}\,\varepsilon$ with
$X\sim N(0,1)$ the systematic factor of the year, serially independent. The PD of the grade is $\Phi(\gamma_g)$ and the asset correlation $w^2$.
Given $X=x$ the grades are independent and the defaults binomial with

$$p_g(x)=\Phi\!\left(\frac{\gamma_g-w\,x}{\sqrt{1-w^2}}\right),\qquad
L(\gamma,w)=\prod_{t=1}^{T}\int_{\mathbb R}\prod_{g}\binom{N_{tg}}{D_{tg}}\,p_g(x)^{D_{tg}}\big(1-p_g(x)\big)^{N_{tg}-D_{tg}}\,\varphi(x)\,dx,$$

the one-dimensional integral of Gordy and Heitfield's equation (10), with one common loading $w$ (their restriction R3, the estimator MLE3). The
cohort sizes are treated as fixed, independent of the factor (their assumption). The default correlation of two obligors of grade $g$ is their
equation (5),

$$\rho_g=\frac{\Phi_2\big(\gamma_g,\gamma_g;\,w^2\big)-\mathrm{PD}_g^{\,2}}{\mathrm{PD}_g\,(1-\mathrm{PD}_g)},$$

with $\Phi_2$ the bivariate normal distribution function (computed as the integral of the squared conditional PD, as `portfolio.asrf_fit` does for one
series; `default_correlation(pd, asset_correlation)`). It increases with the PD at a common asset correlation: the same $w^2$ gives a larger default
correlation to a riskier grade.

**Optimisation.** L-BFGS-B on $(\gamma_1,\dots,\gamma_G,w^2)$ with the analytic gradient (checked against finite differences in the tests), $\gamma_g$
in $[-10,5]$ and $w^2$ in $[10^{-4},0.9999]$, from the thresholds $\Phi^{-1}$ of the pooled rates and three fixed starts of $w^2$ (0.02, 0.10, 0.30), the best
kept: deterministic. **Interval.** The likelihood-ratio interval of $w^2$: the values whose profile log-likelihood (the thresholds maximised at each
$w^2$) is at least $\max-\chi^2_1(\text{level})/2$ (for 0.95, 1.920729), found as in page 06; an endpoint at the parameter space is returned as the bound.
The interval is for the asset correlation; the per-grade default correlations carry no interval.

**The quadrature.** The papers name no rule. The engine's default is the trapezoid rule on $[-10,10]$ with 401 nodes (spacing 0.05): the integrand of a
year is a product of binomial probabilities, an analytic and sharply peaked function, for which the trapezoid rule converges geometrically. On the
reference panel below it agrees with 801 nodes to better than $10^{-10}$ in the log-likelihood and with an adaptive reference integration to
$10^{-13}$. Gauss-Hermite (`quadrature="gauss-hermite"`, up to 256 nodes) is available and is exact to $10^{-10}$ under doubling (128 against 256 nodes) on a
panel of small cohorts (8 years, cohorts 30 and 20), but on the reference panel (40 years, cohorts 400, 250, 100) it is off by about $10^{-5}$ in the
log-likelihood even at 256 nodes, because its nodes do not resolve a year's peak; for that reason it is not the default. With `n_nodes=None` the fit starts at 401
nodes and doubles the rule (halves the spacing) until the log-likelihood at the estimate moves by at most $10^{-6}$ under doubling, up to four doublings; an explicit
`n_nodes` is used as given and `quadrature_diff` and the flag `quadrature_not_converged` report the check, so a result never depends silently on the node count.

## The moment estimator (`fit_panel_moments`)

Frey and McNeil's equation (20) with $k=2$, per grade, over the years $j$ with cohort $m_j$ and defaults $M_j$:

$$\hat\pi=\frac1T\sum_j\frac{M_j}{m_j},\qquad \hat\pi_2=\frac1T\sum_j\frac{M_j(M_j-1)}{m_j(m_j-1)},\qquad
\hat\rho_Y=\frac{\hat\pi_2-\hat\pi^2}{\hat\pi-\hat\pi^2},$$

$\hat\pi_2$ being unbiased for the joint default probability of two obligors of the grade (their text: "unbiased for $\pi_k$ and consistent"); the asset
correlation is the root of $\Phi_2(\Phi^{-1}(\hat\pi),\Phi^{-1}(\hat\pi);\rho)=\hat\pi_2$ (the inversion of equation 5). It is the cross-check, and the fallback where a
likelihood is not identified. Frey and McNeil find the maximum-likelihood estimator better than the moment ones, increasingly so as defaults become rare;
Gordy's own moment estimator (their equation 21; 1998 working paper, Appendix B) is not implemented, its journal text not having been read.

## Where each estimate is not identified: flags and typed errors

Never a filled value.

| Case | Behaviour |
|---|---|
| a grade-year with NaN in either matrix | excluded, counted in `n_missing`, never filled |
| a cohort of fewer than 2 obligors (the term $D(D-1)/(N(N-1))$ is undefined at $N<2$) | excluded, counted in `n_small_cohort`, by both estimators |
| counts not integer, negative, infinite, shapes that differ, $D>N$ | `PanelFitError` (a `ValueError`) |
| a grade with no default in any usable year (threshold unidentified; Gordy and Heitfield drop their two top grades) | dropped from the fit, listed in `dropped`; moments: status `fewer_than_two_defaults` |
| a grade with a single default | the fit keeps it with the flag `grade_few_defaults`; moments: `fewer_than_two_defaults`, NaN (Gordy and Heitfield: the unrestricted parameters are not identified below two defaults, which is why MM and MLE1 cannot be used) |
| no grade with a default; fewer than 3 usable years | `PanelFitError` |
| no excess dispersion (the likelihood drives $w^2$ to its lower bound; moments: $\hat\pi_2\le\hat\pi^2$) | the fit's $w^2$ is the bound $10^{-4}$ with the flag `asset_correlation_at_lower_bound`, `converged=False`; moments: both correlations 0.0 with the status `no_excess_dispersion` |
| $w^2$ at the upper bound | flag `asset_correlation_at_upper_bound`, `converged=False` |
| moment asset correlation above $1-10^{-9}$ | status `second_moment_above_bound`, NaN |
| fewer than 20 usable years (the studied range is 20 to 160) | flag `short_panel`; the downward bias Gordy and Heitfield report in all estimators is not corrected |
| optimiser not converged, quadrature check failed | flags `optimiser_not_converged`, `quadrature_not_converged`, `converged=False` |

Assumptions the estimates rest on (Gordy and Heitfield; Gordy 2000): one systematic factor, serially independent years, cohort sizes independent of the factor,
thresholds and the loading constant in time and one loading for all grades. If they fail the estimate misleads (the authors' words); the per-grade moment
values are the diagnostic.

## Goldens

- `default_correlation(0.05, 0.2)` = 0.057799 and `default_correlation(0.01, 0.2)` = 0.024133.
- One grade, four years, $D=(0,5,0,5)$, $N=10$: $\hat\pi=0.25$, $\hat\pi_2=1/9$, $\hat\rho_Y=7/27=0.259259$, and `effective_n(100, 7/27)` $=100/(1+99\cdot7/27)=3.75$.
- The reference panel `simulate(T=40, PDs 0.0015, 0.01, 0.05, cohorts 400, 250, 100, w^2=0.2, seed 1)` of `tests/portfolio/test_panel_fit.py`: the log-likelihood at
  $\gamma=(-2.9,-2.3,-1.6)$, $w^2=0.2$ is $-188.809685$; the fit gives $w^2=0.203868$ (interval 0.131120 to 0.315634), log-likelihood $-188.014542$, PDs 0.001441, 0.009478, 0.045882,
  default correlations 0.007699, 0.024079, 0.056719; the moment estimator gives default correlations 0.012414, 0.036157, 0.073581 and asset correlations 0.245774, 0.257601, 0.248043
  (grades with few defaults are noisy: the likelihood pools the three grades through the common $w$).
- Recovery over seeds (40 seeds per design, measured before the tests): the standard deviation of the $w^2$ estimate is 0.030 for $T=100$ with the three grades above, 0.0123 for $T=100$,
  four grades, $w^2=0.1$, and 0.044 for $T=50$, two grades, $w^2=0.3$; the mean error over 40 seeds is within 0.01 of the truth in every design (0.196, 0.097, 0.295 for truths 0.2, 0.1, 0.3).

## Limits

The one-factor common-loading model is the setting of the IRB formula; a rating system whose grades load differently on the cycle is fitted as if they did not. The maximum-likelihood
estimate of a short panel is biased downward (Gordy and Heitfield, Section 4; MLE3 least) and the profile interval of 20 years is wide; the engine invents no bias correction. The
Schuermann and Hanson correction itself "assumes that the correlation across draws is known" (their text): an estimated default correlation enters `effective_n` as if it were known, and its
uncertainty is not propagated. Cohorts of tens of thousands of obligors need a fine rule (the adaptive default handles it at a cost); Gordy (2000)'s journal text and de Servigny and Renault (2002) were not
read (**UNVERIFIED**; the estimator attributed to the latter is the $k=2$ case of equation 20). The engine fits a correlation and makes no compliance claim.
