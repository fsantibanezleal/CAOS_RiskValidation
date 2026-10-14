# 06. Behavioural models

Module `riskvalidation.irrbb.behavioural`. Source: dossier 05 sections D.3 (non-maturity deposits) and D.4 (prepayment).

## The static replicating portfolio

Choose weights $w_j\ge0$, $\sum_jw_j=1$, on rolling tranches of maturities $m_j$ so that the margin $m_t=\sum_jw_j\,\bar y_{j,t}-i_t$
(replicating yield minus client rate) has minimal variance:

$$\min_{w}\ \operatorname{Var}\!\Big(\sum_jw_j\,\bar y_{j,t}-i_t\Big)=w^\top C\,w-2\,w^\top c+\operatorname{Var}(i_t)\quad
\text{s.t.}\ w\ge0,\ \mathbf 1^\top w=1$$

Symbols: $C$ the sample covariance of the tranche yields, $c$ the covariance of each tranche yield with the client rate, $\bar y_{j,t}$
the yield of tranche $j$ at date $t$. `replicating_portfolio` returns the global optimum: up to 10 tranches by exhaustive search over
the supports (each an equality-constrained quadratic problem solved exactly through its KKT system, kept when nonnegative), above
that a sequential quadratic programme polished on its support. `margin_volatility` is the sample standard deviation of the margin, or
`None` (the document carries the reason) with fewer than $J+2$ observations. Golden: a client rate equal to the second column of the
yields gives `weights = [0, 1, 0]` and `margin_volatility = 0` to `1e-9`.

The implied duration follows from the weights. `weighted_average_maturity(maturities, weights)` is $\sum_jw_jm_j/2$: a rolling tranche
of maturity $m$ reinvested evenly over its life has a weighted average maturity of $m/2$ (BCBS 368 footnote 38 as read in dossier 05
D.3: one sixtieth reinvested monthly in a five-year fixed gives 2.5 years). The weights must sum to one to `1e-9`. Golden: one tranche of
5 years with weight 1 gives `2.5`; the weights `[0.2, 0.3, 0.3, 0.2]` on `[1, 3, 5, 10]` give `2.3` by the formula.

## The deposit beta

$$i_{new}=i+\beta\,\Delta r,\qquad \beta=\beta_{up}\ (\Delta r\ge0),\quad \beta=\beta_{down}\ (\Delta r<0)$$

Asymmetric pass-through (Drechsler, Savov and Schnabl 2021): low betas make deposits behave like long-duration liabilities. Betas
are nonnegative and the result is not floored (the caller floors). Golden: `deposit_beta_passthrough(0.01, 0.02, 0.4, 0.3) = 0.018`;
`(0.01, -0.02, 0.4, 0.3) = 0.004`.

## The prepayment logit and its fit

$$SMM=\frac{1}{1+\exp\big(-(b_0+b_1\,\text{incentive}+b_2\,\text{seasoning})\big)}$$

A reduced-form logit of the SMM on the refinancing incentive (the contract rate minus the market rate, decimals) and seasoning
(months); burnout and seasonality are not in this version (dossier 05 D.4: practitioner models are reduced-form logit or Cox models of
SMM on refinancing incentive, seasoning, burnout, seasonality and turnover, FRBNY SR 1001 section 4.2.1). Golden: `coef = [0, 0, 0]`
gives `0.5` for every input.

`fit_prepayment_logit(incentive, seasoning_months, prepaid)` maximises the Bernoulli likelihood by Newton's method with step halving
from the zero start, on the standardised regressors, with a fixed tolerance (`1e-10` on the step) and at most 100 iterations. It
returns `converged = False` with NaN coefficients, and never raises, for an empty class, a constant regressor, complete separation (a
standardised coefficient beyond 50) or a likelihood that does not settle; the bake then uses the PSA benchmark and says so. The
coefficients satisfy the score equations $X^\top(y-p)=0$, and `std_err` are the standard errors from the inverse observed
information $(X^\top W X)^{-1}$, $W=\operatorname{diag}(p(1-p))$, in the original units (NaN when not converged).

Golden (the simulation design is part of the golden): $N=20000$, incentive $\sim U(-0.10,0.10)$, seasoning $\sim U(0,360)$ months,
independent, prepaid $\sim$ Bernoulli of the logit with `coef = [-2, 20, 0.01]`; the fit recovers each coefficient to within 3
standard errors (the incentive coefficient's standard error is near 0.35, so the tolerance is about 1.0).

## Sources

- Drechsler, Savov and Schnabl (2021), J. Finance 76(3):1091-1143, doi:10.1111/jofi.13013.
- Kalkbrener and Willing (2004), J. Banking and Finance 28(7):1547-1568, doi:10.1016/S0378-4266(03)00131-6 (the stochastic NMD model;
  the exact SDEs were not read in full text, UNVERIFIED detail).
- Jarrow and van Deventer (1998), J. Banking and Finance 22(3):249-272, doi:10.1016/S0378-4266(97)00058-7.
- BCBS d368 (2016) footnote 38, https://www.bis.org/bcbs/publ/d368.htm.
- Fuster, Lucca and Vickery (2022), FRBNY Staff Report 1001, https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1001.pdf.

## Limits

- The golden recovery of the logit coefficients depends on the simulation design; the design is stated above and the tolerance is
  3 standard errors of the fit, not an absolute number.
- Chilean mortgages are mostly UF-indexed with prepayment fees: the incentive function differs and no Chilean primary calibration
  source was checked (UNVERIFIED, dossier 05 D.4).
- A replicating portfolio fitted on one window is not stable evidence: the weight stability across windows is a validation test
  (page 08), and the SVB lesson is that duration assumptions can be tuned to fit limits.
