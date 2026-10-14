# 02. The copulas

Module `riskvalidation.portfolio.copula`. Source: dossier 04 H.3 (structural models, the Gaussian and Student-t copula) and the
ladder rungs C2 and C3.

## The models

Gaussian copula (multi-factor), with $Z\sim N(0,\Sigma)$ and independent standard normal $e_i$:

$$A_i=\beta_i^\top Z+\sqrt{1-\beta_i^\top\Sigma\beta_i}\;e_i,\qquad \text{name } i \text{ defaults iff } A_i<\Phi^{-1}(\mathrm{PD}_i),$$

and then loses $\mathrm{EAD}_i\,\mathrm{LGD}_i$. The copula framing of default correlation is Li (2000), "On default correlation: a
copula function approach", Journal of Fixed Income 9(4):43-54, doi:10.3905/jfi.2000.319253. The dossier writes the scale of $e_i$ as
$\sqrt{1-\beta_i^\top\beta_i}$; the two are equal for one factor and for uncorrelated factors, and for every portfolio
`portfolio_from_groups` builds (a name loads on one factor). For a name that loads on correlated factors only the form with
$\Sigma$ keeps $A_i$ at unit variance, so that the marginal default probability is $\mathrm{PD}_i$ (checked by simulation), and the engine
uses it.

Student-t copula, with $W=\nu/\chi^2_\nu$ independent of $Z$ and $e$:

$$A_i=\sqrt W\Big(\beta_i^\top Z+\sqrt{1-\beta_i^\top\Sigma\beta_i}\,e_i\Big),\qquad \text{default iff } A_i<t_\nu^{-1}(\mathrm{PD}_i),\quad \nu>2.$$

It adds tail dependence, so joint defaults cluster more than in the Gaussian case for the same asset correlation: Frey and McNeil
(2003), "Dependent defaults in models of portfolio credit risk", Journal of Risk 6(1):59-92, doi:10.21314/jor.2003.089; Demarta and
McNeil (2005), "The t copula and related copulas", International Statistical Review 73(1):111-129,
doi:10.1111/j.1751-5823.2005.tb00254.x.

## Seeds, chunks and memory

`simulate_gaussian` and `simulate_student_t` draw with `numpy.random.Generator(numpy.random.PCG64(seed))` and read no global state. The
result is reproducible for the same `(seed, n_scenarios, chunk_scenarios)`; a different chunk size gives a different but equally
valid sample (the document records `chunk_size`). The scenarios are simulated in row blocks of `chunk_scenarios // 3`, so that the
two float64 working arrays and the default flags never exceed `chunk_scenarios * N * 8` bytes (about 0.8 GB for the default
100,000 scenarios and 1000 names, the bound of the unit sizing); a test measures the traced peak. `track_groups=True` also returns the
loss of each group, which adds up to the loss of the scenario.

## Monte Carlo standard errors

`mc_standard_errors` splits the sample in `n_batches` consecutive equal batches (the remainder is dropped), computes VaR and ES per
batch with the sample estimators of page 01, and returns

$$\mathrm{se}=\frac{\mathrm{sd}(\text{batch estimates},\,\mathrm{ddof}=1)}{\sqrt{n_{\text{batches}}}}.$$

The batch-means estimator is the contract's choice, the dossier names none; its size is tested against the spread of the estimator over 100
independent samples. A VaR on a lattice (for example 25 identical unit names) can be the same in every batch, and its standard error is then exactly 0:
read it as a lattice, not as a certain quantile. At least 2 batches of 10 losses are required. The
standard errors fall as $1/\sqrt S$ (tested on $10^5$ and $4\times10^5$ losses).

## Goldens and gates

- Joint default of two names with PD 0.01 and $\rho=0.3$: Gaussian 0.000556 ($E[p(Z)^2]$ by quadrature); Student-t with $\nu=5$
  0.001647 (a two-dimensional quadrature of the mixture representation, 0.0016473), above the Gaussian one (the gate of rung C3).
  The simulation at $2\times10^7$ scenarios lies within 4 standard errors of both.
- The marginal default probability of each name is its PD and the mean loss is the EL within 4 standard errors (gate of rung C2).
- Convergence to the anchor (gate of rung C2): for identical names (PD 0.02, $\rho$ 0.15) the Monte Carlo VaR per unit of EAD sits
  above `lhp_quantile` for 25 names (the granularity adds to the quantile), the gap shrinks for 200 and 1000 names, and at 1000 names
  it is within its Monte Carlo interval at 99 and 99.9 percent.

## Limits

Default mode only. A Student-t copula with the same asset correlation changes the tail but not the marginals; $\nu$ is an input and
is poorly identified from data (ladder failure mode). Monte Carlo noise at 99.9 percent needs about $10^6$ scenarios; importance
sampling (page 04) reduces the count. The multi-factor correlation between sectors is an input to `portfolio_from_groups`, never a
default.
