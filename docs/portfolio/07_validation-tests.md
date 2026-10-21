# 07. The validation tests

Module `riskvalidation.validation.portfolio`. Source: dossier 05 G.5 (validation tests for an economic capital model: check the
allocation sums to the total, the sensitivity checks) and dossier 04 ladder rungs C2 and C4 for their gates.

Each test returns one `TestResult`. The acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each light rule is
in the docstring, its parameters are in `policy_version`, in `alpha_amber` and `alpha_red` where a p-value is used, and in `extras["thresholds"]` for
the others, so that a view can explain a light. A test that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null
statistic, p-value and metric in the serialised row, never a fabricated number. An invalid parameter raises `ValueError`.

| Test id | Null hypothesis | Statistic | Light |
|---|---|---|---|
| `portfolio.tail_backtest` | the number of evaluation years above the fitted $q$-quantile is Binomial$(n,1-q)$ | observed count at the primary level | p-value of $P(X\ge\text{observed})$ against the alphas |
| `portfolio.mc_convergence` | the Monte Carlo estimate equals the analytic anchor | $z=(\text{estimate}-\text{anchor})/\mathrm{se}$ | two-sided p-value against the alphas |
| `portfolio.euler_additivity` | the contributions add up to the total | the sum of the contributions | metric $\le$ tolerance is green, above is red |
| `portfolio.crp_recursion_vs_mc` | the recursion's distribution is the Monte Carlo sample's | Kolmogorov distance $D$ | p-value against the alphas; red for any negative entry |
| `portfolio.correlation_size_power` | rejection rate $=\alpha$ at the null correlation (size), the rate is the power elsewhere | size cell: $z$ of the rate; power cell: none | size cell by the p-value; power cell by the rate against the power gates |

## Definitions

The tail back-test counts exceedances of the fitted quantile and the other tests compare a statistic with its null law; the two formulas that carry the lights are

$$p_{\text{back-test}}=\Pr\big(X\ge x_{\text{obs}}\big),\ X\sim\mathrm{Binomial}(n,1-q),\qquad
z=\frac{\hat\theta-\theta_0}{\mathrm{se}},\ \ p=2\,(1-\Phi(|z|)).$$


- **Tail back-test.** For each level, `fitted_quantile` = `lhp_quantile(level, pd, rho)` of the fit, `observed` = the years with a rate strictly above it, `expected` $=n(1-q)$ and
  the upper binomial tail. `statistic` is the observed count at `primary_level`, `p_value` its tail, `metric` the expected count; `extras["levels"]` lists every level with the
  indexes of the exceeding years and `extras["n_missing"]` the years the table lacks (excluded, never interpolated). The binomial test on a few years has little power: the p-value is
  shown beside the counts. Light: green when $p\ge\alpha_{\text{amber}}$ (0.05), amber when $\alpha_{\text{red}}\le p<\alpha_{\text{amber}}$ (0.01), red below.
- **Monte Carlo convergence.** $p=2(1-\Phi(|z|))$, metric $=\text{estimate}-\text{anchor}$. The anchor is exact only for a large homogeneous portfolio: a finite portfolio sits above the large-portfolio
  value by the granularity effect (page 04), so compare an LHP-like portfolio and report the granularity gap apart.
- **Euler additivity.** metric $=|\sum\mathrm{RC}-\text{total}|/|\text{total}|$, no p-value; the ES contributions of `euler.es_contributions` pass at $10^{-6}$, the kernel VaR contributions at the
  tolerance the document states (0.01). Tasche (2008), arXiv 0708.2542; dossier 05 G.5. A zero total is not evaluated.
- **CreditRisk+ recursion against Monte Carlo.** $D=\max_n|F_{\text{pmf}}(n)-F_{\text{mc}}(n)|$, $p=2\sum_{j\ge1}(-1)^{j-1}e^{-2j^2SD^2}$ (the asymptotic Kolmogorov law, 1.0 at $D=0$; conservative for a
  discrete null, the contract's definition), `extras["negative_entries"]` the number of negative probabilities (any makes the light red). Fewer than 10 scenarios or a sample above the grid is not evaluated.
- **Correlation size and power.** rate, standard error $\sqrt{r(1-r)/R}$ and the Wilson 95 percent interval (Wilson 1927, JASA 22(158):209-212); at the size cell $z=(r-\alpha)/\sqrt{\alpha(1-\alpha)/R}$
  and the light by the p-value; at a power cell green when the rate is at least `power_green` (0.80), amber when at least `power_amber` (0.50), red below.

## Goldens

- Tail back-test, PD 0.003 and $\rho$ 0.06 (quantiles 0.007791, 0.012340, 0.020018 at 95, 99, 99.9 percent) and the rates (0.0045, 0.0085, 0.0160, 0.0120, 0.0080, 0.0050): one year above the 99 percent quantile, statistic 1.0,
  metric 0.06, p-value 0.058520 (green); with 0.0230 for the fourth year, two years, p-value 0.001460 (red).
- Monte Carlo convergence: estimate 0.1030, anchor 0.1000, se 0.0010: $z=3.0$, p 0.002700, metric 0.003, red; estimate 0.1015: $z=1.5$, p 0.133614, green.
- Euler additivity: (0.5, 0.3, 0.2) against 1.0: statistic 1.0, metric 0.0, green; (0.5, 0.3, 0.1): statistic 0.9, metric 0.1, red.
- CreditRisk+ recursion: pmf (0.5, 0.3, 0.2) against the counts (5, 3, 2): $D=0$, p 1.0, green; counts (3, 3, 4): $D=0.2$, p 0.818621, green.
- Size and power: $R=1000$, rate 0.052 at the size cell with $\alpha=0.05$: $z=0.290191$, p 0.771671, green; 830 rejections of 1000 at a power cell: 0.83, green; 400 of 1000: 0.4, red.

## Size and power of the p-value tests

The p-value tests of this page, measured on a generator where each null holds at the boundary (U16-E2, BL-088; the harness is `riskvalidation.harness`, the null and the alternative of each are named in `riskvalidation.generators.size_studies`, the runs are the `sizepower` suites under `tests/`, a local release gate that CI deselects). Size is the rejection rate at the nominal level over 4,000 repetitions, accepted when it is within the binomial bound alpha plus 3.09 standard errors; power is the rejection rate at 5% against one stated alternative, with no pass threshold. The figures below are the numbers of the run; the suite fails when a page states other ones.

**Size and power of `portfolio.tail_backtest`** (60 years at the 95% level; seed 5207, 4000 repetitions). Assumptions met: independent annual loss rates of an infinitely granular portfolio (PD 2%, correlation 0.12) and a fit equal to the truth, so each year exceeds the fitted 95% quantile with probability exactly 5.00%: the count is exactly binomial and the size is the exact rejection probability of a discrete test (2.97% at 5%, 0.98% at 1%), conservative by construction. The measured size is 2.90% at 5% (bound 6.06%) and 0.80% at 1% (bound 1.49%). Power at 5% against the alternative that the true asset correlation is 0.30, the fit says 0.12 (the 95% quantile is exceeded 9.12% of the years, not 5%) (seed 5307, 4000 repetitions): 29.93%.

**Size and power of `portfolio.tail_backtest`** (250 years at the 99% level; seed 5208, 4000 repetitions). Assumptions met: independent annual loss rates of an infinitely granular portfolio (PD 2%, correlation 0.12) and a fit equal to the truth, so each year exceeds the fitted 99% quantile with probability exactly 1.00%: the count is exactly binomial and the size is the exact rejection probability of a discrete test (4.12% at 5%, 0.40% at 1%), conservative by construction. The measured size is 4.25% at 5% (bound 6.06%) and 0.55% at 1% (bound 1.49%). Power at 5% against the alternative that the true asset correlation is 0.30, the fit says 0.12 (the 99% quantile is exceeded 4.29% of the years, not 1%) (seed 5308, 4000 repetitions): 96.05%.

**Size and power of `portfolio.mc_convergence`** (20000 scenarios, the 99% quantile of the LHP loss rate; seed 5209, 4000 repetitions). Anchor: the closed form of the large-portfolio quantile at PD 2% and correlation 0.12 (0.0917), computed from the normal functions, not by simulation. Estimate: the 99% sample quantile of 20,000 simulated loss rates, with its standard error estimated from the sample (Woodruff's order-statistic interval), as a caller does. The measured size is 5.22% at 5% (bound 6.06%) and 1.55% at 1% (bound 1.49%). Power at 5% against the alternative that the run simulates correlation 0.13, the anchor is the closed form at 0.12 (quantile 0.0964 against 0.0917) (seed 5309, 4000 repetitions): 74.12%. The measured size at 1% exceeds the bound: the excess comes from the estimated standard error, not from the z-test, and the diagnostic with the exact standard error of the sample quantile (seed 5409, 4,000 repetitions) measures 5.20% at 5% and 1.20% at 1%, within both bounds.

**Size and power of `portfolio.crp_recursion_vs_mc`** (2000 scenarios, 400 obligors in five exposure bands; seed 5210, 4000 repetitions). Assumptions met: the pmf of the stable recursion for 400 obligors in five exposure bands (200, 100, 60, 30 and 10 obligors of 1 to 5 loss units, PD 2%, one sector of variance 0.5); the sample is drawn from the model's definition (the Gamma factor, then Poisson counts per band), which shares no step with the recursion, and the generator's check holds the pmf's mean and variance to the closed form. The Kolmogorov law is asymptotic for a continuous distribution, so the test is conservative on integer losses. The measured size is 2.75% at 5% (bound 6.06%) and 0.47% at 1% (bound 1.49%). Power at 5% against the alternative that the sample has sector variance 0.6, the recursion's pmf 0.5 (loss variance 172 units against 150) (seed 5310, 4000 repetitions): 54.45%.

## What must not be claimed

These tests state no compliance claim and no regulatory threshold. A green light says the evidence at hand does not reject the null at the stated alphas, not that the
capital model is adequate; the back-test has low power with a few evaluation years, the Kolmogorov p-value is conservative for a discrete distribution, and the size and power
hold for the simulator's generator and series lengths only. The gates are Contraste internal gates.

## Limits

The tests judge outputs of `riskvalidation.portfolio`; they do not validate the choice of model, the portfolio data or the dependence assumption (dossier 05 G.5 also asks for
the copula-choice sensitivity and worst-case bounds, which the product shows by running the rungs side by side).
