# 03. The single loss approximation and scenarios

Modules `riskvalidation.oprisk.sla` and `riskvalidation.oprisk.scenarios`. Source: dossier 05 F.4, F.5 and F.6.

## The single loss approximation (SLA)

For a subexponential severity $F$ and a frequency with $\sum_n (1+\epsilon)^n P(N=n) < \infty$,
$1 - H(z) \sim E[N]\,(1 - F(z))$ as $z \to \infty$ (Embrechts, Kluppelberg and Mikosch 1997, Theorem 1.3.9, as cited by Shevchenko
equation 69), hence

$$\mathrm{VaR}_\alpha[Z] \approx F^{-1}\!\Big(1 - \frac{1-\alpha}{E[N]}\Big),\qquad \alpha \to 1$$

(Shevchenko equation 70; Bocker and Kluppelberg, Risk 18(12):90-93, 2005, no DOI). `sla_var` returns it. For a Poisson count and a
GPD severity it is $\mathrm{VaR} \approx \text{location} + \frac{\beta}{\xi}\big[(\lambda/(1-\alpha))^{\xi} - 1\big]$
(`sla_var_poisson_gpd`), scaling like $\lambda^{\xi}$. Shevchenko shows that it converges to the exact quantile for heavy tails and
is poor for LN(0, 1) at practical quantiles. Symbols: $\alpha$ the level, $E[N]$ the expected count, $\lambda$ the Poisson mean,
$(\xi, \beta)$ the GPD tail index and scale.

Golden value: $\lambda = 10$, $\xi = 0.5$, $\beta = 100$, level 0.99 gives $200\,(31.622777 - 1) = 6124.555320$. Dossier 05 F.6
(reproduced in the review of the contract and recorded as decision D5): a GPD fitted for the excesses over EUR 20,000 by least
squares on the log of the excess quantiles to the four LDCE median percentiles gives $\xi = 0.96378$ and $\beta = 22349.4$, with
fitted quantiles 27409 / 42039 / 85025 / 412906, and at $\lambda = 41$ ($0.82\times50$), location 20,000 and level 0.999 an SLA of
**EUR 647.1 million**. Rounding $\xi$ to 0.96 and $\beta$ to 22,350 gives 624.1 million: that is the sensitivity of the SLA to
$\xi$ that dossier 05 F.6 warns of, not a different result.

### The mean-corrected form is UNVERIFIED

`sla_var_mean_corrected` adds $E[N]\,E[X]$ to the first-order value: $F^{-1}(1 - (1-\alpha)/E[N]) + E[N]\,E[X]$, the "commonly
quoted corrected form" of dossier 05 F.4. It was **not checked** against Bocker and Sprittulla, "Operational VaR: meaningful
means" (Risk 2006) or Degen (J. Operational Risk 5(4):3-17, 2010, doi:10.21314/JOP.2010.084): **UNVERIFIED**. The function exists
because the C17 ladder asks for rung 5, "SLA with and without correction"; the result (`SlaResult`) carries
`status = "UNVERIFIED"` and the docstring says so; the check against the primary text is an open item (a changed form is a new
minor version of the contract). Golden value: $E[N] = 10$, GPD(0.5, 100, 0), level 0.99 gives $E[X] = 200$, a correction of 2000 and
8124.555320. **With an infinite mean the correction does not exist:** for a GPD tail index of 1 or more the function raises
`ValueError("infinite mean")` (dossier 05 ladder failure modes: "$\xi \ge 1$ (infinite mean, SLA correction invalid)"); the C17 bake
then writes the rung as null with that reason. The first-order form still exists there.

## Scenario analysis and its combination with loss data

Scenarios are expert estimates of the frequency and severity of rare large events (dossier 05 F.5). Three methods combine them with
loss data, and validation must test the sensitivity of capital to each scenario point.

**(i) Pseudo-observations.** `scenario_pseudo_observations` returns each point's loss and its pseudo-count
$\text{exceedance\_frequency}\times n_{years}$ (points $(100, 0.1)$ and $(1000, 0.01)$ with 10 years give weights 1.0 and 0.1).

**(ii) The tail fitted to scenario quantiles.** `fit_tail_to_scenarios` fits a GPD with location equal to the threshold to the points
$(\text{loss}_s, p_s)$, where each point implies $P(X > \text{loss}\mid X > H) = \text{exceedance\_frequency}/\lambda_H$ and the
conditional quantile level is $p_s = 1 - \text{exceedance\_frequency}/\lambda_H$, by least squares on the log excess quantiles

$$\min_{\xi,\beta}\ \sum_s\Big(\ln\big(q_{model}(p_s) - H\big) - \ln(\text{loss}_s - H)\Big)^2,\qquad q_{model}(p) - H = \frac{\beta}{\xi}\big((1-p)^{-\xi} - 1\big),$$

the objective with which dossier 05 F.6 fits the LDCE percentiles; it is exact when the points lie on a GPD (the points $x = 200$,
$S = 0.25$ and $x = 1800$, $S = 0.01$ of GPD(0.5, 100, 0) recover $\xi = 0.5$, $\beta = 100$).

**(iii) Bayesian credibility (Shevchenko and Wuthrich 2006).** The model is the one-parameter Pareto above a known threshold $L$,
$f(x\mid\alpha) = \frac{\alpha}{L}(x/L)^{-\alpha-1}$, with $\alpha = 1/\xi$, and a Gamma prior on $\alpha$. With a
Gamma(shape, rate) prior and $N$ tail losses the posterior is Gamma(shape $+ N$, rate $+ \sum_i \ln(x_i/L)$) (section 3.3,
equation 24), and its mean is the credibility weighting of the maximum likelihood estimate and the prior mean (equation 25):

$$\bar\alpha = w\,\hat\alpha_{MLE} + (1 - w)\,\alpha_0,\qquad w = \frac{\sum_i \ln(x_i/L)}{\sum_i \ln(x_i/L) + \text{rate}},\qquad \hat\alpha_{MLE} = \frac{N}{\sum_i \ln(x_i/L)}.$$

`credibility_pareto` returns it with $\xi = 1/\bar\alpha$, $\beta = \xi L$ and `infinite_mean = (alpha_mean <= 1)`. Golden: prior
Gamma(2, 1), $L = 1$ and three losses $e$ give Gamma(5, 4), $\bar\alpha = 1.25 = 0.75\times1 + 0.25\times2$, $w = 0.75$,
posterior standard deviation $\sqrt5/4 = 0.559017$ and $\xi = 0.8$.

The prior comes from an expert's mean and interval (section 4.3): `gamma_prior_from_mean_interval` solves for the Gamma shape whose
probability on $[\text{lower}, \text{upper}]$ is the interval level (golden: mean 1, interval $[0.051293, 2.995732]$, level 0.9 gives
Gamma(1, 1)). `gamma_prior_from_scenarios` maps the scenario points to that input (decision D4): the scenario GPD of method (ii) gives
$\alpha_s = 1/\xi_s$ as the prior mean; a parametric bootstrap redraws each exceedance frequency as $k/\text{basis\_years}$ with $k$
Poisson (a count of 0 replaced by 0.5), refits the GPD, and the 5th and 95th percentiles of $\alpha$ form the expert interval; the
Gamma is then solved from the two. `basis_years`, the years of experience each scenario frequency is worth, is a policy parameter of
the caller; every point must be worth at least one expected event in it (`exceedance_frequency * basis_years >= 1`), and a refit that
sits on the search bound of the least squares fit is refused, because a bound is not a statistic (both raise `ValueError`). **The prior is unrestricted and says so** (`kind = "unrestricted-gamma-on-pareto-shape"` on every result): Shevchenko and
Wuthrich restrict it to $\alpha > 1$ (a finite mean, section 3.4), which would exclude the true value $\alpha = 0.909$ of the
tail-index 1.1 variant. A posterior with $\bar\alpha \le 1$ is flagged `infinite_mean` and no mean-based number follows.

`scenario_influence(points, capital_fn)` returns $(\text{capital(all)} - \text{capital(without } i))/\text{capital(all)}$ for each
point, calling the caller's capital function $P + 1$ times (golden: capital $=\sum \text{loss}\times\text{frequency}$ over $(100, 0.1)$,
$(1000, 0.01)$, $(50, 1.0)$ gives 0.142857, 0.142857, 0.714286).

## Limits

- The mean-corrected SLA is UNVERIFIED (above); do not present it as a verified result.
- The SLA converges only as $\alpha \to 1$ and for subexponential severities; it is poor for a lognormal at practical quantiles.
- Scenario points are expert judgement; the engine measures how far capital moves with each, it does not judge the points. The
  bootstrap interval reflects the scenario frequencies' sampling noise given `basis_years`, not the expert's actual uncertainty.
- The credibility model is a one-parameter Pareto above $L$, not a full severity; the body is fitted separately.

Sources: dossier 05 F.4 to F.6; Embrechts, Kluppelberg and Mikosch (1997); Bocker and Kluppelberg (2005); Degen (2010),
doi:10.21314/JOP.2010.084; Shevchenko and Wuthrich (2006), doi:10.21314/JOP.2006.016, preprint arXiv 0904.1067,
https://arxiv.org/abs/0904.1067; Shevchenko (2010), arXiv 1008.1108.
