# 01. Distributions and truncation

Module `riskvalidation.oprisk.distributions`. Source: dossier 05 F.2.

## The annual loss and its laws

For one unit of measure (a business line by event type) the annual loss is

$$Z = \sum_{i=1}^{N} X_i,$$

with $N$ the annual count and $X_i$ i.i.d. severities independent of $N$.

Symbols: $N$ the annual count, $X_i$ the severities, $\lambda$ the expected losses per year.

| Frequency law | Parameters | pmf |
|---|---|---|
| `poisson` | $(\lambda)$ | $p_n = e^{-\lambda}\lambda^n / n!$ |
| `negative-binomial` | $(r, q)$ | $p_n = \frac{\Gamma(n+r)}{n!\,\Gamma(r)}\,q^r (1-q)^n$ |
| `binomial` | $(m, q)$ | $p_n = \binom{m}{n} q^n (1-q)^{m-n}$ |

The three laws are those of Shevchenko Table 1 and belong to the $(a, b, 0)$ class of page 02: $p_n = (a + b/n)\,p_{n-1}$.
`frequency_pmf(law, params, n_max)` returns $p_0 \dots p_{n_{max}}$ and does not add the tail beyond $n_{max}$. Golden values:
Poisson(2) gives 0.135335, 0.270671, 0.270671; negative binomial $(2, 0.5)$ gives 0.25, 0.25, 0.1875; binomial $(2, 0.5)$ gives
0.25, 0.5, 0.25.

| Severity law | Parameters | Distribution |
|---|---|---|
| `lognormal` | $(\mu, \sigma)$ | $F(x) = \Phi\big((\ln x - \mu)/\sigma\big)$ |
| `weibull` | (shape $k$, scale $s$) | $F(x) = 1 - e^{-(x/s)^k}$ |
| `gpd` | $(\xi, \beta, \text{location})$ | $Y = X - \text{location}$, $G(y) = 1 - (1 + \xi y/\beta)^{-1/\xi}$, $\xi > 0$ |
| `lognormal-gpd-splice` | $(\mu, \sigma, u, \xi, \beta, w)$ | below $u$ the lognormal body, above it the GPD tail |

The body-tail splice with threshold $u$ and body weight $w = P(X \le u)$ is (dossier 05 F.2, the standard EVT construction;
Chavez-Demoulin, Embrechts and Neslehova 2006, doi:10.1016/j.jbankfin.2005.11.008):

$$f(x) = w\,\frac{f_b(x)}{F_b(u)}\ \ (x \le u),\qquad f(x) = (1-w)\,g_{\xi,\beta}(x-u)\ \ (x > u),$$

so $F(u) = w$ and $F$ is continuous at $u$. `severity_cdf` is zero at and below the origin of each law (0, or the location of
the GPD); `severity_ppf` is its inverse (`q = 0` gives the origin, `q = 1` gives `inf`; for the GPD
$x = \text{location} + \frac{\beta}{\xi}\big((1-q)^{-\xi} - 1\big)$). Golden values: lognormal $(0, 2)$ has $F(1) = 0.5$;
$\mathrm{GPD}(0.5, 100, 0)$ has $F(100) = 1 - 1.5^{-2} = 0.555556$ and $F^{-1}(0.99) = 1800$; Weibull $(1, 1)$ has
$F(1) = 0.632121$.

## Moments and the infinite mean

`severity_moment(law, params, k)` is the raw moment $E[X^k]$: the lognormal $e^{k\mu + k^2\sigma^2/2}$, the Weibull
$s^k\Gamma(1 + k/\text{shape})$, the GPD $\sum_j \binom{k}{j}\,\text{location}^{k-j}E[Y^j]$ with
$E[Y^j] = \beta^j j! / \prod_{i=1}^{j}(1 - i\xi)$, finite exactly when $k\xi < 1$. **A tail index of 1 or more has no mean**
(dossier 05 F.6 and F.4; Degen 2010; Chavez-Demoulin, Embrechts and Neslehova 2006): the function returns `math.inf` for a moment
that does not exist, and every mean-based quantity built on it (page 03) refuses. Golden values: lognormal $(0, 2)$, $k=1$:
$e^2 = 7.389056$; $\mathrm{GPD}(0.25, 100, 0)$: 133.333333 and 53333.333333; Weibull $(2, 1)$: 0.886227;
$\mathrm{GPD}(1.0, 100, 0)$, $k = 1$: `inf`.

## Truncation at the collection threshold

Internal data are collected above a threshold $H$ (for example EUR 20,000 or 10,000), so the severity must be fitted as a
truncated distribution $f(x)/(1 - F(H))$ and the frequency rescaled (Mignola and Ugoccioni 2007,
doi:10.21314/JOP.2006.019; Cope, Mignola, Antonini and Ugoccioni 2009, doi:10.21314/JOP.2009.069):

$$\lambda_{full} = \frac{\lambda_{obs}}{1 - F(H)},\qquad F_H(x) = \frac{F(x) - F(H)}{1 - F(H)}\ \ (x \ge H).$$

Ignoring this biases capital. `truncated_severity_cdf` is $F_H$; for the GPD with $H$ above the location it equals the GPD with
$\beta' = \beta + \xi(H - \text{location})$ and location $H$ (threshold stability). Golden values: $\mathrm{GPD}(0.5, 100, 0)$ with
$H = 100$ has $\beta' = 150$ and $F_H(250) = 0.555556 = (0.802469 - 0.555556)/0.444444$; with $\lambda_{obs} = 10$ the
rescaled frequency is $10 / 0.444444 = 22.5$ (`rescale_frequency`).

Symbols: $H$ the collection threshold, $F$ the severity cdf of the full distribution, $\lambda_{obs}$ the recorded losses per
year above $H$, $\lambda_{full}$ the losses per year of the full severity.

## The fits

`fit_frequency(counts, law)` fits the recorded annual counts by maximum likelihood: Poisson $\hat\lambda = \bar n$ with
$\mathrm{se} = \sqrt{\hat\lambda/Y}$ (golden counts 3, 5, 4: 4.0 and 1.154701); the negative binomial needs a sample variance
above the mean (otherwise `ValueError("underdispersed")`: use Poisson). The fit is of the recorded frequency, to be rescaled by
`rescale_frequency`.

`fit_severity(law, losses, collection_threshold, truncated)` fits the lognormal, the Weibull and the GPD to losses at or above
the threshold. With `truncated=True` the log-likelihood is

$$\ell = \sum_i \ln f(x_i) - N \ln\big(1 - F(H)\big);$$

with `truncated=False` it is $\sum_i \ln f(x_i)$ and the threshold is ignored (rung 1, the biased baseline). The untruncated
lognormal is closed form, $\hat\mu = \overline{\ln x}$ and $\hat\sigma = \sqrt{\overline{(\ln x - \hat\mu)^2}}$ (divisor $N$);
golden losses $1, e, e^2$ give $\hat\mu = 1.0$, $\hat\sigma = 0.816497$ and $\mathrm{se}(\hat\mu) = 0.471405$. The GPD is
fitted to the excesses over the threshold (its location), by profiling the ratio $\theta = \xi/\beta$ (Grimshaw's reduction) over a
log grid and refining; the Weibull and the truncated lognormal by Nelder-Mead. Standard errors come from the inverse of the
observed information (a central-difference Hessian); a parameter on a boundary or a singular information matrix gives `nan` and
`converged = False`, never a silent number. A loss below the collection threshold raises. The truncated lognormal fit degenerates
when the threshold sits deep in the body (a single lognormal cannot reproduce the LDCE percentiles, dossier 05 F.6).

`fit_splice(losses, H, u, min_tail)` fits the lognormal body truncated to $[H, u]$ to the losses at or below $u$ and the GPD to
the excesses over $u$. The body weight follows from the observed tail share $t = n_{tail}/N$ and
$c = F_b(H)/F_b(u)$: since $P(X > u) = 1 - w$ and $P(X \ge H) = 1 - wc$,

$$\frac{1-w}{1-wc} = t\quad\Longrightarrow\quad w = \frac{1-t}{1-tc}.$$

Golden: $t = 0.2$, $c = 0.5$ give $w = 0.888889$. The standard errors of $u$ (fixed by the caller) and of $w$ (derived) are `nan`.

`tail_index_profile(losses, thresholds, min_exceedances)` fits the GPD to the excesses of each strictly increasing threshold (the
threshold-stability plot of dossier 05 F.3 and the ladder); for exact GPD data the modified scale $\hat\beta(u) - \hat\xi(u)\,u$ is
constant in expectation. A threshold with fewer exceedances than the minimum is listed with `nan` and `converged` false, never
dropped.

## Limits

- The laws and their parameter order are fixed by the contract; a Weibull is not a heavy-tailed severity and the engine does not
  judge the choice of law.
- A GPD fit has $\xi > 0$ by construction; a sample whose likelihood is maximal at $\xi \to 0$ (exponential-like excesses) is
  reported as unconverged.
- The binomial frequency has no fit (it is under-dispersed); the caller chooses Poisson or negative binomial.
- Maximum likelihood at a sample of a few hundred losses is noisy in the tail index; the profile page exists to show it.

Sources: dossier 05 F.2 and F.3; Mignola and Ugoccioni (2007), doi:10.21314/JOP.2006.019; Cope et al. (2009),
doi:10.21314/JOP.2009.069; Chavez-Demoulin, Embrechts and Neslehova (2006), doi:10.1016/j.jbankfin.2005.11.008; Shevchenko (2010),
Table 1, arXiv 1008.1108, https://arxiv.org/abs/1008.1108.
