# 02. The aggregate loss

Module `riskvalidation.oprisk.aggregate`. Source: dossier 05 F.3, verified there from Shevchenko (2010), "Implementing loss
distribution approach for operational risk", arXiv 1008.1108 (https://arxiv.org/abs/1008.1108), J. Operational Risk 5(2):3-40,
doi:10.21314/JOP.2010.077.

The compound annual loss $Z = \sum_{i=1}^{N} X_i$ is computed three ways, which must agree.

## Monte Carlo

`simulate_annual_losses` draws the annual count, then that many severities, and sums them. The quantile error decays as
$O(K^{-1/2})$ in the number of simulated years $K$ and needs very large $K$ at 99.9 percent (dossier 05 F.3). The counts of all
years come from one `PCG64` stream and the severities (by the inverse cdf of uniforms) from a second stream consumed in year
order, both spawned from `SeedSequence(seed)`: the result is deterministic for `(n_years, seed)` and independent of
`chunk_years`, which only bounds memory. `mc_quantile(losses, level)` returns the $\lceil n\cdot\text{level}\rceil$-th order
statistic and its standard error, half the distance between the order statistics at $\lceil n\ell + s\rceil$ and
$\lfloor n\ell - s\rfloor$ with $s = \sqrt{n\ell(1-\ell)}$ (a one-standard-deviation binomial interval). Golden values: the
sample $1, \dots, 1000$ gives $(500, 16.0)$ at level 0.5 and $(999, 1.0)$ at 0.999. For Poisson(100) with LN(0, 2) and one
million years the sample mean is within 4 standard errors ($0.545985$ each) of 738.9056, that is within 2.184.

## Panjer recursion

With a discretised severity $f_j = P(X = j\Delta)$ and a frequency of the $(a, b, 0)$ class, $p_n = (a + b/n)\,p_{n-1}$, the
compound probabilities $h_n = P(Z = n\Delta)$ satisfy (Panjer 1981, doi:10.1017/S0515036100006796; Shevchenko Theorem 5.2,
equation 36)

$$h_n = \frac{1}{1 - a f_0}\sum_{j=1}^{n}\Big(a + \frac{bj}{n}\Big) f_j\, h_{n-j},\qquad h_0 = \sum_{k=0}^{\infty} f_0^k\, p_k .$$

| Law | $a$ | $b$ | $h_0$ |
|---|---|---|---|
| Poisson($\lambda$) | $0$ | $\lambda$ | $e^{\lambda(f_0 - 1)}$ |
| NegBin($r, q$) | $1-q$ | $(1-q)(r-1)$ | $\big(1 + (1-f_0)\frac{1-q}{q}\big)^{-r}$ |
| Binomial($m, q$) | $-q/(1-q)$ | $q(m+1)/(1-q)$ | $\big(1 + q(f_0 - 1)\big)^m$ |

(Shevchenko Table 1.) Symbols: $\Delta$ the step, $f_j$ the severity mass at $j\Delta$, $h_n$ the compound mass at $n\Delta$, $(a, b)$
the recursion coefficients, $G$ the number of grid points. Golden values: $(a, b)$ of Poisson(100) $= (0, 100)$, of NegBin(2, 0.5)
$= (0.5, 0.5)$, of Binomial(2, 0.5) $= (-1, 3)$; $h_0$ with $f_0 = 0.5$ is 0.367879, 0.444444 and 0.5625; a severity of 0 or 1 with
equal probability thins Poisson(2) to Poisson(1), $h = [0.367879, 0.367879, 0.183940, \dots]$. The cost is $O(G^2)$ and the
recursion is numerically stable for the Poisson and the negative binomial. It raises when $a f_0 = 1$, for a law outside the class
and when $h_0$ underflows to zero (a large expected count would make every $h_n$ a silent zero: use the FFT).

`discretise_severity(law, params, step, n_points, method)` returns the masses on the grid: central $f_0 = F(\Delta/2)$,
$f_j = F((j+\tfrac12)\Delta) - F((j-\tfrac12)\Delta)$ (the benchmark's, Shevchenko eq. 37); forward $f_j = F((j+1)\Delta) - F(j\Delta)$
(an upper bound, eq. 38); backward $f_0 = 0$, $f_j = F(j\Delta) - F((j-1)\Delta)$ (a lower bound). The sum is at most 1: **the mass
beyond the grid is reported (it is $1 - \sum f_j$), never renormalised or extrapolated.**

## FFT

The transform of the zero-padded severity is passed through the probability generating function of the frequency and inverted,
$\hat h = \psi(\hat f)$ (Shevchenko section 6.2): $\psi(z) = e^{\lambda(z-1)}$ for the Poisson, $\big(q/(1-(1-q)z)\big)^r$ for the
negative binomial, $(1 + q(z-1))^m$ for the binomial.

$$h = \mathcal{F}^{-1}\big[\psi\big(\mathcal{F}[f_{padded}]\big)\big]_{0..G-1}$$

The transform length is `padding_factor * G` rounded up to a power of two (zero padding guards against aliasing); optional
exponential tilting with $\theta \ge 0$ multiplies the severity by $e^{-\theta j}$ before and the result by $e^{\theta n}$ after.
Negative values beyond $-10^{-12}$ are aliasing or an unstable transform and raise; smaller negative roundoff is clipped to 0. On
the benchmark the FFT with the default padding of 2 agrees with the Panjer recursion to $10^{-9}$ per entry (the difference is
about $3\times10^{-14}$); without padding the heavy tail wraps around and the result is wrong.

## What is read from the pmf

`quantile_from_pmf(h, step, level)` is $n\Delta$ for the smallest zero-based $n$ with $\sum_{k\le n} h_k \ge \text{level}$; it
raises when the cumulative sum never reaches the level (mass beyond the grid). `exceedance_from_pmf` returns
$x_n = n\Delta$ and $\mathrm{sf}_n = P(Z > x_n) = 1 - \sum_{k\le n} h_k$ clipped at 0 (golden $h = [0.5, 0.3, 0.2]$ gives
$[0.5, 0.2, 0.0]$).

`es_from_pmf(h, step, level, max_missing_share)` is the expected shortfall

$$\mathrm{ES} = \frac{E[Z; Z > q] + q\,\big(P(Z \le q) - \text{level}\big)}{1 - \text{level}} = q + \frac{E[(Z-q)^+]}{1-\text{level}},\qquad q = \mathrm{VaR}.$$

Golden: $h = [0.5, 0.3, 0.2]$, step 10, level 0.75 gives $q = 10$, $E[Z; Z>q] = 4$ and ES $= 18.0$. **The grid must carry the tail:**
the function raises when the mass beyond the grid exceeds `max_missing_share` (default $10^{-3}$) of the tail $1 - \text{level}$,
because an expected shortfall read off a grid that ends before the tail is too small and can fall below its own VaR. On the
benchmark's grid of 6000 points the missing mass is 0.000936 and a grid-only reading would be 377.8 against a VaR of 5849. The
expected shortfall does not exist for a severity with a tail index of 1 or more (the compound loss has an infinite mean), and
**the grid guard cannot detect that**: the pmf alone does not reveal the tail index, and on a coarse enough grid the missing mass
of such a severity falls below the tolerance (GPD(1.1, 100, 0), Poisson(10), step 1e6 and 32768 points miss 1.7e-7 and give a
finite 3.17e7 against a VaR of 2e6; on finer grids a Pareto-type tail with index 1.1 gives a finite grid-only value that keeps
growing with the grid: 818.6 at 4000 points, 1073.6 at 8000). The caller therefore decides from the severity, before the
expected shortfall, whether it exists: `es_for_severity(h, step, level, sev_law, sev_params)` raises `ValueError("infinite mean")`
when `severity_moment(sev_law, sev_params, 1)` is infinite, whatever the grid, and `es_from_pmf(..., infinite_mean=True)` does the
same for a caller that knows it from a fitted tail index or `CredibilityFit.infinite_mean`. The C17 bake gates on that, never on
the grid guard's `ValueError`, and writes the expected shortfall as null with the reason `infinite mean`.

`compound_poisson_moments(lam, sev_law, sev_params)` gives $E[Z] = \lambda E[X]$, $\mathrm{Var}[Z] = \lambda E[X^2]$ and skewness
$\lambda E[X^3]/\mathrm{Var}[Z]^{3/2}$, with the mean infinite when the severity mean is, the variance infinite when its second
moment is, and the skewness `nan` when the variance is infinite.

## Shevchenko's benchmark and its table values

Poisson($\lambda = 100$) with LN($\mu = 0$, $\sigma = 2$) (Shevchenko Example 9.4 and Tables 2 to 4 and 7): $E[Z] = 738.9056$
(exactly $100e^2$), $\mathrm{Var}[Z] = 298095.7987$ ($100e^8$), skewness $40.3428$ (the exact $e^6/10 = 40.342879$ cut at four
decimals). Panjer with the central discretisation of step 1 ($f_0 = 0.364456$, $f_1 = 0.215872$, $f_2 = 0.096248$; Table 2:
0.364455845, 0.215872117, 0.096248034; $h_0 = 2.504194\times10^{-28}$) gives, in Table 2 on p. 37, with the table's $n$ the
zero-based index of the recursion:

| $n$ | $H_n = \sum_{k \le n} h_k$ |
|---|---|
| 5847 | 0.998999329 (0.998999329180) |
| 5848 | 0.998999773 (0.998999773241) |
| 5849 | 0.999000217 (0.999000217026) |

so the smallest index with $H_n \ge 0.999$ is 5849 and the 99.9 percent quantile at step 1 is **5849** (Table 3). Dossier 05 F.3
first printed the row-5847 value against the label 5849; the erratum of 2026-10-13 corrected it from Table 2, and the engine
reproduces the table's values to $5\times10^{-10}$. The forward bound of Table 4 gives $H^U_{5811} = 0.998999719 < 0.999 \le
H^U_{5812} = 0.999000163$, so its quantile is 5812 (the paper's text says 5811; the table rows rule), and the backward bound
$H^L_{5849} = 0.998970962$. A Monte Carlo sample estimates the continuous quantile, 5853.1 (Table 3 at step 0.0625 gives
5853.06; Table 7: $H(5853.1) = 0.9990000$), not the discretised 5849 to 5850.

## Limits

- The discretisation error is the engine's to report, not to hide: a coarse step moves the quantile by up to one step, and the
  forward and backward bounds bracket it.
- Panjer is $O(G^2)$: a grid of 100,000 points is for the FFT, which needs the padding stated above.
- Monte Carlo at 99.9 percent needs very large $K$; its standard error from `mc_quantile` is the honest statement of that.
- The expected shortfall needs a grid that carries the tail; for a tail index of 1 or more there is none to carry.

Sources: dossier 05 F.3; Shevchenko (2010), arXiv 1008.1108, doi:10.21314/JOP.2010.077; Panjer (1981),
doi:10.1017/S0515036100006796.
