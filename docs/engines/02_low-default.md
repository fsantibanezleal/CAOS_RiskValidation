# 02. Low-default portfolios: most prudent estimation

`riskvalidation.engines.low_default`

Some portfolios (sovereigns, banks, large corporates, specialised lending) have so few defaults that a grade's observed
default rate is zero, or nearly so, for years. A PD of zero is not credible and the IRB framework floors it, but a
validator still needs an estimate that uses the information the portfolio has: how many obligors were observed without
defaulting, and the ordering of the grades. Pluto and Tasche, *Estimating probabilities of default for low default
portfolios* ([arXiv:cond-mat/0411699v3](https://arxiv.org/abs/cond-mat/0411699v3), 2004, revised 2005; book chapter in
Engelmann and Rauhmeier (eds.), *The Basel II Risk Parameters*, Springer 2006,
[doi:10.1007/3-540-33087-9_5](https://doi.org/10.1007/3-540-33087-9_5)), estimate each grade's PD as an upper confidence
bound under the most prudent assumption compatible with the ordering, then propose scaling those bounds and extending
them to several years.

![Most prudent PD bounds against the published values](../assets/most-prudent-bounds.svg)

## 1. The principle

Grades are ordered best first, $A, B, C, \dots$, and the ordering is assumed right: $p_A \le p_B \le p_C$. For grade
$A$ the most prudent assumption is that it is as risky as everything below it, $p_A = p_B = p_C$, so the obligors and
defaults of $A$, $B$ and $C$ are pooled; for $B$ the pool is $B$ and $C$; for $C$ it is $C$ alone. The bound for a grade
is the upper confidence bound at level $\gamma$ of a binomial proportion from its pool, so the bounds are monotone by
construction.

**No defaults, independent obligors** (equations 2.2b, 2.4b, 2.5b), with $n_A$, $n_B$, $n_C$ obligors:

$$\hat p_A = 1 - (1 - \gamma)^{1/(n_A + n_B + n_C)}, \qquad \hat p_B = 1 - (1 - \gamma)^{1/(n_B + n_C)}, \qquad \hat p_C = 1 - (1 - \gamma)^{1/n_C}.$$

**A few defaults** (section 3, equation 3.2): with $N$ pooled obligors and $D$ pooled defaults, the bound is the largest
$p$ at which observing at most $D$ defaults still has probability $1 - \gamma$,

$$1 - \gamma = \sum_{i=0}^{D} \binom{N}{i}\, p^i (1 - p)^{N - i},$$

the Clopper-Pearson upper bound, computed as the $\gamma$-quantile of $\mathrm{Beta}(D + 1, N - D)$ (the paper's A.1).

**Correlated defaults** (section 4, equations 4.1 and 4.3a): in the one-factor model of the
[Vasicek page](01_vasicek-lhp.md), obligor $i$ defaults when $\sqrt{\rho}\,Y + \sqrt{1 - \rho}\,\xi_i$ falls below
$\Phi^{-1}(p)$. Conditional on the systematic factor $Y = y$ defaults are independent with PD
$p(y) = \Phi\big((\Phi^{-1}(p) - \sqrt{\rho}\,y)/\sqrt{1 - \rho}\big)$, and the bound solves

$$1 - \gamma = \int_{-\infty}^{\infty} \varphi(y)\, \sum_{i=0}^{D}\binom{N}{i} p(y)^i\big(1 - p(y)\big)^{N - i}\, dy .$$

(The paper's 4.3b prints the argument of $\Phi$ as $G(p, \rho, y)$ and uses $G$ in the binomial terms; it must be read as
$\Phi(G)$, a probability, as in 6.7.) Correlation widens the bound: the zero-default observation is less informative
when defaults cluster.

## 2. Scaling (section 5)

With a few defaults every grade's bound lies above the portfolio's average default rate, "regarded as too conservative
by some practitioners". Section 5 rescales the bounds by one factor $K$ so that their obligor-weighted average meets a
target, keeping their ordering and ratios ("relative" PDs) and giving up their reading as confidence bounds:

$$K\,\frac{\hat p_A n_A + \hat p_B n_B + \hat p_C n_C}{n_A + n_B + n_C} = \mathrm{PD}_{\text{target}} \quad (5.1), \qquad \hat p_{X,\text{scaled}} = K\,\hat p_X \quad (5.2).$$

The target is the observed portfolio default rate (the central tendency; it needs at least one default, Tables 9 and
10), or, the paper's final proposal, the most prudent bound of the whole portfolio at the same level, which is the best
grade's bound $\hat p_A$ (Tables 11 and 12). The second "works for both the zero default and the few defaults case" and
keeps the degree of conservatism in the confidence level.

## 3. Several years (section 6)

A cohort of $N$ obligors observed for $T$ years, with $k$ defaults over the period. Obligor $i$'s asset change in year
$t$ is $V_{i,t} = \sqrt{\rho}\,S_t + \sqrt{1 - \rho}\,\xi_{i,t}$ (6.1); it defaults in year $t$ when $V$ first falls below
$\Phi^{-1}(p)$ (6.2 to 6.4), so $p$ is a long-run one-year PD. The yearly factors are standard normal with
$\mathrm{corr}(S_s, S_t) = \vartheta^{|s - t|}$ (6.5b). Given the factors an obligor defaults within the period with
probability

$$\pi(S_1, \dots, S_T) = 1 - \prod_{t=1}^{T}\big(1 - \Phi(G(p, \rho, S_t))\big) \quad (6.7, 6.8a),$$

and the bound is the largest $p$ with $1 - \gamma \le \sum_{\ell=0}^{k}\binom{N}{\ell}\,\mathrm{E}\big[\pi^{\ell}(1 - \pi)^{N - \ell}\big]$
(6.6, 6.8b). The paper's example takes $T = 5$ (the Basel minimum history), $\rho = 12\%$ and $\vartheta = 0.3$ (so
$\mathrm{corr}(S_1, S_5) = 0.81\%$).

## 4. Implementation

| Function | What |
|---|---|
| `most_prudent_pd(obligors, defaults, gamma, rho=0.0)` | sections 2 to 4: the bounds per grade, with the pooled obligors and defaults each rests on |
| `scale_bounds(bounds, obligors, target)`, `most_prudent_pd_scaled(..., target="upper_bound" \| "central_tendency" \| PD)` | section 5: $K$ and the scaled bounds |
| `most_prudent_pd_multiperiod(obligors, defaults, gamma, *, rho, theta, years, n_sim=100_000, seed=0)` | section 6, with `defaults` the defaults of each grade over the whole period |

The independent bound is the Beta quantile (SciPy). The one-period correlated integral is computed by Gauss-Hermite
quadrature with 256 nodes (probabilists' Hermite polynomials, `numpy.polynomial.hermite_e`), inside a bracketed root
search on $p$ (Brent's method, relative tolerance $10^{-12}$). The $T$-dimensional expectation of section 6 is a Monte
Carlo average over `n_sim` factor paths drawn once with `seed` and reused for every trial $p$, so the bound is a smooth,
reproducible function of the seed; the paper computed it by Monte Carlo too. The confidence level is the caller's: the
paper argues for "moderate confidence levels" (values below 95% "seem more appropriate") because the usual alternative,
averages of historical default rates, "yields a comparable probability of underestimating the true PD".

## 5. Reproduction of the published tables

The paper's example has three grades with $n_A = 100$, $n_B = 400$, $n_C = 300$ obligors, at six confidence levels,
with no defaults or with 0, 2 and 1 defaults in A, B and C. Percent, two decimals:

| $\gamma$ | 50% | 75% | 90% | 95% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| $\hat p_A$, no defaults, independent (Table 1) | 0.09 | 0.17 | 0.29 | 0.37 | 0.57 | 0.86 |
| $\hat p_B$, no defaults, independent (Table 2) | 0.10 | 0.20 | 0.33 | 0.43 | 0.66 | 0.98 |
| $\hat p_C$, no defaults, independent (Table 3) | 0.23 | 0.46 | 0.76 | 0.99 | 1.52 | 2.28 |
| $\hat p_A$, defaults 0, 2, 1 (Table 4) | 0.46 | 0.65 (exact 0.638) | 0.83 | 0.97 | 1.25 | 1.62 |
| $\hat p_B$, defaults 0, 2, 1 (Table 5) | 0.52 | 0.73 | 0.95 | 1.10 | 1.43 | 1.85 |
| $\hat p_C$, defaults 0, 2, 1 (Table 6) | 0.56 | 0.90 | 1.29 | 1.57 | 2.19 | 3.04 |
| $\hat p_A$, no defaults, $\rho = 12\%$ (Table 7) | 0.15 | 0.40 | 0.86 | 1.31 | 2.65 | 5.29 |
| $\hat p_B$, no defaults, $\rho = 12\%$ (Table 7) | 0.17 | 0.45 | 0.96 | 1.45 | 2.92 | 5.77 |
| $\hat p_C$, no defaults, $\rho = 12\%$ (Table 7) | 0.37 | 0.92 | 1.89 | 2.78 | 5.30 | 9.84 |
| $\hat p_A$, defaults 0, 2, 1, $\rho = 12\%$ (Table 8) | 0.72 (exact 0.711) | 1.42 | 2.50 | 3.42 | 5.88 | 10.08 |
| $\hat p_B$, defaults 0, 2, 1, $\rho = 12\%$ (Table 8) | 0.81 | 1.59 | 2.77 | 3.77 | 6.43 | 10.92 |
| $\hat p_C$, defaults 0, 2, 1, $\rho = 12\%$ (Table 8) | 0.84 | 1.76 | 3.19 | 4.41 | 7.68 | 13.14 |

What the tests (`tests/test_low_default.py`) establish, cell by cell:

- **Tables 1 to 6 (independent)**: 35 of the 36 cells exact at the printed precision. The 36th is Table 4 at 75%: the
  paper prints 0.65% for the pool of 800 obligors with 3 defaults, while the exact bound is 0.638%, since
  $\Pr[\mathrm{Bin}(800, 0.65\%) \le 3] = 0.237$, not 0.25 (0.65% is the bound at 76.3%). It is a computation, not a
  typesetting slip: Table 11 scales with the same 0.65%, so its 75% column (K 0.83; A 0.54, B 0.61, C 0.75) follows from
  it, where the exact inputs give K 0.82 and 0.52, 0.60, 0.73.
- **Table 7 (correlated, no defaults)**: 15 of 18 identical; three sit at a rounding boundary and differ by one unit, in
  both directions (C at 75%: 0.9252%; A at 99%: 2.6563%; B at 99.9%: 5.7650%).
- **Table 8 (correlated, defaults)**: 6 of 18 identical, and every printed cell lies at or above the exact bound, by up to
  0.0094 percentage points (A at 50%: 0.7106% exact, 0.72% printed). An adaptive quadrature (SciPy `quad`, relative
  tolerance $10^{-12}$) gives the same 0.7106% as the 256-node rule, and the paper's own Table 12 prints the same
  portfolio bound at 50% as 0.71%: the paper's integration for (4.3a) erred upward, conservatively.
- **Tables 9 to 12 (scaling)**: every K within one unit of its print, every scaled bound within two units. Table 9, whose
  inputs the paper computed exactly, is identical in 17 of its 18 bounds and all six K. Table 10 inherits Table 8's
  upward offset (K at 50%: 0.4674, printed 0.46), Table 11 the 0.65% above. One misprint: Table 12 prints 9.54% for B at
  99.9%, where the exact value is 9.44% and the paper's own K (0.87) times its own Table 8 bound (10.92%) is 9.50%.
- **Tables 13 and 14 (five years)**: the Monte Carlo bound agrees within 0.5% with a deterministic Gauss-Hermite product
  rule (12 nodes in each of the five factor dimensions), so the implementation solves the stated model. The printed
  values lie above it everywhere: by 1% to 5% at the 99% and 99.9% levels (A, no defaults, 99.9%: 0.534% against 0.55%;
  C, one default, 99.9%: 1.651% against 1.70%), and by more at low levels, where the print has one significant digit (A
  at 50%: 0.023% against 0.03%). At the printed bounds the model's probability of at most $k$ defaults is 0.00086 and
  0.00088 instead of 0.001 at 99.9%, and 0.42 and 0.47 instead of 0.50 at 50% (four cells checked): the same
  conservative direction as Table 8. The paper does not
  state its number of simulations, and no alternative reading of its correlation structure (an exchangeable 0.3, or
  $\vartheta = 0.5$) reproduces both ends of the tables. The gate is the deterministic check; the comparison with the
  print is recorded as it is.

## 6. Reading the bounds

- The bounds are conservative by design: they answer "how high could the PD be, given what was observed", not "what is
  the PD". The paper proposes them as a calibration method for whole low-default portfolios, ideally scaled to the
  portfolio's upper bound (section 5), and doubts their value as a backtest, "as the (e.g. purely expert based) average
  PDs per rating grade would normally be well below our proposed quantitative upper bounds" (section 7).
- They depend on the confidence level, the ordering assumption, the asset correlation and, for several years, the
  intertemporal correlation; every function returns the values it used.
- Open questions the paper itself lists (section 8): the appropriate confidence level, the number of defaults at which
  to switch to average-based estimates, and what non-monotone bounds say about the ranking. Bayesian estimation with an
  informative prior is the usual alternative (Kiefer 2006, OCC working paper 2006-2,
  [occ.gov](https://www.occ.gov/publications-and-resources/publications/economics/working-papers-archived/pub-econ-working-paper-2006-2.pdf);
  not reviewed for this page).

## References

- Pluto, K. and Tasche, D. (2005). Estimating probabilities of default for low default portfolios.
  [arXiv:cond-mat/0411699v3](https://arxiv.org/abs/cond-mat/0411699v3). In Engelmann, B. and Rauhmeier, R. (eds.), *The
  Basel II Risk Parameters*, Springer, 2006, [doi:10.1007/3-540-33087-9_5](https://doi.org/10.1007/3-540-33087-9_5);
  2nd edition 2011, [doi:10.1007/978-3-642-16114-8_5](https://doi.org/10.1007/978-3-642-16114-8_5).
- Clopper, C.J. and Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the
  binomial. *Biometrika* 26(4):404-413. [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404).
- Kiefer, N.M. (2006). Default estimation for low-default portfolios. OCC Economics working paper 2006-2,
  [occ.gov](https://www.occ.gov/publications-and-resources/publications/economics/working-papers-archived/pub-econ-working-paper-2006-2.pdf).
