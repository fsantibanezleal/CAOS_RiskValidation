# 02. Copulas

Module `riskvalidation.aggregation.copulas`. Sources: dossier 05 G.1 item 3 (copula aggregation, Sklar); Li (2000), "On default
correlation: a copula function approach", Journal of Fixed Income 9(4):43-54, doi:10.3905/jfi.2000.319253; Frey and McNeil (2003),
"Dependent defaults in models of portfolio credit risk", Journal of Risk 6(1):59-92, doi:10.21314/jor.2003.089; Demarta and McNeil
(2005), "The t copula and related copulas", International Statistical Review 73(1):111-129,
doi:10.1111/j.1751-5823.2005.tb00254.x; McNeil, Frey and Embrechts, *Quantitative Risk Management*, revised edition, Princeton
University Press, 2015 (chapter 7; the chapter number is UNVERIFIED in the dossier). The copula draw is restated here from the
sources the engine's own `portfolio.copula` docstring cites, because `portfolio.copula` keeps its sampler private.

## Aggregation by copula

By Sklar's theorem any joint distribution is $F(x_1,\dots,x_d)=C(F_1(x_1),\dots,F_d(x_d))$. The aggregate is simulated as

$$L_k=F_k^{-1}(U_k),\quad U\sim C,\qquad L=\sum_k L_k,$$

and its VaR, ES, EC $=$ VaR $-$ EL and Monte Carlo errors are those of the reused `portfolio.portfolio.measures_from_sample` on the
row sums (the batch-means errors of `portfolio.copula.mc_standard_errors`, both `None` below 200 draws).

Gaussian copula: $Z=A\,e$ with $AA^\top=R$ and $e$ independent standard normals, $U=\Phi(Z)$. Student t copula:
$W=\nu/\chi^2_\nu$ independent of $Z$, $X=\sqrt W\,Z$, $U=t_\nu(X)$, with $\nu>2$ (Demarta and McNeil 2005). The factor $A$ is the
Cholesky factor of $R$; for a singular positive semi-definite $R$ (the all-ones matrix of the comonotonic case, which has no
Cholesky factor) it is the eigen factor with the null eigenvalues set to zero and the rows rescaled to unit length, so that every
$Z_k$ is exactly standard normal and a unit correlation is exactly comonotonic. Both families have the Kendall rank correlation
$\tau=(2/\pi)\arcsin\rho$, which is 0.3333 for $\rho=0.5$ (computed).

Symbols: $R$ the correlation matrix, $\Phi$ the standard normal distribution function, $t_\nu$ the Student t distribution function
with $\nu$ degrees of freedom, $\chi^2_\nu$ a chi-square variable, $F_k^{-1}$ the marginal quantile function (money).

Randomness: one `numpy.random.Generator(numpy.random.PCG64(seed))` for the whole draw, in chunks of `chunk_size` rows (the working
memory of the draw is of the order `chunk_size * K * 8` bytes; the returned array is `n_draws * K * 8` bytes, 320 MB at $10^7$ draws
of 4 risks, the known maximum of the sizing). The result is reproducible for the same `(seed, n, chunk_size)`; for the Gaussian
family the stream does not depend on the chunk size, for the t family the chi-square draws interleave per chunk so another chunk
size is a different, equally valid sample.

## Marginal quantile functions

- `marginal_ppf_sample(sorted_losses)`: $\mathrm{ppf}(u)=\mathrm{interp}\big(u;\,(i-\tfrac12)/n,\ x_{(i)}\big)$, linear between the
  Hazen positions and clamped at the ends (the contract's definition).
- `marginal_ppf_pmf(grid, pmf)`: the smallest grid point whose cumulative probability is at least $u-10^{-12}$ (the rule of
  `oprisk.aggregate.quantile_from_pmf`).

## Tail dependence and Clayton

$$\lambda_t=2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right),\qquad C_\theta(u,v)=\left(u^{-\theta}+v^{-\theta}-1\right)^{-1/\theta},\qquad \lambda_L=2^{-1/\theta}$$

The Gaussian copula has no tail dependence for $|\rho|<1$ (dossier 05 G.1). Clayton is bivariate, with $\theta>0$; apply it to survival
or to losses as needed so that the dependence sits in the loss tail (dossier 05 G.1). Clayton sampling and vine copulas are deferred
(BL-078, decision Q13): only the closed forms are here.

## Golden values

| Function | Input | Value |
|---|---|---|
| `marginal_ppf_sample` | [0, 10] | 5.0 at 0.5, 0.0 at 0.1, 10.0 at 0.9 |
| `marginal_ppf_pmf` | grid [0, 1, 2], pmf [0.5, 0.3, 0.2] | 0, 1, 1, 2 at 0.5, 0.51, 0.8, 0.81 |
| `t_tail_dependence` | ($\rho=0.5,\nu=4$) | 0.2532 |
| `t_tail_dependence` | ($\rho=0,\nu=4$) | 0.0756 |
| `t_tail_dependence` | ($\rho=0.5,\nu=8$) | 0.1173 |
| `t_tail_dependence` | ($\rho=0.5,\nu=30$) | 0.0030 |
| `clayton_copula` | (0.5, 0.5, 1) | 0.3333 (1/3), and $C(u,1)=u$ |
| `clayton_lower_tail_dependence` | 1 and 2 | 0.5 and 0.7071 (computed) |

The tail dependence values are computed by Simpson quadrature of the $t_{\nu+1}$ density and checked against `scipy.stats.t` (the
tests do both). Invariants checked by test: the uniforms' Kolmogorov distance below $3/\sqrt n$ at $n=10^5$; the sample correlation of
two independent columns within $4/\sqrt S$ of 0; with $R$ all ones every row comonotonic; with all-ones $R$ the aggregate VaR equals
$\sum_k F_k^{-1}(q)$ within Monte Carlo error; with $R=I$ and two $U(0,1)$ marginals the total has the triangular law ($\mathrm{VaR}_{0.5}=1$);
with normal marginals and a Gaussian copula the VaR agrees with $\mu+z_q\sqrt{\sum_{ij}R_{ij}\sigma_i\sigma_j}$ within 5 standard errors.

## Limits

The Monte Carlo VaR at 99.9 percent has few exceedances at small `n_draws`: read it with its standard error. A marginal sample used
through `marginal_ppf_sample` cannot produce a loss beyond its largest observation. The t copula has the same Kendall rank
correlation as the Gaussian one: only its tail dependence differs.
