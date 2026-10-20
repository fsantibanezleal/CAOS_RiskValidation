# 01. Simple aggregation

Module `riskvalidation.aggregation.simple`. Sources: dossier 05 G.1 (items 1, 2 and 4 of the ladder) and G.4 (the allocation
and diversification quantities); Tasche (2008), "Capital allocation to business units and sub-portfolios: the Euler principle",
https://arxiv.org/abs/0708.2542 (equation 3.4); Rosenberg and Schuermann (2006), J. Financial Economics 79(3):569-614,
doi:10.1016/j.jfineco.2005.03.001; dossier 04 l.652-653 for the capital basis.

## The ladder's first rungs

$$EC_{tot}^{sum}=\sum_{k=1}^{K}EC_k,\qquad EC_{tot}^{vc}=\sqrt{EC^\top R\,EC},\qquad EC_k=\mathrm{VaR}_q(L_k)-EL_k$$

Symbols: $K$ the number of risks (4 in the case: credit, market, IRRBB, operational), $EC$ the vector of marginal economic capitals
(each at least 0), $R$ the inter-risk correlation matrix, $EL_k$ the expected loss of risk $k$. The capital basis is VaR - EL: the
unexpected-loss basis of the BCBS 2005 explanatory note, section 4.4, "$EC_q = \mathrm{VaR}_q(L) - EL$" (dossier 04 l.652-653).

- `simple_sum(ec)` is the headline capital of the case. Perfect dependence for the capital, but not an upper bound for the VaR in
  general (VaR is not subadditive, dossier 05 G.3). Equals `variance_covariance(ec, R)` for $R$ all ones.
- `check_correlation(corr, tol=1e-9)` raises `ValueError` unless the matrix is square, symmetric to `tol`, has a unit diagonal to
  `tol`, every entry in $[-1,1]$ and a smallest eigenvalue of at least `-tol` (positive semi-definite).
- `variance_covariance(ec, corr)` is exact for elliptical losses with a common quantile scaling, an approximation otherwise
  (dossier 05 G.1). It checks the correlation first, lies between $\sqrt{\sum EC_k^2}$ ($R$ the identity) and the simple sum ($R$ all
  ones) for a nonnegative $R$, is invariant under a joint permutation of $EC$ and $R$ and homogeneous of degree 1 in $EC$.
- `hybrid_quantile(samples, q)` is the "H-VaR" of Rosenberg and Schuermann (dossier 05 G.1 item 4): $\sqrt{v^\top R_{emp}\,v}$ with
  $v_k=\mathrm{VaR}_q(\text{column }k)$ from `portfolio.portfolio.var_from_sample` and $R_{emp}$ the Pearson correlation of the
  columns. The dossier names the approach and not the estimator: the Pearson correlation is the contract's definition, UNVERIFIED
  against the paper. It is at most $\sum_k v_k$ and refuses fewer than two rows, a constant column and a negative column quantile (the square-root formula needs nonnegative quantiles).

## Diversification and allocation

$$DF=\frac{\rho(L)}{\sum_i\rho(L_i)},\qquad B_i=1-\frac{\rho(L_i\mid L)}{\rho(L_i)},\qquad \sigma_c(L_i\mid L)=c\,\frac{\mathrm{cov}[L_i,L]}{\sqrt{\mathrm{var}[L]}}$$

Symbols: $\rho$ a risk measure, $\rho(L_i\mid L)$ the Euler contribution of unit $i$ (dossier 05 G.4; Tasche eqs. 3.4, 3.7, 3.11),
$c$ a multiplier, $\mathrm{cov}[L_i,L]=\sum_j \Sigma_{ij}$ for the covariance matrix $\Sigma$ of $(L_1,\dots,L_K)$.

- `diversification_factor(rho_total, rho_marginals)` is not forced below 1: VaR is not subadditive, so the sign of every benefit is
  reported by `validation.aggregation.diversification_sign` and never repaired here. A zero or negative denominator is refused.
- `euler_diversification_benefit(rho_contribution, rho_standalone)` is zero when the contribution equals the standalone measure; a
  nonpositive standalone measure is refused.
- `std_contributions(cov, c)` is the closed-form Euler allocation of the standard deviation measure, so that the
  variance-covariance total allocates exactly: the contributions sum to $c\sqrt{\mathrm{var}[L]}$ to $10^{-12}$ relative. A
  non-symmetric or not positive semi-definite covariance and a zero variance are refused.

## Golden values

| Function | Input | Value |
|---|---|---|
| `simple_sum` | (3, 4) | 7.0 |
| `variance_covariance` | (3, 4), $R=I$ | 5.0 |
| `variance_covariance` | (3, 4), $R$ all ones | 7.0 |
| `variance_covariance` | (3, 4), $R=[[1,0.5],[0.5,1]]$ | $\sqrt{37}=6.0828$ (computed) |
| `check_correlation` | $[[1,0.9,-0.9],[0.9,1,0.9],[-0.9,0.9,1]]$ | refused: smallest eigenvalue $-0.8$ |
| `diversification_factor` | (6.0828, [3, 4]) | 0.8690 (computed, 6.0828 / 7) |
| `euler_diversification_benefit` | (6, 8) | 0.25 |
| `std_contributions` | cov $[[4,2],[2,9]]$, $c=1$ | (1.4552, 2.6679), sum $\sqrt{17}=4.1231$ (computed) |
| `hybrid_quantile` | columns $x$ and $2x$ | $3\,q_x$; two columns of correlation 0: $\sqrt{v_1^2+v_2^2}$ |

The tests recompute each value through a second route (the closed form, a plain search, or an eigenvalue computation), never by
typing the answer.

## Limits

The variance-covariance formula and the hybrid quantile are approximations outside the elliptical case. The Euler contributions
of the standard deviation are exact for the standard deviation measure only; the VaR and ES contributions are those of
`portfolio.euler` (reused unchanged). The engine states no compliance claim and no figure of this page is a regulatory capital.
