# 03. CreditRisk+

Module `riskvalidation.portfolio.creditrisk_plus`. Source: dossier 04 H.4 and the ladder rung C4; Haaf, Reiss and Schoenmakers (2004),
"Numerically stable computation of CreditRisk+", Journal of Risk 6(4):1-10, doi:10.21314/jor.2004.097, preprint
https://www.wias-berlin.de/people/schoenma/hrs_JOR.pdf. The original CSFB document (1997) is not accessed (UNVERIFIED primary).

## The model

Exposures are integer multiples $\nu_i$ of a loss unit, $X=\sum_i\nu_iY_i$, $Y_i$ Poisson with stochastic intensity
$R_i=p_i\big(w_{0,i}+\sum_{k=1}^{K}w_{k,i}S_k\big)$, the sector factors $S_k$ independent Gamma with $E[S_k]=1$ and $\mathrm{var}(S_k)=\sigma_k^2$,
$w_{0,i}+\sum_kw_{k,i}=1$. The probability generating function (dossier 04 H.4):

$$G(z)=\exp\!\left(\sum_{i}w_{0,i}\,p_i\,(z^{\nu_i}-1)-\sum_{k=1}^{K}\frac{1}{\sigma_k^2}\ln\!\Big[1-\sigma_k^2\sum_{i}w_{k,i}\,p_i\,(z^{\nu_i}-1)\Big]\right).$$

`CreditRiskPlusInput` carries `nu`, `pd`, `w0`, `weights` (N, K), `sigma2`, `loss_unit` and `currency`, and raises unless the weights of
each obligor add up to 1. $K=0$ (a `(N, 0)` weight matrix and an empty `sigma2`) is the independent Poisson case.

## The stable recursion (`pmf_stable`)

Write $P_k(z)=\sum_iw_{k,i}p_iz^{\nu_i}$, $\mu_k=P_k(1)$, $Q_k=P_k/\mu_k$ and $t_k=\sigma_k^2\mu_k/(1+\sigma_k^2\mu_k)$. Then
$1-\sigma_k^2(P_k-\mu_k)=(1+\sigma_k^2\mu_k)(1-t_kQ_k)$ and $-\ln(1-t_kQ_k)=\sum_{j\ge1}b_{kj}z^j$, so $\ln G(z)=\sum_ja_jz^j$ with

$$a_0=-\sum_iw_{0,i}p_i-\sum_k\frac{\ln(1+\sigma_k^2\mu_k)}{\sigma_k^2},\qquad
a_j=\sum_{i:\nu_i=j}w_{0,i}p_i+\sum_k\frac{b_{kj}}{\sigma_k^2},\qquad
b_j=t\,q_j+\frac tj\sum_{l<j}(j-l)\,b_{j-l}\,q_l .$$

The last relation comes from the derivative of the series. $G'=(\ln G)'G$ gives

$$p_0=e^{a_0},\qquad n\,p_n=\sum_{j=1}^{n}j\,a_j\,p_{n-j}.$$

Every term is nonnegative: no subtraction of terms of opposite sign, so no probability is negative whatever the grid length (the
series expansions of the log and the exp of Haaf, Reiss and Schoenmakers). The result is the `(n_max + 1,)` array of $P(X=0),\dots,P(X=n_{max})$;
the tail mass $1-\sum$ is left out, never added to the last entry. The mean is $\sum_i\nu_ip_i$ whatever the sectors (mean-one factors).
If $e^{a_0}$ underflows (below $10^{-300}$, a portfolio with thousands of expected defaults) the recursion cannot start and the function raises
with the reason (use a larger loss unit or split the portfolio).

## The Panjer comparison (`pmf_panjer`)

The recursion advocated in the original document clears the denominators $D_k=1+\sigma_k^2\mu_k-\sigma_k^2P_k$ and subtracts terms of opposite sign:
$(n+1)D_0\,p_{n+1}=\sum_jN_jp_{n-j}-\sum_{j\ge1}D_j(n+1-j)p_{n+1-j}$ with $D=\prod_kD_k$ and $N=A_0'D+\sum_kP_k'\prod_{l\ne k}D_l$. It is kept only for
comparison (dossier 04 H.4: numerically unstable, citing Gordy 2002 and Giese 2003). Its entries are returned as computed, never
clipped. On the portfolios of the tests (300 obligors, 40 exposure bands, two sectors, 3000 units) it agrees with the stable
result to $2	imes10^{-18}$ absolute and to a relative $3	imes10^{-14}$ wherever the stable probability exceeds $10^{-250}$; on four other
random portfolios tried (up to 2000 obligors, 100 exposure bands, three sectors, 10,000 units) the relative difference reached $10^{-11}$ and no
negative entry was found, so the instability reported in the literature is not reproduced here and the stable recursion remains the only result.

## Goldens

- K = 0, $\nu=(1,2,3)$, $p=(0.1,0.2,0.3)$, $w_0=1$: $P(X=0..7)$ = 0.548812, 0.054881, 0.112506, 0.175711, 0.027992, 0.034868, 0.028804, 0.006003, mean 1.4;
  equal to the direct convolution of the Poisson laws to $10^{-12}$.
- One obligor, $\nu=1$, $p=0.2$, one sector with $\sigma^2=0.5$: the negative binomial 0.826446, 0.150263, 0.020490, 0.002484.
- `measures_from_pmf`: with $F=\mathrm{cumsum}(\mathrm{pmf})$ and $k=\min\{n:F(n)\ge q-10^{-12}\}$, $\mathrm{VaR}=k\,u$ and
  $\mathrm{ES}=\frac{u}{1-q}\big[\sum_{n>k}n\,\mathrm{pmf}(n)+(F(k)-q)k\big]$; pmf (0.5, 0.3, 0.2), $u=1$: VaR 1.0 and ES 2.0 at 0.8, ES 1.666667 at 0.7, EL 0.7.
  It raises when the grid holds less mass than the level.

## Limits

Default mode only, deterministic recoveries in the basic model, and the Poisson approximation can give several defaults of one
obligor for a large PD (dossier 04 H.4). The distribution is on the integer loss grid: the rounding of exposures to multiples of the
loss unit is the caller's choice and is stated in the document.
