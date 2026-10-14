# 04. Granularity adjustment and importance sampling

Modules `riskvalidation.portfolio.granularity` and `riskvalidation.portfolio.importance`. Source: dossier 04 H.5 (granularity) and H.7
(rare-event simulation); ladder rungs C5 and C6.

## The granularity adjustment (rung C5)

Gordy, M.B. and Lutkebohmert, E. (2013), "Granularity adjustment for regulatory capital assessment", International Journal of Central
Banking 9(3):33-48, https://www.ijcb.org/sites/default/files/journal/v9n3/ijcb-v9n3-granularity-adjustment-regulatory-capital-assessment.pdf.
The portfolio loss rate is $L_n=\sum_is_iU_i$ (a loss per unit of EAD, $s_i$ the exposure shares, eq. 1). The general result (eq. 9, p. 41) is

$$\mathrm{GA}=\alpha_q(L_n)-\mu\big(\alpha_q(X)\big)\approx-\frac{1}{2h(\alpha_q(X))}\frac{d}{dx}\left[\frac{h(x)\,\sigma^2(x)}{\mu'(x)}\right]_{x=\alpha_q(X)},$$

the gap of the portfolio quantile above the infinitely granular one, **already a share of EAD**. The first draft of the contract read eq. 17 as
a ratio to $K^*$ and multiplied it by $K^*$; the review of that contract refuted this on the paper and the engine returns `ga` unscaled.

The Basel form is built on the single-factor CreditRisk+ model with the systematic factor $X\sim\mathrm{Gamma}$ of mean 1 and variance
$1/\xi$ (shape and rate $\xi$, p. 43), conditional PD $\pi_i(x)=\mathrm{PD}_i(1+\omega_i(x-1))$, reserve $R_i=\mathbb E[\mathrm{LGD}_i]\mathrm{PD}_i$,
$K^*=\sum_is_iK_i$, and (eq. 11, 15, 16, 17)

$$C_i=\frac{\mathrm V[\mathrm{LGD}_i]+\mathbb E[\mathrm{LGD}_i]^2}{\mathbb E[\mathrm{LGD}_i]},\quad
\delta=(\alpha_q-1)\Big(\xi+\frac{1-\xi}{\alpha_q}\Big),\quad \mathrm V[\mathrm{LGD}_i]=\gamma\,\mathbb E[\mathrm{LGD}_i](1-\mathbb E[\mathrm{LGD}_i]),$$
$$\widetilde{\mathrm{GA}}=\frac{1}{2K^*}\sum_{i=1}^{n}s_i^2\,C_i\big(\delta(K_i+R_i)-K_i\big).$$

`gamma_quantile(xi, q)` is $\alpha_q(X)$, `delta`, `lgd_variance` and `granularity_adjustment(share, k, pd, lgd_mean, lgd_gamma, xi, q)`
follow; $K_i$ is an **input**, as in the paper's regulatory algorithm (p. 47, step iv: the UL capital from the IRB formula, with which the
PD and $\omega$ inputs are eliminated, p. 45). The helper `k_creditrisk_plus` computes eq. 13, $K_i=\mathbb E[\mathrm{LGD}_i]\mathrm{PD}_i\omega_i(\alpha_q-1)$,
for a caller that has $\omega_i$ and no IRB capital. The exact adjustment (eq. 14) with second-order terms is not implemented.

Goldens. The paper's test vector: $\xi=0.25$ and $q=0.999$ give $\alpha_q=17.505777$ and $\delta=4.833601$ (the paper states 4.83; the other
reading of $\xi$, variance $\xi$, gives 1.0867). For $n=10$ identical names ($s=0.1$, PD 0.01, $\mathbb E[\mathrm{LGD}]=0.45$, $\gamma=0.25$, $\omega=0.5$,
$K_i=0.037138$): $C=0.5875$, $R=0.0045$ and $\widetilde{\mathrm{GA}}=0.129817$ of EAD; at $n=100$, 0.012982 (computed from eq. 17; the paper's own
numbers for this portfolio are not in the dossier). For identical names $\mathrm{GA}\cdot n$ is constant and the adjustment tends to zero.

## Importance sampling (rung C6)

Glasserman, P. and Li, J. (2005), "Importance sampling for portfolio credit risk", Management Science 51(11):1643-1656,
doi:10.1287/mnsc.1050.0415. Given the factor $z$ the conditional default probabilities are twisted, with $c_i=\mathrm{EAD}_i\mathrm{LGD}_i$,

$$p_{i,\theta}(z)=\frac{p_i(z)\,e^{\theta c_i}}{1+p_i(z)\,(e^{\theta c_i}-1)},$$

with $\theta=\theta(z)$ the root of $\sum_ic_ip_{i,\theta}(z)=x$ (`twisted_pd`, `solve_theta`; $\theta=0$ where the untwisted mean is at or above
$x$), the factor is drawn with mean $\mu$, and each scenario carries the likelihood ratio

$$w=\exp\Big(-\theta L+\sum_i\ln\big(1+p_i(e^{\theta c_i}-1)\big)\Big)\;\exp\big(-\mu^\top z+\tfrac12\mu^\top\mu\big).$$

**UNVERIFIED (dossier 04 H.7).** The transcription of the twist and of the factor mean shift (the mode of the zero-variance density) was not
checked against the paper, which is paywalled. The engine therefore accepts nothing on trust. The likelihood ratio is exact whatever
$\theta(z)\ge0$ and $\mu$ are, so $E_Q[w]=1$ and $E_Q[wL]=\mathrm{EL}$ hold exactly (a test integrates over $z$ and the default patterns of three
names to $10^{-9}$), and the acceptance test is **unbiasedness against plain Monte Carlo**: the weights average 1, the weighted loss
averages the EL, and the weighted tail probability, VaR and ES agree with `simulate_gaussian` within 4 combined standard errors, for
several thresholds and for the default, a zero and a negative shift. The default shift maximises
$-\theta(z)\,x+\ln M(\theta(z),z)-z^\top z/2$ by Nelder-Mead from $z=0$ (deterministic). `weighted_var_es` gives the smallest loss whose
weighted upper tail probability $\sum_iw_i\mathbf 1\{L_i>x\}/S$ is at most $1-q$ and the weighted Acerbi-Tasche ES built on the same tail,
both **normalised by the scenario count $S$ and never by the realised sum of the weights**, with batch-means standard errors, and `variance_ratio` is $(\mathrm{se}_{\text{plain}}/\mathrm{se}_{IS})^2$.

Goldens: $p=0.1$, $c=1$, $\theta=\ln2$ gives $p_\theta=0.181818$; the threshold $2/11$ gives $\theta=\ln 2$ (a threshold rounded to 0.181818 gives 0.693146);
`weighted_var_es` on the losses 0 to 3 with unit weights at $q=0.5$ gives VaR 1.0 and ES 2.5; `variance_ratio(0.002, 0.0005)=16.0`. On the 30-name
illustrative portfolio of the tests (PD 0.02, $\rho$ 0.3) the default twist and shift cut the variance of a 99 percent tail probability by a factor above 1.5.

## Limits

The twisted sample is unbiased and its tail estimators agree with plain Monte Carlo at the threshold equal to the plain 99.9 percent VaR of a
60-name portfolio (VaR and ES within their errors; a test), but its weights are heavy tailed for a threshold far in the tail: on a 60-name portfolio (PD 0.02, $\rho$ 0.3) with
$2\times10^5$ scenarios and the default shift, the sample mean of the weights was 0.70 for a threshold of 7.2 and 0.40 for 14.4, because a few
scenarios carry most of the weight (the largest 43 percent of the total at 14.4). The realised weight sum is then far from S (0.034 of it in the review at the 99.9 percent VaR), which is why the quantile and the ES are normalised by S: dividing
by the realised sum gave a VaR of 35 against 21.0 and an ES of 39 against 25.6. The mean-weight identity is not a useful diagnostic
at that sample size; report the variance ratio and the standard errors with the sample, and raise the scenario count. Only the Gaussian
copula with uncorrelated factors is supported: a correlated factor matrix raises, and there is no Student-t twist (the multi-factor twist is a
failure mode of the ladder). The granularity adjustment is the simplified form, second-order terms are missing for very concentrated books,
and it depends on the $K_i$ the caller chooses (the IRB capital at the regulatory correlation in the product).
