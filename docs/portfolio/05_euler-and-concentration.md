# 05. Euler contributions and concentration

Module `riskvalidation.portfolio.euler`. Source: dossier 04 H.2 and H.6; dossier 05 G.4.

## Euler contributions

Tasche (2008), "Capital allocation to business units and sub-portfolios: the Euler principle", arXiv 0708.2542. For a positively
homogeneous, differentiable risk measure $\rho$ and $L(u)=\sum_iu_iL_i$:

$$\rho(L)=\sum_iu_i\,\frac{\partial\rho(L(u))}{\partial u_i}\Big|_{u=1},\qquad
\mathrm{RC}_i^{\mathrm{VaR}}=\mathbb E\big[L_i\mid L=\mathrm{VaR}_q(L)\big],\qquad
\mathrm{RC}_i^{\mathrm{ES}}=\mathbb E\big[L_i\mid L\ge\mathrm{VaR}_q(L)\big].$$

The contributions add up to the total (full allocation). The equation numbers of Tasche (2008) were not checked in dossier 04 H.2
(UNVERIFIED numbering); dossier 05 G.4 verified the statements from the arXiv abstract page and states that conditioning on a single
point is very noisy in simulation, so ES contributions or kernel-smoothed VaR contributions are preferred.

**ES (`es_contributions`).** The Acerbi-Tasche weights of page 01 give $\mathrm{RC}_g=\frac{1}{1-q}\sum_sw_sL_{g,s}$ with $w_s=1/S$ above the
boundary scenario, $k/S-q$ at it and 0 below; because the group losses add up to the loss in every scenario, $\sum_g\mathrm{RC}_g=\widehat{\mathrm{ES}}$ exactly
(to rounding), ties at the boundary share its fractional mass in input order. It raises when the groups do not add up to the losses.

**VaR (`var_contributions_kernel`).** Tasche (2009), "Capital allocation for credit portfolios with kernel estimators", Quantitative Finance
9(5):581-595, doi:10.1080/14697680802620599: the Nadaraya-Watson estimator with a Gaussian kernel at $L=\mathrm{VaR}_q$,

$$\widehat{\mathrm{RC}}_g=\frac{\sum_sK\big((L_s-\widehat{\mathrm{VaR}})/h\big)L_{g,s}}{\sum_sK\big((L_s-\widehat{\mathrm{VaR}})/h\big)},\qquad h=1.06\,\mathrm{sd}(L)\,S^{-1/5}.$$

The bandwidth is Silverman's rule: the dossier names the paper, not the bandwidth, so the choice is the contract's (UNVERIFIED against the
paper). The estimator is biased at the boundary: its sum agrees with the sample VaR within a tolerance the caller states (the product
uses 0.01 relative at $10^6$ scenarios), not exactly. Contributions are nonnegative for nonnegative group losses and do not depend on the
order of the scenarios; with constant shares $L_g=a_gL$ the estimate tends to $a_g\mathrm{VaR}$.

**The asymptotic anchor (`asrf_var_contributions`).** The loss is monotone in the factor, so the loss quantile is the conditional mean at
$z=G(1-q)$ (dossier 04 C.4):

$$\mathrm{RC}_i=\mathrm{EAD}_i\,\mathrm{LGD}_i\,p_i\big(G(1-q)\big),\qquad p_i(z)=\Phi\!\Big(\frac{G(\mathrm{PD}_i)-\sqrt{\rho_i}\,z}{\sqrt{1-\rho_i}}\Big),$$

from `engines.vasicek.conditional_pd`. It is the first term of Emmer and Tasche (2005), "Calculating credit risk capital charges with the
one-factor model", Journal of Risk 7(2):85-101, doi:10.21314/jor.2005.110 (arXiv cond-mat/0302402, eq. 2.17 times $u_j$, the asymptotic
portfolio-invariant part), as read on the paper by the review of the contract; it is **not** their full contribution, which adds
granularity terms. The contributions sum to the infinitely granular VaR, which for identical names equals
$\mathrm{EAD}\,\mathrm{LGD}\,\mathtt{lhp\_quantile}(q,\mathrm{PD},\rho)$; the sample VaR of a finite portfolio sits near it, not on it.

Goldens: `es_contributions` on 10 scenarios with group losses $0..9$ and $0,2,..,18$ at $q=0.9$: contributions (9.0, 18.0), sum 27.0 =
the sample ES. `asrf_var_contributions` for two names, EAD (60, 40), LGD 0.45, PD (0.01, 0.05), $\rho=0.15$, $q=0.999$ ($G(0.001)=-3.090232$):
conditional PDs (0.110265, 0.313506), contributions (2.977148, 5.643106), sum 8.620255.

## Concentration

$\mathrm{HHI}=\sum_is_i^2$ with $s_i=\mathrm{EAD}_i/\sum_j\mathrm{EAD}_j$ (a definition, dossier 04 H.6); the effective number of names is $1/\mathrm{HHI}$, between 1 and $N$,
invariant to the scale of the EAD. Goldens: EAD (50, 30, 20) give HHI 0.380000 and 2.631579 effective names; the fixture portfolio
of page 01 has HHI 0.01789. The granularity adjustment (page 04) is driven by an $s_i^2$-weighted sum, so it is a model-based refinement of HHI.
Sector concentration (Dullmann and Masschelein 2007, doi:10.1007/s10693-007-0014-3; Pykhtin 2004, details UNVERIFIED in dossier 04 H.6) is not implemented.

## Limits

The kernel VaR contributions need many scenarios and a stated tolerance; the ES contributions need a tail of enough scenarios
(at 99.9 percent and $10^5$ scenarios the tail holds 100). The anchor is exact only for the infinitely granular portfolio.
Contributions are for the default-mode loss of a portfolio, in the portfolio currency.
