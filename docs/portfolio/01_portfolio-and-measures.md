# 01. The portfolio and its measures

Module `riskvalidation.portfolio.portfolio`. Source: dossier 04 H.1 (loss, risk measures and capital) and C.4 (the large homogeneous
portfolio).

## The loss and the measures

The default-mode loss over the horizon (one year) for obligors $i=1,\dots,N$ with default indicators $D_i$ is

$$L=\sum_{i=1}^{N}\mathrm{EAD}_i\,\mathrm{LGD}_i\,D_i,\qquad \mathrm{EL}=\mathbb E[L]=\sum_i \mathrm{EAD}_i\,\mathbb E[\mathrm{LGD}_i]\,\mathrm{PD}_i,$$

the second form holding when the LGD is independent of the default (`expected_loss`). The expected LGD is deterministic in the
simulations; its variance parameter $\gamma$ ($\mathrm V[\mathrm{LGD}]=\gamma\,\mathbb E[\mathrm{LGD}](1-\mathbb E[\mathrm{LGD}])$) is used only by
the granularity adjustment (page 04).

$$\mathrm{VaR}_q(L)=\inf\{\ell:\Pr(L\le\ell)\ge q\},\qquad
\mathrm{ES}_q(L)=\frac{1}{1-q}\int_q^1\mathrm{VaR}_u(L)\,du,\qquad \mathrm{EC}_q=\mathrm{VaR}_q(L)-\mathrm{EL}.$$

The expected shortfall equals $\mathbb E[L\mid L\ge \mathrm{VaR}_q]$ for a continuous loss (Acerbi and Tasche 2002,
doi:10.1016/S0378-4266(02)00283-2); the unexpected-loss capital $\mathrm{EC}_q$ is the UL concept of the IRB formula (BCBS 2005 explanatory
note, section 4.4, https://www.bis.org/publications/explanatory-note-basel-ii-irb-risk-weight-functions.pdf).

## The portfolio record

`Portfolio` is a frozen record: `ead`, `pd`, `lgd` (N,), `loadings` (N, F), `factor_corr` (F, F), `group` (N,) indexes into
`group_ids`, `lgd_gamma`, `currency` (ISO 4217, three capital letters) and `label` (`"illustrative"` for a constructed portfolio).
The arrays are copied and read-only. It raises `ValueError` for a shape mismatch, an EAD of 0 or less, a PD outside (0, 1), an LGD
outside [0, 1], a loading row with a sum of squares of 1 or more, a factor correlation that is not a symmetric positive definite
correlation matrix, an index out of range, an empty group id and a bad currency. A name's asset return is
$A_i=\beta_i^\top Z+\sqrt{1-\beta_i^\top\Sigma\beta_i}\,e_i$ with $Z\sim N(0,\Sigma)$ (page 02); the engine adds the check
$\beta_i^\top\Sigma\beta_i<1$, which equals the contract's $\beta_i^\top\beta_i<1$ for one factor and for uncorrelated factors.

`portfolio_from_groups` expands groups of identical names (keys `id`, `n_names`, `ead_each`, `pd`, `lgd`, `loading`, `sector`) in group
order. With `one_factor` every name loads on the single factor; otherwise there is one factor per distinct sector in order of first
appearance and the correlation between sectors is an input (`factor_corr`), never a default.

## Sample estimators

For a sample of $S$ losses sorted ascending $L_{(1)}\le\dots\le L_{(S)}$ and $k=\lceil qS-10^{-9}\rceil$ (the guard handles
$0.9\times10=9.000000000000002$):

$$\widehat{\mathrm{VaR}}=L_{(k)},\qquad
\widehat{\mathrm{ES}}=\frac{1}{1-q}\left[\sum_{i>k}\frac{L_{(i)}}{S}+\left(\frac kS-q\right)L_{(k)}\right].$$

The boundary scenario carries the fractional mass, so the tail has exactly mass $1-q$ (Acerbi and Tasche weights); the estimators are
the contract's definition, the dossier gives the integral. `measures_from_sample` adds $\mathrm{EC}=\mathrm{VaR}-\mathrm{EL}$ with the
supplied EL and the batch-means standard errors of page 02. A sample too short for the batches (fewer than 10 losses per batch)
returns both standard errors as `None`; a sample whose VaR is below the EL raises rather than return a negative capital.

## The large homogeneous portfolio (rung C1)

With $p(z)=\Phi\big((G(p)-\sqrt\rho z)/\sqrt{1-\rho}\big)$ decreasing in $z$, $\mathrm{VaR}_q=\mathrm{EAD}\cdot\mathrm{LGD}\cdot\Phi\big((G(p)+\sqrt\rho\,G(q))/\sqrt{1-\rho}\big)$
(`engines.vasicek.lhp_quantile`, never recomputed here) and

$$\mathrm{ES}_q=\mathrm{EAD}\,\mathrm{LGD}\;\mathbb E\big[p(Z)\mid Z\le G(1-q)\big]=\frac{\mathrm{EAD}\,\mathrm{LGD}}{1-q}\int_{-\infty}^{G(1-q)}p(z)\varphi(z)\,dz,$$

computed by Simpson's rule on $[G(1-q)-12,\,G(1-q)]$ with 4000 steps (`lhp_measures`).

Goldens (EAD = 1, LGD = 1, PD = 0.02, $\rho$ = 0.15): at 99 percent VaR 0.105587 and ES 0.135971; at 99.9 percent VaR 0.176329,
ES 0.209404, EL 0.02 and EC 0.156329. With an LGD of 0.45 the EC at 99.9 percent is $0.45\,(0.176329-0.02)=0.070348$, the K of
`regulatory.irb.capital_requirement("residential_mortgage", 0.02, 0.45)`. Sample goldens on the losses 0 to 9: VaR 8.0, 9.0 and 8.0 at
0.9, 0.95 and 0.85; ES 9.0, 9.0 and 8.666667. The fixture portfolio (10 names of 4,000,000, 50 of 600,000 and 1000 of 30,000 with PDs
0.0015, 0.0030, 0.0050 and LGD 0.45) has N = 1060, a total EAD of 100,000,000 and EL 135,000.

## Limits

Default mode only: no migration loss, no random LGD in the simulations. The sample estimators are one choice among several; the
Monte Carlo VaR of a finite portfolio sits above the large-portfolio value by the granularity effect (page 04). The engine states no
compliance claim and the portfolios in the product are labelled illustrative.
