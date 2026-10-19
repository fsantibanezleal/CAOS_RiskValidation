# 05. Model risk measures

Module `riskvalidation.aggregation.model_risk`. Sources: dossier 06 B.12 (model risk aggregation); Barrieu, P. and Scandolo, G.
(2015), "Assessing financial model risk", European Journal of Operational Research 242(2):546-556, doi:10.1016/j.ejor.2014.10.032
(arXiv 1307.0684); Glasserman, P. and Xu, X. (2014), "Robust risk measurement and model risk", Quantitative Finance 14(1):29-58,
doi:10.1080/14697688.2013.822989. These are sensitivity information, never a regulatory capital figure (dossier 06 l.699-704): the
ECB requires a quantitative or qualitative assessment of model risk without prescribing a formula.

## Barrieu and Scandolo

For a risk measure $\rho$, a reference model $X_0$ and a set of models with maximum $\overline\rho$ and minimum $\underline\rho$:

$$\mathrm{AM}=\frac{\overline\rho}{\rho(X_0)}-1,\qquad \mathrm{RM}=\frac{\overline\rho-\rho(X_0)}{\overline\rho-\underline\rho}\in[0,1]$$

AM "generalizes the Basel multiplier": multiplying $\rho(X_0)$ by $\mathrm{AM}+1$ reaches the worst case. `barrieu_scandolo(rhos,
reference_index)` takes the risk measure under each model of the set (the band rows) and the position of the reference model. The
maximum is taken over a set that contains the reference, so $\mathrm{AM}\ge0$ and $0\le\mathrm{RM}\le1$. $\rho(X_0)>0$ is required.
RM is 0 when $\overline\rho=\underline\rho$, the contract's convention for the undefined $0/0$ (UNVERIFIED in the paper).

Symbols: $\rho$ a law-invariant risk measure (here the capital of a band row, money), $X_0$ the reference model.

Golden: `rhos = (100, 130, 90)`, reference 0: $\mathrm{AM}=0.3$ and $\mathrm{RM}=(130-100)/(130-90)=0.75$ (computed).

## Glasserman and Xu

Given a baseline law and a loss function $V$, the alternatives are likelihood ratios $m\ge0$, $E[m]=1$, inside a relative-entropy budget
$\eta$:

$$\sup_{m:\,E[m\log m]\le\eta}E[m\,V],\qquad m_\theta=\frac{e^{\theta V}}{E[e^{\theta V}]},\qquad \theta>0\text{ with }E[m_\theta\log m_\theta]=\eta$$

The worst case is computed by reweighting the baseline sample, so it costs little beyond the baseline simulation.
`glasserman_xu_worst_case(v, budget, tol)` finds $\theta$ by bisection on the standardised sample (the shift and scale of $V$ do not move
the tilt, only $\theta$'s units), so that the entropy equals the budget to `tol` (default $10^{-10}$), and returns `GlassermanXu` with
`theta`, `relative_entropy`, `baseline` $=E[V]$ and `worst_case` $=E[m_\theta V]\ge E[V]$. The worst case is increasing in the budget.

Symbols: $V$ the loss function evaluated on the baseline sample (S,), $\eta$ the budget, $\theta$ the tilt, $m$ the likelihood ratio.

Golden: $V=(0,1)$ equally likely, $\theta=1$: $m=(0.5379,1.4621)$, relative entropy 0.1109, worst case 0.7311 against the baseline 0.5; calling
with `budget = 0.1109` recovers $\theta=1$ to $10^{-3}$ (computed; the tests recompute the weights, the entropy and the mean by hand).

## Limits

- The entropy of the tilt rises to $\log(S/k)$ with $k$ the number of sample points at the maximum, at most $\log S$: a budget at or above
  that value is refused (the contract refuses a budget above $\log S$; a sample whose maximum is attained more than once reaches less,
  so the engine refuses it there too).
- A constant $V$ is refused (no tilt changes its mean). The bound is the sup over the likelihood ratios of the sample, so it cannot
  exceed the sample maximum.
- Neither measure is a regulatory capital figure; they are shown beside the band.
