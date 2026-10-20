# 04. Reverse stress

EBA/GL/2018/04 defines a reverse stress test as an institution stress test that "starts from the identification of the pre-defined
outcome (e.g. points at which an institution business model becomes unviable, or at which the institution can be considered as
failing or likely to fail in the meaning of Article 32 of Directive 2014/59/EU) and then explores scenarios and circumstances that
might cause this to occur" (dossier 01 l.884-889). `riskvalidation.stress.reverse` solves it as the most plausible scenario that
reaches the failure point.

## The formulation

The formal statement is Contraste's, not regulatory text (dossier 01 l.891-894; dossier 05 H.5): given a loss model $L(x)$ over the
deviations $x$ of the factors from the baseline, with covariance $\Sigma$, find the most plausible scenario reaching the failure point.
For Gaussian (and, as a standard result, elliptical) factors the most plausible scenario on the loss surface $\{x : L(x) = \ell\}$
maximises the factor density, which is the minimum Mahalanobis distance scenario

$$
x^{*} = \arg\min_{x}\; x^\top \Sigma^{-1} x \quad \text{subject to} \quad L(x) \ge \ell,\qquad d^{*} = \sqrt{x^{*\top}\Sigma^{-1}x^{*}}.
$$

The attribution of the elliptical result to a specific paper is UNVERIFIED in the dossier (05 l.412). In C18 the failure point is CET1 at the
Chilean legal minimum stack (decision Q11, `capital.chilean_cet1_stack`) and the factors are the standardised satellite regressors with
their covariance estimated on 1985 to 2019.

## The closed form for a linear loss

For $L(x) = g^\top x$ the Lagrangian gives

$$
x^{*} = \frac{\Sigma g\,\ell}{g^\top \Sigma g},\qquad d^{*} = \frac{\ell}{\sqrt{g^\top \Sigma g}},\qquad g^\top x^{*} = \ell.
$$

`reverse_stress_linear` returns it with `z_star = x_star / sd` (the factor standard deviations, the square roots of the diagonal of
$\Sigma$). It refuses a covariance that is not symmetric positive definite, a gradient with $g^\top \Sigma g = 0$ and a negative threshold.
Golden value (computed): $\Sigma = \mathrm{diag}(1, 4)$, $g = (1, 1)$, $\ell = 5$ give $g^\top \Sigma g = 5$, $x^{*} = (1, 4)$, $d^{*} = 5/\sqrt{5} = 2.2361$ and loss
$g^\top x^{*} = 5$. No point with $L(x) \ge \ell$ is closer: the test draws 2000 random feasible points and checks it.

## SLSQP for a general loss

`reverse_stress(loss_fn, cov, threshold, starts, tolerance)` minimises $x^\top \Sigma^{-1} x$ subject to $L(x)/\ell - 1 \ge 0$ with
`scipy.optimize.minimize(method="SLSQP")` from each start, with the analytic objective gradient, `ftol` of $10^{-14}$ and at most 1000
iterations. A start counts as converged when SLSQP reports success and $|L(x) - \ell| / \ell \le$ `tolerance`; the function returns the best
converged solution (the smallest distance) and the per-start results (the unconverged ones carry `converged = False`). It requires at
least three starts (the search is local, so a single start proves nothing), a positive threshold and a positive tolerance, and raises
`ValueError` when no start converges. Golden values: for the linear loss above the best solution equals the closed form to $10^{-6}$
from every start; for the quadratic loss $L(x) = x_1^2 + x_2^2$ with $\Sigma = I$ and $\ell = 4$ the distance is $2.0$ and the solution lies on
the circle $|x| = 2$.

## The plausibility

$$
p(d, K) = P\big(\chi^2_{K} \ge d^2\big),
$$

the tail probability of a scenario at least as far from the baseline as the solution, with $K$ the number of factors (`plausibility`, 1 at
distance 0 and decreasing). Golden values (computed): $d = \sqrt{5} = 2.2361$ gives $0.0821$ ($\exp(-2.5)$) for $K = 2$ and $0.1718$ for $K = 3$.

## Limits

- The solution is the most plausible scenario under a Gaussian or elliptical factor distribution with the covariance given; real macro
  factors have fat tails and regime shifts, so the plausibility is a distance-based indicator, not a probability of the event.
- SLSQP is a local method with finite-difference constraint gradients; a loss that is not smooth or has several local minima needs many
  well-spread starts. The solution check `validation.stress.reverse_solution_check` tests the loss at the solution and the other starts.
- Empirical likelihood (Glasserman, Kang and Kang 2015) is out: only its abstract is in the dossier (05 l.412, l.545); its detail is
  UNVERIFIED.
- The engine states no compliance claim: a reverse stress scenario is a diagnostic of where the bank breaks, not a regulatory scenario.

## Sources

- EBA/GL/2018/04, Guidelines on institutions' stress testing (the reverse stress test definition), dossier 01 J.4.
- Glasserman, P., Kang, C. and Kang, W. (2015), Stress scenario selection by empirical likelihood, Quantitative Finance 15(1):25-41,
  doi:10.1080/14697688.2014.926019 (abstract only; the Mahalanobis form is a standard result, attribution UNVERIFIED).
- Dossier 05 H.5 and dossier 01 J.4 of the Contraste initiative research.
