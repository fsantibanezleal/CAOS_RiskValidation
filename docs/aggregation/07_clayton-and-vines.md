# 07. Clayton sampling and vine copulas

Modules `riskvalidation.aggregation.copulas` (the Clayton additions) and `riskvalidation.aggregation.vines` (optional extra
`vine`). Requirements `RV-1901` to `RV-1910` in `docs/design/features/u19-ladder-rungs/requirements.md`, each naming its test.
These two rungs were deferred in 1.0 (BL-078); the closed forms of the bivariate Clayton copula (`clayton_copula`,
`clayton_lower_tail_dependence`) are on [page 02](02_copulas.md).

Sources, as read for this page (the symbols of the PDFs were re-derived where the text extraction lost them):

- Hofert, "Sampling Archimedean copulas", Ulm preprint series 2007-10 (version 2008-05-16),
  https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi/forschung/PreprintServer/2007/preprintmariushofert.pdf: Theorems 2.2
  and 2.3, Algorithm 1 and Table 1. The journal version (Computational Statistics and Data Analysis, 2008) was not checked
  (UNVERIFIED).
- McNeil and Neslehova, "Multivariate Archimedean copulas, d-monotone functions and l1-norm symmetric distributions", Annals
  of Statistics 37(5B):3059-3097, doi:10.1214/07-AOS556, arXiv:0908.3750, Example 2.3.
- Marshall and Olkin, "Families of multivariate distributions", Journal of the American Statistical Association 83:834-841
  (1988): taken from Hofert's statement of the algorithm, not read (UNVERIFIED against the original).
- Aas, Czado, Frigessi and Bakken, "Pair-copula constructions of multiple dependence", Insurance: Mathematics and Economics
  44(2):182-198, 2009, doi:10.1016/j.insmatheco.2007.02.001; equation numbers below are those of the preprint
  https://mediatum.ub.tum.de/doc/1083600/1083600.pdf, which was read.
- Dissmann, Brechmann, Czado and Kurowicka, "Selecting and estimating regular vine copulae and application to financial
  returns", Computational Statistics and Data Analysis 59:52-69, 2013, doi:10.1016/j.csda.2012.08.010; equation numbers are
  those of arXiv:1202.2002v2, which was read.
- Bedford and Cooke, "Vines: a new graphical model for dependent random variables", Annals of Statistics 30(4):1031-1068
  (2002): cited by the two papers above, not read (UNVERIFIED as a primary text).
- Joe, *Dependence Modeling with Copulas*, 2014, p. 168 (Kendall's tau of the Clayton copula): read only through the citation in
  the statsmodels source (`ClaytonCopula.tau`); the general Archimedean integral of Genest and MacKay (1986) and Nelsen (2006,
  Corollary 5.1.4) was not read as a citation and is used here only as a numerical check.
- pyvinecopulib 1.0.1 (Nagler and Vatter, MIT licence, PyPI 2026-10-06, Python 3.11 or later): its documentation at
  https://pyvinecopulib.readthedocs.io/ and its PyPI metadata.

## The Clayton copula

The Clayton copula has the generator $\psi(x)=(1+x)^{-1/\theta}$, completely monotone for $\theta>0$, so that
$C(u_1,\dots,u_d)=\psi\big(\psi^{-1}(u_1)+\dots+\psi^{-1}(u_d)\big)=\big(\sum_k u_k^{-\theta}-d+1\big)^{-1/\theta}$ is a copula in every dimension
$d\ge 2$ (Kimberling's theorem, Hofert Theorem 2.2; McNeil and Neslehova Example 2.3). It is exchangeable with one parameter for
the whole portfolio. It is lower tail dependent with $\lambda_L=2^{-1/\theta}$ and has no upper tail dependence.

Kendall's tau and its inverse (`clayton_kendall_tau`, `clayton_theta_from_tau`, RV-1901):

$$\tau=\frac{\theta}{\theta+2},\qquad \theta=\frac{2\tau}{1-\tau},$$

with the goldens $\tau=1/3,\ 1/2,\ 2/3$ at $\theta=1,\ 2,\ 4$. The refusals name the parameter: $\theta>0$ and $\tau\in(0,1)$.

### Sampling (`simulate_clayton_uniforms`, RV-1902)

Marshall and Olkin's frailty algorithm, for any $d\ge 2$ (Hofert Algorithm 1 with the Clayton row of Table 1: $\psi(t)=(1+t)^{-1/\theta}$,
$F=\mathrm{Gamma}(1/\theta,1)$, shape $1/\theta$ and scale 1):

$$V\sim\mathrm{Gamma}(1/\theta,1),\quad X_1,\dots,X_d\sim U(0,1)\ \text{independent},\quad U_k=\Big(1-\frac{\ln X_k}{V}\Big)^{-1/\theta}.$$

Only $d+1$ random numbers are used per row. For $\theta$ large the shape $1/\theta$ is tiny and $V$ underflows to 0 in a few per
cent of the draws at $\theta=200$; the engine therefore draws $V$ in the log scale, as $\mathrm{Gamma}(1/\theta+1)\,W^{\theta}$ with
$W\sim U(0,1)$ when the shape is below 1 (Stuart's identity, the same distribution), and evaluates
$U_k=\exp\{-\theta^{-1}\,\mathrm{logaddexp}(0,\ln(-\ln X_k)-\ln V)\}$.

For $d=2$ only, the conditional inverse (`method="conditional"`; Aas et al. Appendix B.3 and equations (6) and (7); Dissmann et al.
Section 2.4): draw $U_2$ and $W$ uniform and set $U_1=h^{-1}(W\mid U_2)$ with the h-function and its closed-form inverse

$$h(u_1\mid u_2)=\frac{\partial C}{\partial u_2}=u_2^{-\theta-1}\big(u_1^{-\theta}+u_2^{-\theta}-1\big)^{-1-1/\theta},$$

$$h^{-1}(w\mid u_2)=\Big[\big(w\,u_2^{\theta+1}\big)^{-\theta/(\theta+1)}+1-u_2^{-\theta}\Big]^{-1/\theta}$$

(`clayton_h_function`, `clayton_h_inverse`, RV-1904). The printed formulas of the preprint lost their exponents in the text
extraction; they were reconstructed, differentiated by hand and confirmed numerically.

Conventions decided:

- **Survival by default.** The Clayton copula puts its dependence in the joint small values; a loss aggregate needs it in the joint
  large losses. `survival=True`, the default, returns $1-U$ (dossier 05 l.354: "apply to survival or to losses as needed");
  `survival=False` returns the Clayton copula itself. Kendall's tau is the same for both.
- Every draw is clipped to $[\mathrm{UNIFORM\_FLOOR},\,1-\mathrm{UNIFORM\_FLOOR}]$, strictly inside $(0,1)$, as for the Gaussian
  and Student t draws.
- Randomness is `numpy.random.Generator(PCG64(seed))`. Unlike the Gaussian and t path, the rows depend on $(\theta,d,n,\text{seed},\text{method})$
  only and never on the chunk size: the draw is made in fixed blocks of `CLAYTON_BLOCK` = 16,384 rows from one generator, and
  `chunk_size` of `simulate_clayton_losses` only groups blocks for the quantile functions (the working memory).
- A separate `clayton_aggregate` rather than a `family="clayton"` branch of `copula_aggregate`: the Clayton copula has one parameter
  and no correlation matrix, so the `(corr, nu)` signature does not fit.

### The aggregate (`clayton_aggregate`, RV-1905)

$L_k=F_k^{-1}(U_k)$ with $U$ the Clayton draw and $L=\sum_k L_k$; the VaR, the ES, $EC=\mathrm{VaR}-EL$ and the batch-means errors
are those of the reused `measures_from_sample`, as for `copula_aggregate`. The result is a `ClaytonAggregate` (theta, method,
survival, n, seed, level, el, var, es, ec, var_se, es_se). As $\theta$ grows the aggregate approaches the comonotonic sum
(`bounds.comonotonic_var`: $\theta=200$ is within Monte Carlo error of it); as $\theta$ falls it approaches the independent aggregate
($\theta=0.01$ against the identity-correlation Gaussian aggregate).

### Evidence

Every golden is recomputed by a second route in `tests/aggregation/test_clayton_sampling.py`, never typed:

| Check | Route |
|---|---|
| tau | closed form; the integral $1+4\int_0^1\phi/\phi'$ with $\phi(t)=(t^{-\theta}-1)/\theta$; the empirical Kendall tau of 400,000 pairs within 0.003 at $\theta=1,2,4$ |
| copula | empirical $C(0.5,0.5)$ within 0.003 of $(2\cdot 0.5^{-\theta}-1)^{-1/\theta}$; the trivariate $C(0.5,0.5,0.5)$ at $\theta=2$ within 0.003 of $0.31623$ |
| the two algorithms | a two-sample Kolmogorov-Smirnov test of $C(U_1,U_2)$ at $d=2$ does not reject at 1% |
| h-function | $h$ against a central finite difference of $C$; the round trip of $h^{-1}$ (limit below) |
| statsmodels | `ClaytonCopula` `tau`, `theta_from_tau`, `cdf`, and the Kendall tau of its draws, where the `stress` extra is installed |

Limit stated rather than hidden (RV-1904): $h(u_1\mid u_2)$ is flat in $u_1$ where the copula density is small ($\partial h/\partial u_1$ is
the density $c(u_1,u_2)$, $5\cdot10^{-8}$ at $\theta=4$ in the corner $(0.99,0.01)$), so that the rounding of $w$ (a few units in the last
place) moves $u_1$ by about $64\,\varepsilon\,w/c$. On the grid of 25 by 25 points of $(0.01,0.99)$ the round trip
$h^{-1}(h(u_1\mid u_2)\mid u_2)-u_1$ is below $10^{-12}$ at $\theta=0.5$ and $1$; at $\theta=2$ and $4$ it is $4.7\cdot10^{-12}$ and
$5.7\cdot10^{-8}$ at the worst corner, which is the conditioning of the problem, not the formula. The test asserts $10^{-12}$ plus that
propagated rounding, and the other direction, $h(h^{-1}(w\mid u_2)\mid u_2)=w$, to $10^{-12}$ on the whole grid for every $\theta$.

## Vine copulas

### The pair-copula construction

A $d$-dimensional density factors into $d(d-1)/2$ pair-copula densities (Aas et al. equations (3) and (8) to (10)). For three
variables, equation (10),

$$f=f_1f_2f_3\,c_{12}(F_1,F_2)\,c_{23}(F_2,F_3)\,c_{13|2}\big(F(x_1\mid x_2),F(x_3\mid x_2)\big).$$

The conditional distributions come from the h-function $h(x\mid v,\Theta)=\partial C_{xv}(x,v\mid\Theta)/\partial v$ (equation (7)). A D-vine
(path trees, equation (8)) and a C-vine (star trees, equation (9)) are the two extreme regular vines; a regular vine
(Bedford and Cooke 2002; Dissmann et al. Definition 2.1) is a sequence of trees $T_1,\dots,T_{d-1}$ in which the edges of $T_i$ are
the nodes of $T_{i+1}$ and two nodes are joined only if their edges share a node (the proximity condition).

Log-likelihood, equations (15) to (17): the sum over the pair copulas, tree by tree, of the log-densities evaluated on the
h-function transforms of the data, $\sum_t\log c(\cdot,\cdot\mid\Theta)$. For the three-variable D-vine

$$\ell=\sum_t\Big[\log c_{12}(u_{1t},u_{2t})+\log c_{23}(u_{2t},u_{3t})+\log c_{13|2}\big(h(u_{1t}\mid u_{2t}),h(u_{3t}\mid u_{2t})\big)\Big].$$

Sampling, equation (14) and Algorithms 1 and 2 (Dissmann et al. Algorithm 2.2): $x_1=w_1$, $x_2=F^{-1}_{2|1}(w_2\mid x_1)$,
$x_3=F^{-1}_{3|1,2}(w_3\mid x_1,x_2),\dots$ with independent uniforms $w$, by the inverse h-function recursion. The Rosenblatt transform
(Aas et al. Algorithms 5 and 6) maps a vine's own sample back to independent uniforms. For three Gaussian pair copulas the implied
correlation is $\rho_{13}=\rho_{13|2}\sqrt{(1-\rho_{12}^2)(1-\rho_{23}^2)}+\rho_{12}\rho_{23}$, which is $0.3\cdot0.75+0.25=0.475$ for $0.5$, $0.5$ and $0.3$.

### Selection (`fit_vine`, RV-1907)

Dissmann et al. Algorithm 3.1, as pyvinecopulib implements it: the empirical Kendall's tau of every pair; the maximum spanning tree
of $\sum|\tau_{jk}|$ (Prim); per edge the pair-copula family of smallest AIC with its parameters by maximum likelihood; then the
pseudo-observations of the next tree through the fitted h-functions, the conditional Kendall's tau of the pairs that may be edges
(the proximity condition) and the next spanning tree, up to tree $d-1$ or the truncation level. The default family set is
independence, Gaussian, Student t, Clayton, Gumbel, Frank, Joe, BB1, BB6, BB7 and BB8, each with its rotations (the nonparametric
and extreme-value families of pyvinecopulib's own default are left out). `structure="d_vine"` or `"c_vine"` fixes the variable order
$1,\dots,d$ and fits only the pair copulas. `FitControlsVinecop` is given `seeds=[seed]` and `num_threads=1`, so a fit is reproducible.

The inputs are pseudo-observations strictly inside $(0,1)$ (ranks over $n+1$); the margins stay with the caller, as the quantile
functions of an aggregate do. The result is a frozen `VineFit`: the `Vinecop` object, `loglik`, `aic`, `n_params`, `n_obs`, `dim`
and `description`, a JSON-able dict of the structure kind, the variable order and, per tree and edge, the conditioned pair, the
conditioning set, the family, the rotation, the parameters and the pair's Kendall tau. On data from a known C-vine with a strong
star in tree 1 the fit returns that star.

### Simulation and aggregate (`simulate_vine`, `vine_aggregate`, RV-1908 and RV-1910)

`simulate_vine(fit, n, seed)` is `Vinecop.sample(n, seeds=[seed], num_threads=1)`. `vine_aggregate(ppfs, fit, el, level, n_draws, seed)`
applies one quantile function per variable and returns a `VineAggregate` through `measures_from_sample` (the draw is one array of
$n\,d\cdot 8$ bytes); it refuses a number of quantile functions that differs from the vine's dimension. `vine_fit_from_model(model, u=None)`
wraps a vine given directly (known parameters) in a `VineFit`.

### Pair-copula building blocks and the reconciliation (RV-1909)

`gaussian_h_function`, `gaussian_log_density` and `clayton_log_density` (with `copulas.clayton_h_function`) are the pair copulas in
NumPy and SciPy alone: the Gaussian h-function $\Phi\big((z_1-\rho z_2)/\sqrt{1-\rho^2}\big)$ and log-density
$-\tfrac12\ln(1-\rho^2)-\{\rho^2(z_1^2+z_2^2)-2\rho z_1z_2\}/\{2(1-\rho^2)\}$, and the Clayton log-density
$\ln(1+\theta)-(1+\theta)\ln(u_1u_2)-(2+1/\theta)\ln(u_1^{-\theta}+u_2^{-\theta}-1)$ (Aas et al. Appendices B.1 and B.3). The test sums
them as in the three-variable $\ell$ above on a fixed dataset and compares the total with `Vinecop.loglik` to $10^{-9}$, for a
Clayton-Gaussian-Clayton vine, an all-Gaussian vine and one with a rotated (survival) Clayton pair.

## Limits

- `pip install riskvalidation[vine]` installs `pyvinecopulib>=1.0.1,<2` for Python 3.11 or later only (its wheels need 3.11; it pulls
  numpy 2 or later, matplotlib, networkx and pydot). The module imports it inside the functions, so the engine core stays NumPy and
  SciPy only (and importable under Pyodide); a call without the package raises `ImportError` naming the extra. The test of the
  installed version holds the version the reconciliation was measured on (1.0.1); the tests use `pytest.importorskip`, so the Python
  3.10 job skips them.
- A vine is a model of the dependence of pseudo-observations: the number of parameters grows with $d^2$ unless truncated
  (`trunc_lvl`), and the selection is a greedy heuristic, not a global optimum (Dissmann et al. state this). A vine fitted to a short sample of
  a high-dimensional portfolio is over-parameterised; the caller chooses the truncation and the family set.
- Not read, so UNVERIFIED: the C++ vinecopulib documentation of the selection heuristics and the argument lists that the pyvinecopulib
  documentation truncates.
