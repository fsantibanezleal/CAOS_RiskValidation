# 04. CAViaR, its ES extension, the dynamic quantile statistic and the GAS-FZ models

Modules `riskvalidation.market.caviar` and `riskvalidation.market.gas_fz`. Sources: dossier 05 A.1 and A.2 rows 11 and 12.
Return convention throughout; `f_t` is the VaR of day `t`.

## CAViaR, rung 8

$$\text{sav: } f_t=b_1+b_2f_{t-1}+b_3|y_{t-1}|\qquad
\text{as: } f_t=b_1+b_2f_{t-1}+b_3(y_{t-1})^++b_4(y_{t-1})^-$$

with $(y)^+=\max(y,0)$ and $(y)^-=-\min(y,0)$ (the convention is equivalent to the indicator form
$(\gamma_+\mathbb 1\{y>0\}+\gamma_-\mathbb 1\{y\le0\})|y|$ restated in arXiv 2603.02357 eq. 4; the Engle and Manganelli PDF,
https://www.nber.org/papers/w7341.pdf, is a scan without a text layer, so the dossier's open item is closed by that equivalence
and not by the original text). Source: Engle and Manganelli, JBES 22(4):367-381, 2004, doi:10.1198/073500104000000370.

`fit_caviar(y, alpha, spec, n_starts, seed)` minimises the tick loss $\sum_t(\alpha-\mathbb 1\{y_t<f_t\})(y_t-f_t)$ over
`n_starts >= 2` starts (the objective is non-convex: A.2 rung 8). The recursion starts at the empirical `alpha`-order statistic
of the estimation sample (the `var` of `conventions.var_es_from_sample`); the first start is the constant benchmark
(`b = (q, 0, 0)`), the others random with the stationary level at the empirical quantile, each polished by two Nelder-Mead runs;
the recursion is a linear filter run with `scipy.signal.lfilter`. The reported `tick_loss` is the best found and does not exceed
the tick loss of any constant, since the first start is that constant. Deterministic for the same seed; `N >= 500`.
`caviar_var_forecast(fit, y)` runs the recursion through `y` and returns `f_{T+1}`, reading nothing beyond `y[-1]`.

`caviar_es_forecast(fit, y)` is the autoregressive ES-CAViaR of Taylor (2019, JBES 37(1):121-133,
doi:10.1080/07350015.2017.1281815):

$$\mathrm{ES}_t=\mathrm{VaR}_t-x_t,\qquad x_t=\begin{cases}\gamma_0+\gamma_1(\mathrm{VaR}_{t-1}-r_{t-1})+\gamma_2x_{t-1}&r_{t-1}\le\mathrm{VaR}_{t-1}\\ x_{t-1}&\text{otherwise}\end{cases}$$

estimated by maximising the asymmetric Laplace likelihood
$f(r)=\frac{\alpha-1}{\mathrm{ES}_t}\exp\!\left(\frac{(r-\mathrm{VaR}_t)(\alpha-\mathbb 1\{r\le\mathrm{VaR}_t\})}{\alpha\,\mathrm{ES}_t}\right)$ on the VaR
path of the fit, with $\gamma_0,\gamma_1\ge0$ and $0\le\gamma_2<1$ so that $x_t\ge0$ and `es <= var` always; $x_1$ is the sample
gap `VaR - ES` of `y`. **The primary text was not retrieved:** the equations are those restated in Section 3.2 of arXiv 2603.02357
(the paper gives no constraints of its own; the constraints above are this unit's, to keep `es <= var`). No golden value: the
invariants (`es <= var`, determinism, the gap of the order of the empirical one) are the tests.

`dynamic_quantile_statistic(hits, var, alpha, lags=4)` is the dynamic quantile (DQ) test statistic of Engle and Manganelli:

$$DQ=\frac{\mathrm{Hit}'X(X'X)^{+}X'\,\mathrm{Hit}}{\alpha(1-\alpha)},\qquad \mathrm{Hit}_t=\mathbb 1\{y_t<\mathrm{VaR}_t\}-\alpha,\qquad X_t=(1,\mathrm{Hit}_{t-1},\dots,\mathrm{Hit}_{t-4},\mathrm{VaR}_t)$$

asymptotically chi-square; the first `lags` days only supply lags; the degrees of freedom are the rank of `X` (`lags + 2` in
general); invariant to an affine change of the VaR, so the sign convention does not matter. The regressor set is the contract's
(constant, lagged hits, contemporaneous VaR); the degrees of freedom of the paper were not read (UNVERIFIED), and the
statistic ignores parameter estimation uncertainty. Limit: with few observations the rank drops and the test has no power.

## GAS-FZ, rung 9

Source read in full text: Patton, Ziegel and Chen (2019), J. Econometrics 211(2):388-413, arXiv 1707.05108
(https://arxiv.org/abs/1707.05108). The FZ0 score of its eqs 10 to 12 and 18, with $\lambda_{v,t}=-v_t(\mathbb 1\{y_t\le v_t\}-\alpha)$
and $\lambda_{e,t}=\frac1\alpha\mathbb 1\{y_t\le v_t\}y_t-e_t$ (eqs 11 and 12):

$$\begin{pmatrix}v_{t+1}\\e_{t+1}\end{pmatrix}=w+B\begin{pmatrix}v_t\\e_t\end{pmatrix}+A\begin{pmatrix}\lambda_{v,t}\\\lambda_{e,t}\end{pmatrix}\quad\text{(two-factor, eqs 9 to 16)}$$

$$v_t=ae^{\kappa_t},\ e_t=be^{\kappa_t},\ b<a<0,\qquad \kappa_{t+1}=\beta\kappa_t+\gamma\,\frac{-1}{e_t}\Big(\frac1\alpha\mathbb 1\{y_t\le v_t\}y_t-e_t\Big)\quad\text{(one-factor, eq 20)}$$

$$\kappa_{t+1}=\beta\kappa_t+\gamma\,\frac{-1}{e_t}\Big(\frac1\alpha\mathbb 1\{y_t\le v_t\}y_t-e_t\Big)+\delta\log|y_t|\quad\text{(hybrid GAS/GARCH, eq 27)}$$

The intercept of the one-factor and hybrid recursions is not identified under FZ loss and is fixed at 0 (the paper does the same,
footnote 10). The matrix $B$ of the two-factor model is diagonal (Table 7 reports one persistence per equation) and $A$ is full:
`params = [w_v, w_e, B_vv, B_ee, A_vv, A_ve, A_ev, A_ee]`; one-factor `params = [beta, gamma, a, b]`; hybrid `params = [beta,
gamma, delta, a, b]`. Initial values (decisions of this unit, estimation-sample only): the sample $(v,e)$ for the two-factor model,
$\kappa_1=0$ for the one-factor and $\kappa_1=\delta m/(1-\beta)$ for the hybrid with $m$ the sample mean of $\log|y|$. The contract's
`fit_hybrid_garch_fz` is the paper's hybrid model (eq 27), "a hybrid GARCH-FZ" of dossier 05 A.1.

`fit_gas_fz(y, alpha, spec, n_starts, seed)` and `fit_hybrid_garch_fz(y, alpha, n_starts, seed)` minimise the average FZ0 loss
of the estimation sample by Nelder-Mead from `n_starts >= 2` starts; the first start is the constant empirical VaR and ES
(`beta = gamma = 0`, `a = v`, `b = e`; `B = A = 0`, `w = (v, e)`), so `mean_fz0` does not exceed the benchmark's; a path that
leaves `e <= v < 0` costs a large constant, so every fitted path stays in the domain; `converged = False` is a result, not an
error (A.2 rung 9: the estimation is unstable for small alpha). Deterministic for the same seed; `N >= 500`.
`gas_fz_forecast(fit, y)` runs the recursion through `y` and returns `(var, es)` for the next day with `es <= var < 0`
(`ValueError` if a non-converged fit leaves the domain). Limits: the objective is non-smooth; small `alpha` and short windows
give unstable parameters; the GAS recursions react to a return only through a VaR violation (the paper's Figure 3).
