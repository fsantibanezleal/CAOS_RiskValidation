# 05. Scenario models

Rungs 2 and 3 of the macro stress ladder (dossier 05 H.2, l.388 and l.390): the VAR with a conditional forecast, and growth-at-risk.
They are engine functions with their own acceptance tests that no C18 variant bakes (decision Q13). The Minnesota BVAR and the panel
quantile satellite are backlog item BL-078. The module is pure NumPy and SciPy except `gar_quantile_regressions`, which reads
`statsmodels` (the optional extra `stress`, imported inside the function; without it the function raises `ImportError` naming the extra).

## The VAR

$$
y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,\qquad u_t \sim (0, \Sigma).
$$

`fit_var(y, lags)` estimates it by OLS equation by equation on $y$ of shape $(T, n)$; the residual covariance is $\hat\Sigma = U^\top U /
(T - p - (np + 1))$. It refuses $T \le p\,n + 1$ and a $T$ that leaves no residual degree of freedom. Golden value: data generated exactly by
$y_t = A y_{t-1}$ with $A = \begin{pmatrix}0.5 & 0\\0.2 & 0.8\end{pmatrix}$ recovers $A$ to $10^{-10}$ and an intercept of 0; the test also compares a VAR(2) with
intercept to an explicit loop and, when `statsmodels` is installed, to its `VAR` fit.

## The conditional forecast

A stress scenario is a conditional forecast given a path for a subset of variables (dossier 05 H.2, l.388): the conditioned variables
follow the path exactly. `var_conditional_forecast(fit, y_history, path, conditioned)` takes `y_history` of shape $(p, n)$ (oldest first)
and `path` of shape $(H, m)$ and returns $(H, n)$. In each period, with $\hat y_t = c + \sum_l A_l y_{t-l}$ the unconditional one-step prediction,
$c$ the conditioned indices and $f$ the free ones:

$$
y_{c,t} = \text{path}_t,\qquad u_{c,t} = y_{c,t} - \hat y_{c,t},\qquad y_{f,t} = \hat y_{f,t} + \Sigma_{fc}\,\Sigma_{cc}^{-1}\,u_{c,t}.
$$

With a diagonal $\Sigma$ the free variables follow their own equations with the conditioned path substituted; with correlated shocks the
conditional mean uses the regression of the free shocks on the conditioned ones. This is the period-by-period form; the full-horizon
Waggoner and Zha form is UNVERIFIED against the paper (the dossier names the method only). Golden value (hand-computed, diagonal
$\Sigma$): $A$ as above, $y_0 = (1, 1)$, the first variable conditioned on $(2.0, 3.0)$: the second variable is $(1.0, 1.2)$ ($0.2\cdot1 + 0.8\cdot1$, then
$0.2\cdot2.0 + 0.8\cdot1.0$); with no conditioning the forecast is $A^h y_0 = ((0.5, 1.0), (0.25, 0.9))$.

## The skewed t density

Growth-at-Risk step 2 (Adrian, Boyarchenko and Giannone 2019, dossier 05 H.2) fits the Azzalini-Capitanio skewed t density

$$
f(y;\mu,\sigma,\alpha,\nu) = \frac{2}{\sigma}\; t\!\left(\frac{y-\mu}{\sigma};\nu\right) T\!\left(\alpha\,\frac{y-\mu}{\sigma}\sqrt{\frac{\nu+1}{\nu+\big(\frac{y-\mu}{\sigma}\big)^2}};\;\nu+1\right),
$$

with $t(\cdot;\nu)$ and $T(\cdot;\nu)$ the Student t density and distribution function. `skewed_t_pdf` refuses $\sigma \le 0$ and $\nu \le 0$. Golden
values (computed): $\alpha = 0$ reduces to the Student t, `skewed_t_pdf(0, 0, 1, 0, 5) = 0.379607` (the $t_5$ density at 0); the density integrates
to 1 to $10^{-6}$ for $(\mu, \sigma, \alpha, \nu) = (0, 1, 2, 5)$. The mode lies on the side of the sign of $\alpha$.

`fit_skewed_t_to_quantiles(quantiles, taus)` minimises the squared distance between the model quantiles and the four estimated
quantiles (the 5, 25, 75 and 95 percent points by default). For each shape $(\alpha, \nu)$ the location and scale that minimise the distance
are the exact least squares solution of $q_i = \mu + \sigma z_i(\alpha,\nu)$, so only $(\alpha, \nu)$ are searched, by Nelder-Mead from four fixed starts, and
`distance` is the minimum found. The standard quantile $z_\tau$ solves $F(z) = \tau$ with $F(0) = \tfrac12 - \arctan(\alpha)/\pi$ (the sign of the variate is
that of the skew-normal numerator, so $F(0)$ does not depend on $\nu$; the test checks it against quadrature) and $F(z) - F(0)$ by Gauss-Legendre
panels. Golden value: the quantiles of a standard Student t with $\nu = 5$ at the four taus are recovered with $\alpha$ within $10^{-3}$ of 0 and a distance
of at most $10^{-8}$. Decreasing quantiles are refused.

## Growth-at-risk, step 1

$$
\hat\beta_\tau = \arg\min_\beta \sum_t \Big[\tau\,\mathbb{1}(y_{t+h}\ge x_t\beta)\,\lvert y_{t+h}-x_t\beta\rvert + (1-\tau)\,\mathbb{1}(y_{t+h}<x_t\beta)\,\lvert y_{t+h}-x_t\beta\rvert\Big],\qquad \hat Q_{y_{t+h}\mid x_t}(\tau) = x_t\hat\beta_\tau,
$$

the quantile regressions of average annualised growth on a constant, current growth and the Chicago Fed NFCI (dossier 05 l.390);
`gar_quantile_regressions(y, x, taus)` returns $(\text{len(taus)}, k)$ coefficients using `statsmodels` `QuantReg` (IRLS, at most 5000 iterations, `p_tol`
$10^{-10}$). It refuses a $\tau$ outside $(0, 1)$. Golden value: an exact line $y = 1 + 2x$ gives $(1.0, 2.0)$ at every $\tau$ to $10^{-6}$; the test
also solves the same problem as a linear program (SciPy HiGHS) and compares the attained check loss and the coefficients. The NFCI is in no
dossier 08 row, so a bake that needs it must register it first.

## Limits

- The conditional forecast is the period-by-period conditional mean; it is not the full-horizon Waggoner and Zha construction and states no
  forecast density. A VAR is a reduced-form association over a stated window, not a structural model.
- The skewed t fit has four free parameters for four quantiles: the fit is exact when the target is a skewed t and a compact summary
  otherwise. `nu` is searched on $[0.5, 1000]$; a normal target leaves the likelihood-free objective flat in $\nu$.
- The Minnesota BVAR (Litterman 1986; Banbura, Giannone and Reichlin 2010) and the panel quantile satellite (Covas, Rump and Zakrajsek 2014)
  are not implemented (BL-078). The full text of the quantile-regression sources was not read beyond the dossier's summary (UNVERIFIED detail).
- These functions are engine functions with acceptance tests; no C18 variant bakes them.

## Sources

- Adrian, T., Boyarchenko, N. and Giannone, D. (2019), Vulnerable growth, American Economic Review 109(4):1263-1289,
  doi:10.1257/aer.20161923.
- The Azzalini-Capitanio density as quoted in dossier 05 l.390; the original paper is not cited in the dossier (UNVERIFIED reference).
- Litterman, R. (1986), doi:10.1080/07350015.1986.10509491; Banbura, M., Giannone, D. and Reichlin, L. (2010), doi:10.1002/jae.1137;
  Covas, F., Rump, B. and Zakrajsek, E. (2014), doi:10.1016/j.ijforecast.2013.11.003 (dossier 05 H.2).
