# 07. The Minnesota BVAR and the panel quantile satellite

Modules `riskvalidation.stress.scenario_models` (`fit_bvar_minnesota` and its companions) and `riskvalidation.stress.panel_quantile`.
Requirements `RV-1911` to `RV-1919` in `docs/design/features/u19-ladder-rungs/requirements.md`, each naming its test. These two
rungs were deferred in 1.0 (BL-078). Both need NumPy and SciPy only; the tests reconcile with `statsmodels` (the `stress` extra)
where it is installed.

Sources, as read for this page:

- Banbura, Giannone and Reichlin, "Large Bayesian VARs", ECB Working Paper Series No 966, November 2008,
  https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp966.pdf: equations (1), (2), (5) to (9), from the rendered pages (the Greek symbols were
  lost in the text extraction). The journal version (Journal of Applied Econometrics, 2010) was not checked (UNVERIFIED).
- Karlsson, "Forecasting with Bayesian Vector Autoregression", Orebro University School of Business working paper 2012:12,
  https://www.oru.se/globalassets/oru-sv/institutioner/hh/workingpapers/workingpapers2012/wp-12-2012.pdf, equations (13) to (22); the
  chapter is Handbook of Economic Forecasting vol. 2, 2013, pp. 791-897, doi:10.1016/B978-0-444-62731-5.00015-4 (the DOI as listed in
  search results; the chapter itself was not opened).
- Litterman (1986), Doan, Litterman and Sims (1984), Kadiyala and Karlsson (1997): paywalled, not read; the two reviews above restate them (UNVERIFIED as primaries).
- Koenker, "Quantile regression for longitudinal data", Journal of Multivariate Analysis 91(1):74-89, 2004,
  doi:10.1016/j.jmva.2004.05.006: equations (2.2) to (2.4) and Theorems 1 and 2, read in the author's working version.
- Canay, "A simple approach to quantile regression for panel data", The Econometrics Journal 14(3):368-386, 2011,
  doi:10.1111/j.1368-423X.2011.00349.x: equations (2.1), (4.1), (4.2), Theorem 4.1, (4.7), appendix A.17 and B, Table 1, read in the
  author's copy.
- Besstremyannaya and Golovan, "Reconsideration of a simple approach to quantile regression for panel data", CEFIR/NES Working Papers
  248 and 249, 2018 and 2019; Econometrics Journal 22(3):292-308, 2019, doi:10.1093/ectj/utz012 (the journal article was not read: UNVERIFIED).
- Chen and Huo, "A simple estimator for quantile panel data models using smoothed quantile regressions", arXiv:1911.04729v1,
  doi:10.48550/arXiv.1911.04729; Econometrics Journal 24(2):247-263, 2021, doi:10.1093/ectj/utaa023 (the published version was not read: UNVERIFIED).
- Covas, Rump and Zakrajsek, "Stress-testing US bank holding companies: a dynamic panel quantile regression approach", International
  Journal of Forecasting 30(3):691-713, 2014, doi:10.1016/j.ijforecast.2013.11.003, read as FEDS Working Paper 2013-55.
- Chernozhukov, Fernandez-Val and Galichon, "Quantile and probability curves without crossing", Econometrica 78(3):1093-1125, 2010 (the
  rearrangement, cited by Covas et al.; not read).

## The Minnesota BVAR

### The prior

The VAR $Y_t=c+A_1Y_{t-1}+\dots+A_pY_{t-p}+u_t$, $u_t\sim N(0,\Psi)$ in $n$ variables. Litterman's Minnesota prior (Banbura et al. (2)):
$E[(A_k)_{ij}]=\delta_i$ if $j=i$ and $k=1$, else $0$; $V[(A_k)_{ij}]=\lambda^2/k^2$ if $j=i$ and $\theta(\lambda^2/k^2)(\sigma_i^2/\sigma_j^2)$ otherwise,
with $\delta_i=1$ for a persistent variable (a random walk) and $0$ for a mean-reverting one, $\lambda$ the overall tightness
($\lambda\to0$ the prior, $\lambda\to\infty$ the OLS fit), $k^{-2}$ the lag decay, $\sigma_i^2/\sigma_j^2$ the scale correction, and $\theta\in(0,1)$ the
cross-variable tightness. The scales $\sigma_i$ are the residual standard deviations of a univariate AR($p$) of each variable (Banbura et al.,
after Litterman and Sims and Zha); the engine uses $\sqrt{\mathrm{RSS}/(T-p-(p+1))}$, and $\mathrm{RSS}/(T-p)$ when no degree of freedom is left.

### The dummy-observation form (`bvar_dummies`, `bvar_posterior`, `fit_bvar_minnesota`, RV-1911)

With $Y=XB+U$, $B=(A_1,\dots,A_p,c)'$ of size $k\times n$, $k=np+1$, the Normal-inverted-Wishart prior with $\theta=1$ is imposed by $T_d$
dummy observations (Banbura et al. (5) and the text before it), with $J_p=\mathrm{diag}(1,\dots,p)$:

$$Y_d=\begin{pmatrix}\mathrm{diag}(\delta_1\sigma_1,\dots,\delta_n\sigma_n)/\lambda\\ 0_{n(p-1)\times n}\\ \mathrm{diag}(\sigma_1,\dots,\sigma_n)\\ 0_{1\times n}\end{pmatrix},\quad
X_d=\begin{pmatrix}J_p\otimes\mathrm{diag}(\sigma_1,\dots,\sigma_n)/\lambda&0_{np\times1}\\ 0_{n\times np}&0_{n\times1}\\ 0_{1\times np}&\varepsilon\end{pmatrix},$$

the three blocks being the prior on the autoregressive coefficients, on the covariance and the nearly uninformative prior on the intercept
($\varepsilon=10^{-6}$ by default); $T_d=np+n+1$. The posterior (Banbura et al. (6) and (7)) stacks the data and the dummies, $Y_*=(Y',Y_d')'$,
$X_*=(X',X_d')'$:

$$\tilde B=(X_*'X_*)^{-1}X_*'Y_*,\qquad \tilde\Sigma=(Y_*-X_*\tilde B)'(Y_*-X_*\tilde B),\qquad \Psi\mid Y\sim iW\big(\tilde\Sigma,\ T+T_d+2-k\big),\quad
B\mid\Psi,Y\sim N\big(\tilde B,\ \Psi\otimes(X_*'X_*)^{-1}\big),$$

the improper prior $|\Psi|^{-(n+3)/2}$ having been added for the mean of $\Psi$ to exist. The posterior mean of the coefficients is the OLS
fit of the stacked regression and equals the posterior mean of the Minnesota setup (2) with $\theta=1$. `fit_bvar_minnesota` returns it as a `VarFit` (the
intercept, $A_1,\dots,A_p$, `sigma`) that `var_conditional_forecast` consumes unchanged; `bvar_posterior` returns the whole posterior as a
`BvarPosterior`, and `bvar_dummies` the dummies.

The sum-of-coefficients prior (Banbura et al. (8) and (9); Doan, Litterman and Sims 1984), added when `sum_coef_tau` is set, appends
$n$ rows, $Y_d=\mathrm{diag}(\delta_i\mu_i)/\tau$ and $X_d=\big(\mathbf 1_p'\otimes\mathrm{diag}(\delta_i\mu_i)/\tau,\ 0\big)$ with $\mu_i$ the sample mean of $y_i$:
as $\tau\to0$ the sum $A_1+\dots+A_p$ tends to the identity (inexact differencing; the model tends to a random walk with drift), and
$\tau\to\infty$ removes the prior. Banbura et al. use $\tau=10\lambda$. **Transcription note:** the research report read the weight row of (9) as
$(1\ 2\ \dots\ p)$; Karlsson's equivalent dummy uses a vector of ones, and only a row of ones gives the limit $\sum_lA_l=I$ that RV-1913 requires,
so the engine uses ones (UNVERIFIED against the printed (9)).

### The inverse-Wishart convention (RV-1912)

The posterior of $\Psi$ is iW$(\tilde\Sigma,\ T+T_d+2-k)$ under the convention whose density is proportional to
$|\Psi|^{-(v+n+1)/2}\exp\{-\mathrm{tr}(S\Psi^{-1})/2\}$ (SciPy's `scipy.stats.invwishart`), whose mean is $S/(v-n-1)$. Neither source states its
convention in the pages read; the engine decides and tests this one. `VarFit.sigma` is the posterior mean of $\Psi$,

$$\Sigma_{\text{post}}=\frac{\tilde\Sigma}{T+T_d+2-k-n-1}.$$

**The two sources differ.** Karlsson's proper-prior form (19) has $\bar v=T+v_0$, and the dummy form has $v_0=\alpha_0=T_d-k$, so that
$\bar v=T+T_d-k$: **two degrees of freedom fewer** than Banbura et al., who add the improper prior. The coefficients and $\tilde\Sigma$ are the same in both; only the
degrees of freedom, and so the mean $\Sigma_{\text{post}}$ and the draws of $\Psi$, differ. The engine follows Banbura et al. (the posterior mean of $\Psi$ exists for
every $T$, since $T+T_d+2-k-n-1=T+1$).

`bvar_posterior_draws(posterior, n_draws, seed)` draws $\Psi$ from `invwishart` and then $B\mid\Psi$ from the matricvariate normal,
$B=\tilde B+L_\Omega Z L_\Psi'$ with $L_\Omega L_\Omega'=(X_*'X_*)^{-1}$, $L_\Psi L_\Psi'=\Psi$ and $Z$ standard normal (Karlsson Algorithm 1 with (18) and (19)); the
`BvarDraws` hold the draws of $\Psi$ and $B$ and convert a draw to a `VarFit` for a scenario fan. The mean of the $\Psi$ draws is within Monte Carlo error of $\Sigma_{\text{post}}$.

### Identities the tests assert (RV-1913)

Karlsson's closed form for the dummy prior, $B_0=(X_d'X_d)^{-1}X_d'Y_d$, $\Omega_0=(X_d'X_d)^{-1}$, $S_0=(Y_d-X_dB_0)'(Y_d-X_dB_0)$:

$$\bar B=(\Omega_0^{-1}+X'X)^{-1}(\Omega_0^{-1}B_0+X'Y)\ \ (18),\qquad \bar S=S_0+S+(B_0-\hat B)'\big(\Omega_0+(X'X)^{-1}\big)^{-1}(B_0-\hat B)\ \ (19),$$

agree with the stacked regression to $10^{-12}$ and $10^{-10}$ (measured $10^{-16}$ and $10^{-13}$ in the research). A very large $\lambda$ ($10^8$)
reproduces `fit_var` to $10^{-6}$; $\lambda\to0$ gives $A_1=\mathrm{diag}(\delta)$ and zero other lags; a univariate AR(1) equals the scalar ridge
to the random walk, $\hat a=(\tilde x'\tilde y+\sigma^2/\lambda^2)/(\tilde x'\tilde x+\sigma^2/\lambda^2)$ on the demeaned series (the intercept is nearly free);
the dummies are written entry by entry in the test as a second route. No published numeric table of the Minnesota prior can be asserted (Banbura et al.'s Table 1 are
out-of-sample relative MSFEs on a 131-series dataset that is not available here): a gap, the identities carry the test.

### Refusals and limits (RV-1914)

`ValueError` naming the field for fewer than `lags + 2` observations (`y`), `lam` of 0 or less or not finite, a non-finite `y`, a `delta` or `sigma_scale` of the wrong
length, non-positive `sum_coef_tau` and `eps`, and, when the AR($p$) scale cannot be estimated (a variable with zero residual), `sigma_scale`.

- The conjugate form has **one** cross-variable tightness: $\theta=1$ for all coefficients, so the distinction of Litterman's $\pi_2$ (other variables tighter than
  own lags) is not available, and the equation-by-equation Litterman estimator (a known diagonal covariance, pseudo-data of Karlsson (15)) is not implemented.
- There is no default $\lambda$. Karlsson reports 0.2 for the overall tightness in Litterman's data and Banbura et al. choose it per model size by an in-sample
  fit criterion; both are inputs here, not hidden constants. `delta`, `sum_coef_tau`, `eps` and `sigma_scale` are arguments.
- Not read: the tabulated $\lambda$ per model size and the series-specific $\delta$ of Banbura et al. (Appendix A).

## The panel quantile satellite

### The model and the two estimators (`fit_panel_quantile`, RV-1915)

For entities $i=1,\dots,n$ (banks) and periods $t$: $Q(\tau\mid x_{it})=\alpha_i+x_{it}'\beta(\tau)$ (Koenker (2.2), Canay (2.3)): the entity effect is a pure
location shift common to the quantiles, only the slopes depend on $\tau$.

**Two-step** (Canay (4.1), (4.2), A.17; the default). Step 1: the within estimator $\hat\gamma$ of the mean slopes and
$\hat\alpha_i=\bar T^{-1}\sum_t(Y_{it}-X_{it}'\hat\gamma)$; step 2: the pooled quantile regression of the corrected response,

$$\hat\theta(\tau)=\arg\min_\theta\ \frac1{nT}\sum_{i,t}\rho_\tau\big(Y_{it}-\hat\alpha_i-X_{it}'\theta\big),\qquad\rho_\tau(u)=u\{\tau-\mathbf 1(u<0)\},$$

a linear program per $\tau$. A design column that does not vary within entities (the constant) has no within estimate: it is left out of step 1 and kept in step 2, and
$\hat\alpha_i$ is centred to a zero mean over entities (Canay's $E(\alpha_i)=0$), so that the constant is the location of the corrected response.

**Penalized** (Koenker (2.4); `method="penalized"`). The linear program

$$\min_{\alpha,\beta}\ \sum_k\sum_{i,t}w_k\,\rho_{\tau_k}\big(y_{it}-\alpha_i-x_{it}'\beta(\tau_k)\big)+\lambda\sum_i|\alpha_i|,$$

solved by `scipy.optimize.linprog(method="highs")` on sparse matrices (variables $\beta_k$ free, $\alpha_i=\alpha_i^+-\alpha_i^-$, and the positive and negative parts of the residuals).
$\lambda=0$ is the fixed-effects estimator (give no constant column: the effects absorb it); $\lambda\to\infty$ drives every $\alpha_i$ to 0, the pooled quantile regression;
the weights default to $1/q$ and the penalty to $\lambda=1$ (Covas et al.; Lamarche 2010 on the choice of $\lambda$ was not read). The effects are common to the quantiles, so with
several $\tau$ the problem is joint, with one linear program of $n+pq$ parameters (Koenker states the cost; Canay's simulations find it about 15 times slower than the two-step).

The result is a frozen `PanelQuantileFit`: `taus` (sorted), `coef` $(q,p)$, `alpha` $(n)$, `entities`, `method`, `n`, `T` (mean periods per entity), `n_obs`, `converged`
(every linear program optimal), the regime `n_over_t`, `n2_over_t` and `wide_panel` (true at $n/T\ge1$), the within slopes `gamma` and the penalized `lam` and `weights`.

### Why no analytic standard error (RV-1918)

The fit carries no standard error, covariance or interval field; the intervals of the satellite and of `project_panel_quantiles` come only from `panel_quantile_bootstrap`.

- Canay's Theorem 4.1 needs $n/T^s\to0$ for some $s>1$ and gives the covariance $J_1^{-1}\Sigma J_1^{-1\prime}$ with $\Sigma=S+J_2\Gamma_g+\Gamma_g'J_2'+J_2\Lambda J_2'$ (4.7).
- **Besstremyannaya and Golovan** (working paper 249, Proposition 2.1) show the slope bias is of exact order $1/T$ and does not fall with $n$: $\sqrt{nT}$ times it is nonzero unless
  $n/T\to0$, so the limit has a nonzero mean or does not exist in wide panels; and the standard error of the constant is wrong (Propositions 2.2 and 4.3: the constant is at best
  $\sqrt n$-consistent, with $\mathrm{Var}(\hat\mu)\ge(\sigma_\alpha^2+\sigma^2/T)/n$, not $\sqrt{nT}$). Their simulations find the slope standard error about right and the problem the bias.
  Their own proof of the bias is a heuristic expansion (their Remark 2.1 calls it a conjecture).
- **Chen and Huo** locate the error in Canay's Lemma A.1, find an asymptotic bias of order $\sqrt{n}/T^{1/4}$ in the stated remainder, so that the condition to ignore the bias is
  $n^2/T\to0$ (not $n/T\to0$; the two critiques differ on the sufficient rate), and a **first-step term** $AB^{-1}\tilde X_{it}\varepsilon_{it}$ **missing from Canay's covariance**, so analytic
  slope intervals are invalid too. They propose a smoothed second step with bias corrections (not implemented: a nonlinear minimization with a bandwidth that remains open).
- The panels the satellite targets (a bank-level panel with $n$ in the tens or hundreds and $T$ near 20 to 60) sit in the regime of the two critiques; `wide_panel` and `n2_over_t` state it.

The entity bootstrap does not cure the bias in wide panels either (Besstremyannaya and Golovan, Remark 2.2: the bootstrap distribution converges to a limit with a large bias), and its validity is a
conjecture in Canay's appendix. The intervals are therefore labelled: always conjecture-based, heuristic when `wide_panel`.

### Bootstrap and projection (`panel_quantile_bootstrap`, `project_panel_quantiles`, RV-1918 and RV-1919)

`panel_quantile_bootstrap(..., scheme, n_boot, mean_block, seed, level)`: `scheme="entity"` resamples whole entities with replacement (Canay appendix B.5 and B.6); `"stationary"` resamples the periods
with the stationary bootstrap of Politis and Romano (1994), as Covas et al.: a start uniform on the $T$ periods, then the next period with probability $1-1/m$ and a new uniform start otherwise, wrapping
around, mean block $m=4$ (their 5,000 replications give 95 percent intervals from the 125th and 4,875th values); every entity's row at a resampled period is kept, so the cross-section is preserved (a balanced panel and a `time` index
are needed). The result holds the replicates, the percentile limits at `level` and the `label` of what the interval is.

`project_panel_quantiles(fit, x_future, bootstrap=None)` returns the quantile grid $\alpha_i+x_{it}'\beta(\tau_k)$ per entity, period and $\tau$ for future covariates $(H,p)$ (the scenario path, the same for
every entity) or $(n,H,p)$; the grid is **rearranged** by sorting along $\tau$ when quantiles cross (Chernozhukov, Fernandez-Val and Galichon 2010, as Covas et al.), so that it is monotone in $\tau$, and
`crossings` counts the cells that crossed. The width of `x_future` must be the fit's $p$. With a bootstrap the limits are the percentiles of the rearranged grids of the replicates (the effects of the fit for the
entity scheme, the replicate's for the stationary one): a distribution per bank and quarter, not a point.

### Evidence (RV-1916, RV-1917)

- **Canay's Table 1**, Model 1 (the golden values are the published table, not typed results): $Y_{it}=(\varepsilon_{it}-1)+\varepsilon_{it}X_{it}+\alpha_i$, $\alpha_i=2(\sum_tX_{it}+\eta_i)-E\alpha_i$, $X_{it}\sim U(0,1)$, $\eta_i\sim N(0,1)$, $\varepsilon_{it}\sim N(2,1)$,
  so that the slope at $\tau=0.25$ is $\Phi^{-1}(0.25)+2=1.3255$; $n=100$, 1,000 replications (`tests/stress/test_panel_quantile.py::test_canay_table_1`, marker `sizepower`: the local release gate, not CI).
  Measured by the engine, against the published relative bias and MSE: $T=20$ 0.0390 and 0.0283 (0.0377 and 0.0264), $T=10$ 0.0781 and 0.0591 (0.0793 and 0.0605), $T=5$ 0.1342 and 0.1373 (0.1494 and 0.1473); the test
  asserts bias within 0.02 and MSE within 25 percent, and the bias within 0.02 of Chen and Huo's reproduction (0.036 at $T=20$, 0.077 at $T=10$).
- The limits of Koenker's estimator ($\lambda=0$ against `statsmodels.QuantReg` with entity dummies, $\lambda=10^6$ against the pooled `QuantReg`, to $10^{-6}$), a single entity against `gar_quantile_regressions`, and
  the invariance to a constant per entity (slopes unchanged, effects shifted).

## Limits

The two satellites are statistical models, not capital figures: no compliance claim is made. The limits of the BVAR and of the panel inference are stated above (the single cross-variable tightness, the conventions of the
inverse Wishart, no analytic panel interval); the data shape and the sources follow.

A balanced (or near balanced, at least 2 periods per entity) panel of a bank-level rate (a charge-off rate, a PPNR component as a percent of assets), predetermined bank covariates and the scenario macro paths
(Covas et al. Section 4: 15 bank holding companies, 1997 to 2011, $T=60$). No bank-level panel is registered in dossier 08; the FFIEC Call Report and NIC pages refused automated access (CAPTCHA, HTTP 403) and
their formats are UNVERIFIED; the CMF Chile monthly bank statements are candidates for registration, their layout not opened. The satellite is a conditional-distribution model of one rate: it does not replace the
top-down comparison of `validation.stress` and it is not capital.
