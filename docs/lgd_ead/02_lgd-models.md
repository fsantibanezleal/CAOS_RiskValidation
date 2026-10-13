# 02. LGD models

Module `riskvalidation.lgd_ead.lgd_models`. Source: dossier 04 F.2 and F.3. A design matrix $X$ includes its intercept column;
no function adds one. All fits are pure NumPy and SciPy and deterministic (a fixed start, optimiser and tolerance).

## The rungs

| Rung | Specification | Primary source |
|---|---|---|
| OLS (baseline) | $\mathrm{LGD}=x^\top\beta+\varepsilon$, clipped to $[0,1]$ in `predict_lgd` only | benchmark in Loterman et al. (2012), doi:10.1016/j.ijforecast.2011.01.006 |
| Fractional logit | $\mathbb E[y\mid x]=1/(1+e^{-x^\top\beta})$, Bernoulli quasi-ML, consistent for the endpoints | Papke and Wooldridge (1996), doi:10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1 |
| Beta regression | $y\sim\mathrm{Beta}(\mu\phi,(1-\mu)\phi)$, logit mean link, log precision link | Ferrari and Cribari-Neto (2004), doi:10.1080/0266476042000214501 |
| Hurdle (two-stage) | $\mathbb E[\mathrm{LGD}]=\Pr(\text{loss}>0)\,\mathbb E[\mathrm{LGD}\mid\text{loss}>0]$ | Leow and Mues (2012), doi:10.1016/j.ijforecast.2011.01.010; Tong, Mues and Thomas (2013), doi:10.1016/j.ijforecast.2013.03.003 |
| Tobit (two-limit) | $y^*=x^\top\beta+\varepsilon$, $y=\min(1,\max(0,y^*))$, $\varepsilon\sim N(0,\sigma^2)$ | Tobin (1958), doi:10.2307/1907382 |
| Cure decomposition | $\mathrm{LGD}=p_{\text{cure}}\,\mathrm{LGD}_{\text{cure}}+(1-p_{\text{cure}})\,\mathrm{LGD}_{\text{liq}}$ | practitioner decomposition; the single academic source was not pinned in dossier 04: attribution UNVERIFIED |

Symbols: $y$ the realised LGD in [0, 1], $x$ the covariates with the intercept, $\beta$ the coefficients, $\phi$ the beta
precision, $\sigma$ the Tobit scale, $G$ the logistic function.

## Fractional logit

The estimator maximises the Bernoulli quasi-log-likelihood

$$\ell(\beta)=\sum_i\big[y_i\log G(x_i^\top\beta)+(1-y_i)\log(1-G(x_i^\top\beta))\big],$$

solved by Newton iterations with step halving from a zero start; the score equation $X^\top(y-\mu)=0$ holds to $10^{-8}$ and,
with an intercept, $\sum\mu=\sum y$. A response outside [0, 1] raises. A coefficient that diverges (perfect separation, a
group sitting only on the endpoints) raises rather than returning a boundary value.

## Beta regression and the endpoint sensitivity

The beta density exists only on (0, 1), so the endpoints need a transformation:
$y^\epsilon=\min(\max(y,\epsilon),1-\epsilon)$ (`endpoint_transform`, $\epsilon\in(0,0.5)$). Qi and Zhao (2011),
doi:10.1016/j.jbankfin.2011.03.011, find the beta and inverse-Gaussian methods very sensitive to this small adjustment.
`fit_beta` therefore takes $\epsilon$ as an argument and records it in `extras["epsilon"]`; the precision is constant, so
`extras["phi"]` is the precision at the mean of $X$. The fit is a function of $\epsilon$ and the sensitivity (the same fit at
several $\epsilon$) is reported by the case, never hidden. The reference implementation in dossier 04 is
`statsmodels.othermod.betareg.BetaModel`, which its documentation marks experimental; this engine fits the same model with
SciPy (BFGS, analytic gradient) so that the core keeps NumPy and SciPy only.

## Hurdle

Stage 1 is a logit of $y>0$ on $X$; stage 2 is a fractional logit of $y$ on the severity design among the exposures with
$y>0$. `coefficients` hold the stage 2 values and `extras["stage1_coefficients"]` the stage 1 values. A severity design
other than $X$ must be a set of columns of $X$ (matched by value), whose indices are kept in `extras["severity_columns"]`, so
that `predict_lgd(fit, X)` needs only $X$. Fewer than `min_positive` (default 5) positive losses raise.

## Tobit

The log-likelihood has three parts: the normal density for $0<y<1$, $\log\Phi(-x^\top\beta/\sigma)$ for $y=0$ and
$\log[1-\Phi((1-x^\top\beta)/\sigma)]$ for $y=1$. It is maximised from the OLS start with the analytic gradient. With no $y$ at
0 or 1 the coefficients equal the OLS ones and $\sigma=\sqrt{\mathrm{RSS}/N}$. The prediction is the expectation of the
censored variable, $m[\Phi(b)-\Phi(a)]+\sigma[\varphi(a)-\varphi(b)]+1-\Phi(b)$ with $m=x^\top\beta$, $a=-m/\sigma$,
$b=(1-m)/\sigma$.

## Golden values

| Call | Value |
|---|---|
| `endpoint_transform([0, 0.5, 1], 0.001)` | `[0.001, 0.5, 0.999]` |
| `fit_ols`, $X=[[1,0],[1,1]]$, $y=[0,1]$ | coefficients `[0, 1]` (1e-12); `predict_lgd` of `[[1, 2]]` is `1.0` (clipped) |
| `fit_fractional_logit`, $X=[[1,0],[1,0],[1,1],[1,1]]$, $y=[0,0.5,0.5,1]$ | `[-1.098612, 2.197225]` = $[\ln(1/3), 2\ln 3]$; fitted means `[0.25, 0.75]` |
| `fit_hurdle`, intercept only, $y=[0,0.4,0.4,0.4]$, `min_positive=3` | stage 1 $\ln 3$, stage 2 $\ln(0.4/0.6)$, prediction 0.3 |
| `fit_tobit`, $X=[[1,0],[1,1],[1,2]]$, $y=[0.2,0.3,0.6]$ | `[0.166667, 0.2]`, $\sigma=0.0471405$ |
| `cure_decomposition(0.3, 0.05, 0.6)` | 0.435 |

## Limits

LGD $R^2$ is low in all the benchmark studies (average between 4% and 43% in Loterman et al. 2012; abstract content from
search snippets), so an LGD claim is judged by calibration (mean realised against mean predicted by segment and over time),
not by $R^2$ alone (page 06). Zero-or-one inflated beta (Ospina and Ferrari 2012), trees and survival-based LGD are not
rungs of this module. Every prediction of a rung is in [0, 1] and carries the flag `in-range`.
