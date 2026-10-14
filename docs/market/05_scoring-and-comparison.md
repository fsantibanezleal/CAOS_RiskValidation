# 05. Scoring and comparison: FZ0, tick loss, Diebold-Mariano, Model Confidence Set

Module `riskvalidation.market.scoring`. Sources: dossier 05 A.1 and B.6. Return convention (VaR `v < 0`, ES `e < 0`, `e <= v`).

## The FZ0 loss

Patton, Ziegel and Chen (2019, J. Econometrics 211(2):388-413, doi:10.1016/j.jeconom.2018.10.008; arXiv 1707.05108 eq. 4)
restate the Fissler-Ziegel class (Annals of Statistics 44(4):1680-1707, 2016, doi:10.1214/16-AOS1439):

$$L_{FZ}(Y,v,e;\alpha,G_1,G_2)=(\mathbb 1\{Y\le v\}-\alpha)(G_1(v)-G_1(Y))+G_2(e)\Big(e-v+\frac1\alpha\mathbb 1\{Y\le v\}(v-Y)\Big)-\mathcal G_2(e)$$

and, with $G_1=0$, $G_2(x)=-1/x$, $\mathcal G_2(x)=-\log(-x)$ (Proposition 1: the loss differences are homogeneous of degree
zero), the FZ0 loss (eq. 6):

$$L_{FZ0}(Y,v,e;\alpha)=-\frac{1}{\alpha e}\mathbb 1\{Y\le v\}(v-Y)+\frac ve+\log(-e)-1$$

`fz0_loss(y, v, e, alpha)` returns the per-day loss (dimensionless; it can be negative; only differences between models are
meaningful) and refuses `v >= 0`, `e >= 0`, `e > v` and a shape mismatch. The test compares it to the general loss with the
reduction above to `1e-12`. Golden: `(y, v, e, alpha) = (-0.03, -0.02, -0.025, 0.01)` gives `36.111121`; `(0.01, -0.02, -0.03,
0.025)` gives `-3.839891`; `(0.0, -0.02, -0.02, 0.01)` gives `-3.912023` (`log 0.02`: with `e = v` and `y > v` the loss is `log(-v)`).
FZ0 has no tuning parameter, which makes it the default scoring rule for the model comparison. The indicator is `1{y <= v}`; the
exceedance of the back-tests is `y < v`; the two differ only at equality, which the documents treat as no exceedance.

`tick_loss(y, v, alpha)` is $(\alpha-\mathbb 1\{y<v\})(y-v)$, nonnegative, zero at `y = v`: golden `(-0.03, -0.02, 0.01)` gives
`0.0099` and `(-0.01, -0.02, 0.01)` gives `0.0001`. `mean_scores(y, v, e, alpha)` returns `Scores(n, n_fz0, fz0_mean,
tick_mean)`: the tick loss over all `n` days and the FZ0 over the days with a finite `e` (NaN where the ES is undefined); a model
that forecasts no ES has `n_fz0 = 0` and `fz0_mean = None`, never a number.

## Diebold-Mariano (decision D-2)

$$d_t=L_t^A-L_t^B,\qquad DM=\frac{\bar d}{\sqrt{\widehat{LRV}/T}},\qquad \widehat{LRV}=g_0+2\sum_{j=1}^{h}\Big(1-\frac{j}{h+1}\Big)g_j,\qquad g_j=\frac1T\sum_{t}(d_t-\bar d)(d_{t-j}-\bar d)$$

with $p=2(1-\Phi(|DM|))$. Source: Diebold and Mariano, JBES 13(3):253-263, 1995, doi:10.1080/07350015.1995.10524599. The dossier
says only "a HAC long-run variance"; **decision D-2:** the Bartlett kernel (Newey-West), named `bartlett` in the documents. The
forecasts of the contract are one-step ahead, so `lags = 0` (the documents record the value in `comparison.hac_lags`); the
primary text for the lag rule was not read. For VaR use the tick loss, for the pair (VaR, ES) use FZ0. A negative `mean_diff`
means model A has the lower loss. `ValueError` if `d` has no spread or `T <= lags + 1`. Golden: `d = [0.1, -0.05, 0.2, 0.1, 0,
0.15]`, `lags = 0` gives `DM = 2.401922` and `p = 0.016309`; the tests recompute the statistic for `lags` 1 and 5 with an
independent implementation.

## Model Confidence Set

`model_confidence_set(losses, confidence, statistic, n_bootstrap, block_length, seed)` runs `arch.bootstrap.MCS` (the stationary
bootstrap, block length `block_length`, `statistic` `TR` or `Tmax`) on the `(T, M)` losses and returns `McsResult(members,
p_values, confidence, statistic)`: a model is in the set iff its MCS p-value is at least `1 - confidence`; the set is never empty
(the last survivor has p-value 1); `M = 1` gives that model with p-value 1; the same inputs and seed give the same result.
Source: Hansen, Lunde and Nason, Econometrica 79(2):453-497, 2011, doi:10.3982/ECTA5771. Limits: the MCS needs the optional
`arch`; with few days the set is large (it cannot separate close models); it is a comparative backtest in the sense of Nolde and
Ziegel (Annals of Applied Statistics 11(4):1833-1874, 2017, doi:10.1214/17-AOAS1041), not a proof that the winner is right.
