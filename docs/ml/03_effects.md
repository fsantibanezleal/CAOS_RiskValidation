# 03. Effect curves: partial dependence, ICE and accumulated local effects

Module `riskvalidation.ml.effects`. Sources: dossier 06 C.7 l.994-1003 (PDP, ICE, ALE) and I.1 l.1193; Friedman (2001),
doi:10.1214/aos/1013203451; Goldstein, Kapelner, Bleich and Pitkin (2015), doi:10.1080/10618600.2014.907095; Apley and Zhu
(2020), doi:10.1111/rssb.12377. A model enters as a predict callable on arrays.

## Partial dependence and ICE

$$\hat f_S(x_S)=\frac1n\sum_{i=1}^{n}f(x_S,x_{i,C}),\qquad \mathrm{ICE}_i(x_S)=f(x_S,x_{i,C})$$

where $S$ is the feature varied and $C$ the others, kept at their observed values (06 C.7 l.994-999; the PDP assumes weak
dependence between $x_S$ and $x_C$). `ice_curves(predict, X, feature, grid)` returns one curve per row, shape `(n, m)`;
`partial_dependence` is the mean of the ICE curves over rows, **exactly** (the same array operation), shape `(m,)`. On an additive
model $f=\sum_j h_j(x_j)$ the curve is the component plus a constant (identity tested against a closed form).

## Accumulated local effects

Uncentred effect for feature $j$ with grid $z_{0}<\dots<z_{K}$ and neighbourhoods $N(k)=(z_{k-1},z_k]$ holding $n(k)$ rows
(06 C.7 l.1000-1003):

$$\hat g(z_k)=\sum_{l=1}^{k}\frac1{n(l)}\sum_{i:\,x_{ij}\in N(l)}\big[f(z_l,x_{i,\setminus j})-f(z_{l-1},x_{i,\setminus j})\big],\qquad
\hat g^{c}(z_k)=\hat g(z_k)-\frac1n\sum_{l=1}^{K}n(l)\,\hat g(z_l)$$

`accumulated_local_effects(predict, X, feature, n_bins)` returns the `n_bins + 1` edges and the centred effect at the **upper edge**
of each of the `n_bins` bins. The edges are the sample quantiles of the feature with the `inverted_cdf` definition, so every edge
is a data value and every bin $(z_{k-1},z_k]$ holds at least one row (the first bin also holds the minimum); a feature whose
quantile edges coincide (ties that would empty a bin) raises with `n_bins` named. The count-weighted mean of the returned effect
is 0 within 1e-12 (tested), and on an additive model it equals the component at the upper edges, centred with the same count
weights (identity tested with correlated features, where ALE avoids the PDP's extrapolation into unsupported regions).

## What these checks do not claim

- An effect curve describes the fitted model, not the world: it says how the prediction moves, not why a default occurs.
- The PDP and ICE evaluate the model at feature combinations that may not occur; ALE avoids that but depends on the quantile bins.
- The engine computes and makes no compliance claim.

## Sources

- Friedman (2001), "Greedy function approximation: A gradient boosting machine", Annals of Statistics 29(5), doi:10.1214/aos/1013203451.
- Goldstein, Kapelner, Bleich and Pitkin (2015), JCGS 24(1):44-65, doi:10.1080/10618600.2014.907095.
- Apley and Zhu (2020), "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models", JRSS-B
  82(4):1059-1086, doi:10.1111/rssb.12377. Dossier 06 C.7.

## Limits

The effect is the value at the upper edge of each bin; Apley and Zhu interpolate within bins to place the effect at each
observation, which the engine does not need for the range, sign and monotonicity checks that read the curve. UNVERIFIED: the
contract names "the Apley and Zhu estimator centred to mean zero" and does not fix the bin-edge definition; the `inverted_cdf`
quantiles are the engine's choice (they guarantee a row per bin).
