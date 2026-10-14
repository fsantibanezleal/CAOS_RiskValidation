# 01. Historical simulation and the sign convention

Modules `riskvalidation.market.conventions` and `riskvalidation.market.historical`. Sources: dossier 05 A.0 and A.2 rows 1 to 3.
All results are in the return convention (VaR and ES negative decimals, `e <= v`); positive loss numbers only at the reporting
layer.

## The sign convention

`to_loss_numbers(values)` is `-values` and `from_loss_numbers` is its exact inverse (a NaN stays NaN, so an undefined ES is
reported as null by the caller, never as a number). It is the only place the sign is flipped. Golden: `[-0.023263, -0.026652]`
gives `[0.023263, 0.026652]`; `[0.02, 0.03]` gives `[-0.02, -0.03]`.

## VaR and ES from a sample (decision D-3, review F4)

$$k=\lceil\alpha N\rceil,\qquad \mathrm{VaR}=r_{(k)},\qquad \mathrm{ES}=\frac{\sum_{i<k}r_{(i)}+(\alpha N-(k-1))\,r_{(k)}}{\alpha N}$$

with `r(1) <= ... <= r(N)` sorted ascending. VaR is the left-continuous inverse `inf{y : F(y) >= alpha}` of dossier 05 A.0, and
ES the tail integral `(1/alpha) int_0^alpha F^-1(u) du` of the empirical quantile function: the k-th return carries the
fractional weight `alpha N - (k - 1)` when `alpha N` is not an integer (2.5 and 6.25 for the 250-day rung at 0.01 and 0.025; the
mean of the k smallest would overstate the tail count). Golden: `returns = [-0.001 i, i = 1..100]`: `alpha = 0.05` gives
`(-0.096, -0.098)`; `alpha = 0.025` gives `(-0.098, -0.0992)`. A floating product a hair above an integer (`0.05 * 100`) must not
move `k`, so `k = ceil(alpha N - 1e-9)`. The golden of the volatility-updated HS uses `alpha = 0.5`, which the contract's stated
domain `(0, 0.5)` excludes: this function accepts the closed upper bound 0.5 and refuses anything above it.

`var_es_from_distribution(ppf, alpha, n_nodes)` returns `var = ppf(alpha)` and `es = (1/alpha) int_0^alpha ppf(u) du` by
Gauss-Legendre quadrature. The integrand of an unbounded tail is singular at 0; the substitution `u = alpha s^3` keeps the
rule a Gauss-Legendre rule on (0, 1) and gives the normal goldens to 1e-5: `-2.326348` and `-2.665214` at 0.01, `-1.959964` and
`-2.337803` at 0.025.

## Historical simulation, rung 1

`historical_var_es(returns, alpha, window)` is `var_es_from_sample(returns[-window:], alpha)`; `ValueError` if there are fewer
returns than the window. A crisis return that leaves the window moves the result (the "ghost" effect of A.2 row 1 is a property
to display, not to hide). Golden: the 100-return series, `window = 100`, `alpha = 0.05` gives `(-0.096, -0.098)`.

## Age-weighted (hybrid) HS, rung 2 (decision D-9)

$$w_i=\frac{1-\lambda}{1-\lambda^{K}}\,\lambda^{\,i-1},\quad i=1..K,\qquad \sum_i w_i=1$$

`age_weights(lam, k)`: index 0 is the return observed one day ago. Golden: `lam = 0.98`, `k = 100` gives `w_1 = 0.023058` and
a sum of 1; the worked example of Boudoukh, Richardson and Whitelaw (1998) shows the weight `0.0221` three days ago and `0.0063`
sixty-five days ago, which the tests reproduce.

`age_weighted_var_es(returns, alpha, lam, window)` follows section 3 of the paper (read in full text for this unit): order the
returns ascending, cumulate the weights, and "linear interpolation is used between adjacent points". The paper fixes which
points: half of a return's weight lies on each side of it, up to the point halfway to the neighbouring return. With
$C_j$ the cumulated weight of the sorted return $j$, the quantile function $Q$ has **two knots per return**:

$$\big(C_j-w_j/2,\ r_j\big)\quad\text{and}\quad\Big(C_j,\ \tfrac{r_j+r_{j+1}}{2}\Big),$$

linear between knots, constant at the smallest return below the first knot and at the largest above the last; the VaR is
$Q(\alpha)$. The worked example of pages 6 and 7 uses exactly these knots: the -2.70% return sits at the 4.78th percentile, the
point halfway between -2.70% and -2.50% (-2.60%) sits at the cumulated weight 5.11%, and the 5% VaR is
$2.70\%-(2.70\%-2.60\%)\cdot(0.05-0.0478)/(0.0511-0.0478)=2.63\%$. A first version of this engine interpolated only between the return
knots and gave 2.647%; the Opus review (B1) caught it. Golden (the table of page 6, `lam = 0.98`, `window = 100`, `alpha = 0.05`,
the six returns -3.3, -2.9, -2.7, -2.5, -2.4 and -2.3% observed 3, 2, 65, 45, 5 and 30 days ago, every other return above
-2.3%): `var = -0.026338` (the paper: 2.63%) and `es = -0.030471`; the same six returns observed 28, 27, 90, 70, 30 and 55 days
ago give `var = -0.023419` (the paper: 2.34%) and `es = -0.028055`. The tests reproduce both.

The paper gives no rule for the ES (the contract marked it UNVERIFIED). **Decision D-9 of this unit:** the ES is the tail
integral $\frac1\alpha\int_0^\alpha Q(u)\,du$ of the same piecewise linear $Q$ (constant at the smallest return on
$(0,p_1)$), the definition of dossier 05 A.0 applied to the distribution the VaR is read from; the integral is exact (trapezoids).
It satisfies `es <= var` because $Q$ is nondecreasing. As `lam` tends to 1 the result is the equal-weight interpolation of the
paper (its historical simulation: 2.35% in the example, halfway between 2.40% and 2.30%), within one return spacing of
`historical_var_es`, not equal to it (review m1).

## Filtered HS and volatility-updated HS, rung 6 inputs (decision D-4)

$$r^*_{T+1,s}=\mu_{T+1}+\sigma_{T+1}\,z_s,\qquad z_t=\frac{r_t-\mu_t}{\sigma_t}\qquad\text{(Barone-Adesi, Giannopoulos and Vosper 1999)}$$

$$r^*_i=r_i\,\frac{\hat\sigma_{T+1}}{\hat\sigma_i}\qquad\text{(Hull and White 1998)}$$

`filtered_hs_var_es(std_residuals, mu_next, sigma_next, alpha)` uses every standardised residual (decision D-4: no random
bootstrap, so the forecast is deterministic) and returns `var_es_from_sample(scenarios, alpha)`; the conditional mean is removed
in `z`, so it is not counted twice. Golden: `z = [-2,-1,0,1,2]`, `mu_next = 0`, `sigma_next = 0.01`, `alpha = 0.2` gives
`(-0.02, -0.02)`. `volatility_updated_hs_var_es(returns, sigma_hist, sigma_next, alpha)`: golden `returns = [-0.02, 0.01]`,
`sigma_hist = [0.01, 0.02]`, `sigma_next = 0.015`, `alpha = 0.5` gives the scenarios `[-0.03, 0.0075]` and `(-0.03, -0.03)`;
with a constant volatility equal to `sigma_next` it equals `historical_var_es` on the whole input.

## Limits

HS needs the window to contain the tail it estimates (250 days at 1% have 2.5 tail days); it reacts slowly to a volatility
regime (the ghost effect); the age-weighted ES rule (D-9) is this unit's decision, not the paper's; filtered HS passes the
misspecification of its filter through. Sources: Pritsker (2001), FEDS 2001-27 https://www.federalreserve.gov/Pubs/feds/2001/200127/200127pap.pdf;
Boudoukh, Richardson and Whitelaw, Risk 11(5):64-67, 1998; Barone-Adesi et al., doi:10.1002/(SICI)1096-9934(199908)19:5<583::AID-FUT5>3.0.CO;2-S;
Hull and White, doi:10.21314/JOR.1998.001; Acerbi and Tasche, doi:10.1016/S0378-4266(02)00283-2 (the tail-integral ES).
