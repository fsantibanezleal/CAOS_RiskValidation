# 04. The dependence-uncertainty band

Module `riskvalidation.aggregation.band`. Sources: dossier 05 G.3 ("Contraste should show the dependence-uncertainty band (lower
bound, Gaussian, t, comonotonic, worst case) instead of a single diversified number") and G.1; Embrechts, Puccetti and Ruschendorf
(2013), doi:10.1016/j.jbankfin.2013.03.014; the SDD (the five rows) with decision Q12 of the unit sizing adding the lower bound.

## The rows

`dependence_band(ppfs, ec, el, corr, nu_set, level, horizon_years, n_draws, seed, chunk_size, bounds_n, bounds_epsilon)` returns a
`Band` whose rows run over the same marginals, in this order:

| Row (`method`) | `nu` | Capital |
|---|---|---|
| `lower-bound` | none | best VaR (`worst_var_bounds` `lower_range[0]`) less the sum of the expected losses; signed |
| `simple-sum` | none | $\sum_k EC_k$, the headline |
| `variance-covariance` | none | $\sqrt{EC^\top R\,EC}$ |
| `gaussian-copula` | none | $\mathrm{VaR}-EL$ of the Gaussian copula aggregate |
| `t-copula` | each $\nu$ of `nu_set` | $\mathrm{VaR}-EL$ of the t copula aggregate |
| `worst-case` | none | worst VaR (`worst_var_bounds` `upper_range[1]`) less the sum of the expected losses |

$$\text{capital}_{row}=\mathrm{VaR}_{row}-\sum_kEL_k,\qquad DF_{row}=\frac{\text{capital}_{row}}{\sum_kEC_k}$$

Symbols: $\mathrm{VaR}_{row}$ the VaR of the total under the row's dependence, $EL_k$ the expected loss of marginal $k$, $EC_k$ its
economic capital, $DF$ the diversification factor `diversification_factor`. Every simulated row uses the same `seed`, so the band is
reproducible; `mc_se` is the Monte Carlo standard error of the VaR of a simulated row and is `None` for the others.

Each row maps one to one onto the document row `outputs.aggregation.band[*]` (`method`, `nu`, `capital_clp_m`,
`diversification_factor`, `status`, `mc_se_clp_m`). The status follows the method exactly: `headline-capital` for the simple sum,
`information-not-capital` for every other row (dossier 05 l.519); `nu` is set exactly for the `t-copula` rows. `BandRow` refuses a
row that breaks either rule.

## Conventions and refusals

- The lower-bound row is a signed capital: the best VaR less the sum of the expected losses may be negative (Table 4, $d=56$: best
  VaR 45.82 against an expected loss of 56) and is reported as computed, with its diversification factor; every other row is not
  below zero.
- A row above the headline is a true result (VaR is not subadditive): the band reports it, `validation.aggregation.diversification_sign`
  flags it, and nothing repairs or refuses it here.
- One horizon and one confidence level apply to every row: `level` lies in $(0,1)$ and `horizon_years` is a positive integer (the
  case uses 1 and 0.999); the engine cannot see the level of a marginal quantile function, so the caller converts the marginals.
- `ValueError` names the field for an empty, unsorted, repeated or not-above-2 `nu_set`, a correlation that fails
  `check_correlation`, inputs of different lengths, a negative `ec` or `el`, and an `ec` whose sum is zero (no diversification factor).
- An infinite-mean marginal arrives as `ec = VaR` and `el = 0.0`, so that every row is built on the EC the document carries.

## Golden values

`ec = (3, 4)`, `el = (0, 0)`, $R=[[1,0.5],[0.5,1]]$: the `simple-sum` row is 7.0 and the `variance-covariance` row is 6.0828 (computed),
with lower $\le 6.0828\le$ worst. On two scaled Pareto(2) marginals whose 99.9 percent quantiles are 3 and 4 (the tests' marginals),
measured at seed 5 with $2\cdot10^5$ draws and `bounds_n` 2000: lower 3.243, variance-covariance 6.083, Gaussian 5.569 (standard
error 0.161), t with $\nu=4$ 6.274 (0.304) and $\nu=8$ 5.766 (0.235), worst 9.962, against the simple sum 7.0.

## Limits

The band is information for the validator. It states no compliance claim and no capital except the simple-sum headline. The worst and
lower rows are rearrangement ranges (page 03), approximations with the limits stated there; the copula rows carry Monte Carlo error.
