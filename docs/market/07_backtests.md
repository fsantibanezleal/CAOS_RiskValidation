# 07. The market back-tests

Module `riskvalidation.validation.market`. Sources: dossier 05 B.1 to B.5, dossier 01 D.4. Each function returns one
`TestResult` that serialises into a row of `common/test-result` with `test_id = market.<function name>` and the `segment
alpha=<alpha>` for a test at a tail level. The lights are **Contraste internal gates**, not regulatory thresholds: red when `p <
alpha_red`, amber when `p < alpha_amber`, otherwise green (the fixtures use `alpha_amber = 0.10`, `alpha_red = 0.05`); the
parameters that are not alphas travel in `extras["thresholds"]`. A test that cannot run returns `not_evaluated` with its reason
in `notes` and a null statistic, metric and p-value, never a number. Flags (`exceedances`, `hits`) are `I_t = 1{loss_t >
VaR_t}`; the ES tests and the desk test take positive loss numbers (the document convention); a NaN return raises.

## Coverage tests

| Test | Statistic | Golden |
|---|---|---|
| `market.kupiec_pof` | $LR_{POF}=-2\ln\frac{(1-p)^{T-x}p^x}{(1-\hat\pi)^{T-x}\hat\pi^x}$, $\hat\pi=x/T$, chi-square(1); metric the exact binomial p-value | `T=250`, `x=5`, `alpha=0.01`: 1.956810, p 0.161855, exact 0.107812; `x=0`: 5.025168, p 0.024982 |
| `market.christoffersen_independence` | $LR_{ind}$ of the transitions $n_{ij}$, chi-square(1) | 20 days, exceedances at days 11 and 12 (transitions 16, 1, 1, 1): 2.407835, p 0.120729 |
| `market.christoffersen_conditional_coverage` | $LR_{cc}=LR_{POF}+LR_{ind}$, chi-square(2) | same sequence at `alpha=0.01`: $LR_{POF}$ 5.779174, 8.187009, p 0.016681 |

$$LR_{ind}=-2\ln\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}$$

with $\pi_{01}=n_{01}/(n_{00}+n_{01})$, $\pi_{11}=n_{11}/(n_{10}+n_{11})$, $\pi=(n_{01}+n_{11})/\sum n$ and $0^0=1$.
Sources: Kupiec, J. Derivatives 3(2):73-84, 1995, doi:10.3905/jod.1995.407942; Christoffersen, International Economic Review
39(4):841-862, 1998, doi:10.2307/2527341. Critical values at 5 percent: `chi2.ppf(0.95, 1) = 3.841`, `chi2.ppf(0.95, 2) = 5.991`.

**Decision D-1:** the exact binomial p-value is the total probability of the outcomes at most as likely as `x` under `Bin(T,
alpha)` (the dossier asks for it next to the asymptotic one and does not define the two-sided form). With `T = 250` and
`alpha = 0.01` the expected count is 2.5: the power is low (dossier 05 B.1) and the asymptotic test is not of nominal size (it
rejects when `x = 0` and when `x >= 6`, about 12 percent under a correct model, which the size test in `tests/validation_market`
computes exactly); the exact p-value is the safe reading at that window, the light follows the asymptotic p-value as the contract
states, and the row's `notes` say so. **Decision D-7:** the independence and conditional coverage tests are `not_evaluated` when
there is no exceedance or all days are exceedances (nothing is left to test for clustering; the convention $0^0=1$ would return
the number 0 for a test that did not run).

## Duration-based test (decision D-8)

Christoffersen and Pelletier model the no-hit durations with the Weibull hazard $\lambda(d)=a^bb\,d^{b-1}$; under correct
conditional coverage the durations are memoryless, $b=1$. `market.duration_test` maximises, for the completed spells $D_i$
between consecutive exceedances (density $a^bbD^{b-1}e^{-(aD)^b}$) and the censored first and last spells (survival
$e^{-(aD)^b}$), the log-likelihood over $(a,b)$ and over $b=1$:

$$LR=2\big[\ell(\hat a,\hat b)-\ell(\tilde a,1)\big],\qquad a^b=\frac{d_c}{\sum_i D_i^b}\ \text{(profiled out)},\qquad \ell(b)=d_c\ln\frac{d_c}{\sum D_i^b}+d_c\ln b+(b-1)\sum\ln D_i-d_c$$

with $d_c$ the number of completed spells. The p-value is by Monte Carlo (Dufour 2006): `n_mc` Bernoulli(alpha) sequences of the
same length, $p=(1+\#\{LR_{sim}\ge LR\})/(n_{mc}+1)$, and a simulated sequence with fewer than 2 exceedances has statistic 0;
the same inputs and seed give the same p-value. The first spell is censored only when the sample does not start with an exceedance (review m2); `extras["weibull_b_at_bound"]` and `notes` say when the shape reached the bound of its search `[0.05, 20]`, as it does for perfectly regular spells (review m3). **Decision D-8:** fewer than 2 exceedances is `not_evaluated` (no completed
spell; the dossier gives no minimum, so the minimum is the smallest that leaves a completed duration). Source: Christoffersen
and Pelletier, J. Financial Econometrics 2(1):84-108, 2004, doi:10.1093/jjfinec/nbh004. The Weibull parameterisation is the
standard one and was not re-checked against the paper's text (UNVERIFIED in dossier 05 B.3); no golden value exists, so the tests
compare the statistic with an independent optimisation of the same likelihood and measure the size of the whole test by
simulation (below).

## Basel traffic light

`market.basel_traffic_light` counts the exceptions in the last `window` days (`statistic`), returns the cumulative probability in
percent as `metric`, the zone as the light and the multiplier in `extras`, with a null p-value; `not_evaluated` with fewer than
`window` days. Golden (MAR32.9 Table 1): `k = 4` gives 89.22, green, 1.50; `k = 5` gives 95.88, amber, 1.70; `k = 9` gives
99.97, amber, 1.92; `k = 10` gives 99.99, red, 2.00.

## ES back-tests

`market.acerbi_szekely`: with $X_t=-\mathrm{loss}_t$ and $I_t=\mathbb 1\{\mathrm{loss}_t>\mathrm{VaR}_t\}$,

$$Z_1=\frac1{N_T}\sum_t\frac{X_tI_t}{\mathrm{ES}_t}+1,\qquad Z_2=\sum_t\frac{X_tI_t}{T\alpha\,\mathrm{ES}_t}+1$$

both of expectation 0 under $H_0$ and negative when risk is underestimated. The critical values are simulated per model from
`simulate_null(rng)` (one loss path per draw): $p=(1+\#\{Z_{sim}\le Z\})/(n_{sim}+1)$, light against `alpha_amber = 0.05` and
`alpha_red = 0.0001` (which needs at least 10,000 draws to resolve). The values -0.70 (5 percent, yellow) and -1.8 (0.01 percent,
red) for `T = 250` and `alpha = 2.5%` are a published reference point verified only through Moldenhauer and Pitera (arXiv
1709.01337, section 6), carried in `extras["reference"]` and never a regulatory threshold; the primary text (Risk, December 2014)
is gated and was not retrieved. Golden: `loss = [0,0,0,0,0.05]`, `var = 0.02`, `es = 0.03`, `alpha = 0.025` gives `Z2 = -12.333333`.
`not_evaluated` when an ES is NaN (for example a conditional EVT with `xi >= 1`) or, for $Z_1$, without an exceedance.

`market.du_escanciano_unconditional` and `market.du_escanciano_conditional` (Du and Escanciano, Management Science 63(4):940-958,
2017, doi:10.1287/mnsc.2015.2342): with the PIT $u_t=F_t(Y_t)$ the cumulative violation is
$H_t=\frac1\alpha(\alpha-u_t)\mathbb 1\{u_t\le\alpha\}$, with $E[H_t]=\alpha/2$ and $\mathrm{Var}(H_t)=\alpha(1/3-\alpha/4)$ (derived in dossier
05 B.5 from $u\sim U(0,1)$):

$$U_{ES}=\frac{\sqrt T(\bar H-\alpha/2)}{\sqrt{\alpha(1/3-\alpha/4)}}\to N(0,1),\qquad C_{ES}(m)=T\sum_{j=1}^m\hat\rho_j^2\to\chi^2_m$$

**Decision D-5:** a large positive $U_{ES}$ means underestimated risk, so the p-value is the one-sided upper tail (the dossier does
not state the side). The conditional statistic is the Box-Pierce statistic of the sample autocorrelations of $H_t$; a constant `H`
has no spread and is `not_evaluated`. The plain asymptotic versions ignore parameter estimation uncertainty, which `notes` states.
Golden: `alpha = 0.025` gives $E[H]=0.0125$ and $\mathrm{Var}(H)=0.0081770833$; `pit = 0.5` for `T = 100` days gives `H = 0` and
`U_ES = -1.382327`.

`market.mcneil_frey_residual_test` (McNeil and Frey, J. Empirical Finance 7(3-4):271-300, 2000): on the exceedance days
$r_t=(L_t-\mathrm{ES}_t)/\sigma_t$ should be i.i.d. with mean zero. The statistic is the mean of $r_t$; the p-value is one-sided
bootstrap: the residuals centred at their mean are resampled `n_boot` times and $p=(1+\#\{\text{mean}^*\ge\text{mean}\})/(n_{boot}+1)$
(a positive mean means the loss exceeds the ES). Golden: residuals `[1, 2, 3]` give `statistic = 2.0` and every centred resample
mean is at most 1.0, so `p = 1/(n_boot + 1)`. `not_evaluated` with fewer than 2 exceedances (decision D-8: a bootstrap needs
spread) or an NaN ES.

## Dynamic quantile, PLA and desk

`market.dynamic_quantile_test` uses the statistic of `caviar.dynamic_quantile_statistic` (page 04) and is `not_evaluated` when `T <= lags + 2`.
The asymptotic chi-square law is poor with sparse hits: the plain test rejected correct forecasts 9.9% of the time at (alpha 0.01,
T 250), 8.2% at (0.025, 250) and 14.3% at (0.01, 750) (review M1). The **light therefore follows a Monte Carlo p-value** (Dufour
form, as the duration test): `n_mc` Bernoulli(alpha) hit series with the same VaR column, $p=(1+\#\{S_{sim}\ge S\})/(n_{mc}+1)$,
reproducible for the same `seed`; the chi-square p-value stays in `extras["p_value_asymptotic"]`. The size test runs 2000
replications at each of the three configurations and requires a rate between 0.03 and 0.07.
`market.pla_test` returns the Spearman correlation as `statistic`, the KS statistic as `metric`, the zone of `frtb.pla_zone` as the
light, null p-value and alphas, and the boundaries in `extras["thresholds"]`; the zone is decided on integer ranks (`frtb.compare_spearman`, exact rational arithmetic) and the integer step gap (`frtb.pla_zone_exact`), never on a float quotient compared with a boundary (review m4); `not_evaluated` unless both series have exactly 250
finite observations and a nonzero spread. Golden: identical series give `statistic = 1.0`, `metric = 0.0`, green. `market.desk_backtest`
returns the exception count of the last 250 days as `statistic`, the limit of `frtb.desk_exception_limit` as `metric` and the basis
(`HPL` or `APL`) in `extras`; red when the count exceeds the limit, otherwise green (the dossier states one limit; there is no
amber). Golden: 12 exceptions at `alpha = 0.01` is green and 13 is red; 30 at 0.025 is green and 31 is red. Source: MAR32.18 to
MAR32.19 and MAR32.36 to MAR32.42 (dossier 01 D.4).

## Size of each back-test on correct forecasts

`tests/validation_market/test_market_size.py` simulates correct forecasts and counts the rejections at the red level (5 percent):
Kupiec, Christoffersen independence and conditional coverage and the dynamic quantile test at `alpha = 0.05`, `T = 1500`, 700
replications; the duration test at `alpha = 0.05`, `T = 400`, 240 replications with 99 Monte Carlo draws; Acerbi and Szekely
`Z2` (simulated critical values, normal losses, `T = 250`, 2.5 percent), the two Du and Escanciano tests (`alpha = 0.1`, `T = 2000`)
and McNeil and Frey (percentile bootstrap of a skewed mean, a documented band of 2 to 9.5 percent). Each rate lies within the
Monte Carlo error of 5 percent. Limits: the asymptotic Kupiec test is not of nominal size at `T = 250` and `alpha = 0.01`
(about 12 percent, computed exactly in the same file); the Weibull duration test and the DQ degrees of freedom are not checked
against the primary papers; a back-test states a sample fact, never a verdict on the model.

## Size and power of the p-value tests

The p-value tests of this page, measured on a generator where each null holds at the boundary (U16-E2, BL-088; the harness is `riskvalidation.harness`, the null and the alternative of each are named in `riskvalidation.generators.size_studies`, the runs are the `sizepower` suites under `tests/`, a local release gate that CI deselects). Size is the rejection rate at the nominal level over 4,000 repetitions, accepted when it is within the binomial bound alpha plus 3.09 standard errors; power is the rejection rate at 5% against one stated alternative, with no pass threshold. The figures below are the numbers of the run; the suite fails when a page states other ones.

**Size and power of `market.mcneil_frey_residual_test`** (2500 days at 5%, 125 exceedances expected; seed 5206, 4000 repetitions). Assumptions met: independent days of normal losses with a date-specific scale, the true VaR and ES at the 5% level (125 exceedances expected per run) and 999 bootstrap resamples inside each run (`n_boot`, the seed drawn by the harness from the repetition's stream). The test is conservative here; the exceedance residuals of a normal loss are right-skewed. The measured size is 3.25% at 5% (bound 6.06%) and 0.45% at 1% (bound 1.49%). Power at 5% against the alternative that the ES forecast is 95% of the true 5% expected shortfall of a normal loss (mean residual 0.103) (seed 5306, 4000 repetitions): 94.27%.

**Size and power of `market.kupiec_pof`** (1500 days at 5%; seed 101, 700 repetitions). Null: independent hits at 5%. A test rejects when the light is red (the red level, 5%). The measured size is 6.29% over 700 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that the VaR is exceeded 1.3 times too often (a hit rate of 6.5%, not 5%) (seed 5311, 2000 repetitions): 68.85%.

**Size and power of `market.christoffersen_independence`** (1500 days at 5%; seed 101, 700 repetitions). Null: independent hits at 5%. A test rejects when the light is red (the red level, 5%). The measured size is 7.00% over 700 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 15%, not 5% (clustered exceedances at the same hit rate) (seed 5312, 2000 repetitions): 87.45%.

**Size and power of `market.christoffersen_conditional_coverage`** (1500 days at 5%; seed 101, 700 repetitions). Null: independent hits at 5%. A test rejects when the light is red (the red level, 5%). The measured size is 8.14% over 700 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 15%, not 5% (clustered exceedances at the same hit rate) (seed 5313, 2000 repetitions): 80.85%.

**Size and power of `market.duration_test`** (400 days at 5%; seed 102, 240 repetitions). Null: independent hits at 5%. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 3.33% over 240 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 40%, not 5% (clustered exceedances at the same hit rate) (seed 5314, 1000 repetitions): 63.10%.

**Size and power of `market.acerbi_szekely`** (250 days at 2.5%; seed 103, 300 repetitions). Null: normal losses with their true VaR and ES. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 3.67% over 300 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that the realised losses are 1.25 times as dispersed as the VaR and ES forecasts say (seed 5315, 1000 repetitions): 90.80%.

**Size and power of `market.du_escanciano_unconditional`** (2000 days at 10%; seed 103, 600 repetitions). Null: uniform PITs. A test rejects when the p-value is below 0.05. The measured size is 5.33% over 600 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that the lower tail has too much mass (u = r^1.05: 11.2% of the PITs below 0.10, not 10%) (seed 5316, 2000 repetitions): 56.85%.

**Size and power of `market.du_escanciano_conditional`** (2000 days at 10%; seed 103, 600 repetitions). Null: uniform PITs. A test rejects when the p-value is below 0.05. The measured size is 4.50% over 600 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that the tail PITs (below 0.10) cluster: one follows another with probability 20%, not 10% (seed 5317, 2000 repetitions): 73.75%.

**Size and power of `market.dynamic_quantile_test`** (1500 days at 5%; seed 101, 700 repetitions). Null: independent hits. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 6.57% over 700 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 15%, not 5% (clustered exceedances at the same hit rate) (seed 5318, 1000 repetitions): 74.70%.

**Size and power of `market.dynamic_quantile_test`** (250 days at 1%; seed 104, 2000 repetitions). Null: independent hits. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 5.15% over 2000 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 10%, not 1% (clustered exceedances at the same hit rate) (seed 5319, 1000 repetitions): 15.60%.

**Size and power of `market.dynamic_quantile_test`** (250 days at 2.5%; seed 104, 2000 repetitions). Null: independent hits. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 4.85% over 2000 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 15%, not 2.5% (clustered exceedances at the same hit rate) (seed 5320, 1000 repetitions): 24.40%.

**Size and power of `market.dynamic_quantile_test`** (750 days at 1%; seed 104, 2000 repetitions). Null: independent hits. A test rejects when the Monte Carlo p-value is at most 0.05. The measured size is 3.90% over 2000 repetitions (one stream of the scenario, read by `tests/validation_market/test_market_size.py`). Power at 5% against the alternative that a hit is followed by a hit with probability 10%, not 1% (clustered exceedances at the same hit rate) (seed 5321, 1000 repetitions): 29.50%.
