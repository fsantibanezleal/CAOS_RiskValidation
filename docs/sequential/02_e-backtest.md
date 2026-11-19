# 02. E-backtests of VaR and expected shortfall

Module `riskvalidation.validation.sequential.e_backtest` (names re-exported from `riskvalidation.validation.sequential`).
Requirements `RV-2101` to `RV-2115` in `docs/design/features/u20-e-backtest/requirements.md`, each naming its test. The
registered ids are `sequential.e_backtest_var` and `sequential.e_backtest_es` (BL-121).

The page transcribes the e-backtesting method of Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991 (version 7,
read in full; equation, theorem, proposition and remark numbers are those of that version). The method is theirs; the
engine implements it, tests it against the paper and against a reference function written in the Contraste manuscript
on e-process monitors, and does not change it. The monitors of [the sequential index](README.md) are bets on stated
alternatives; this one is a bet on the forecaster's own risk number, with no alternative to choose.

## What it tests

A forecaster reports, before day $t$, a VaR $v_t$ and (for the ES test) an expected shortfall $r_t$ at the same level for
the loss $L_t$ of day $t$. The tester is a supervisor concerned with under-forecasting, which is the case a regulator
penalises: the paper tests the one-sided null

$$
H_0^{\mathrm{VaR}}:\; v_t \ge \mathrm{VaR}_{p}(L_t \mid \mathcal F_{t-1}),\qquad
H_0^{\mathrm{ES}}:\; r_t \ge \mathrm{ES}_{p}(L_t \mid \mathcal F_{t-1}),\; v_t = \mathrm{VaR}_{p}(L_t \mid \mathcal F_{t-1}),
\quad t \ge 1, \tag{8, 9}
$$

with no model for the losses, no independence and no stationarity: the filtration $\mathcal F_{t-1}$ holds everything known
before day $t$. In the engine's convention losses are positive, a hit is $L_t > v_t$ and the exceedance probability is
$\alpha = 1 - p$ (a 97.5% VaR has $\alpha = 0.025$). Over-forecasting is not rejected: the test is one-sided (Remark 5: for
the VaR the other side uses the negated loss; testing an over-forecast of the ES is, in the paper's words, "very
challenging" and is not offered).

## The backtest e-statistics

An e-statistic is a function of the loss and the forecast whose expectation is at most 1 when the forecast is not an
under-forecast and more than 1 when it is (Definitions 1 and 2). The paper's two:

$$
e^{\mathrm{VaR}}(x, v) = \frac{\mathbf 1\{x > v\}}{\alpha}
\qquad \text{equation (1), Example 3: } \mathbb E\big[e^{\mathrm{VaR}}(L, v)\big]\le 1 \text{ for } v \ge \mathrm{VaR}_p(L), \tag{1}
$$

$$
e^{\mathrm{ES}}(x, r, v) = \frac{(x - v)_+}{\alpha\,(r - v)}
\qquad \text{equation (3), Theorem 1: } \mathbb E\big[e^{\mathrm{ES}}(L, r, \mathrm{VaR}_p(L))\big] = \frac{\mathrm{ES}_p(L) - \mathrm{VaR}_p(L)}{r - \mathrm{VaR}_p(L)} \le 1
\text{ for } r \ge \mathrm{ES}_p(L). \tag{3}
$$

Equation (3) comes from the Rockafellar and Uryasev representation $\mathrm{ES}_p = \min_z\{z + \mathbb E[(L - z)_+]/(1 - p)\}$ with
minimiser $\mathrm{VaR}_p$; the loss $z + (x - z)_+/(1 - p)$ makes $(\mathrm{ES}_p, \mathrm{VaR}_p)$ a Bayes pair. Functions:
`var_backtest_e_statistic(loss, var, alpha)` and `es_backtest_e_statistic(loss, var, es, alpha)`. Theorem 4 and Theorem 5
show that, among all one-sided e-statistics (for the ES: those increasing in the loss), these two are essentially the only
useful ones: every other is dominated by $1 - h + h\,e$ with a function $h \in [0, 1]$, which is the bet below. The paper's
conventions $0/0 = 1$ and $1/0 = \infty$ apply when $r = v$ (an ES forecast equal to the VaR forecast: the e-statistic is 1
on a day without a loss above the VaR and infinite on a day with one). An ES forecast below the VaR forecast is refused: the
paper sets $e^{\mathrm{ES}} = \infty$ there as "a case of no relevance since $\mathrm{ES} \ge \mathrm{VaR}$".

Proposition 1 and Example 6: $e^{\mathrm{ES}}$ stays an e-variable when both forecasts are above the truth with their
difference at least the true $\mathrm{ES} - \mathrm{VaR}$, for instance both multiplied by the same factor above 1, both
shifted by the same amount, or both taken at a higher level. It is not an e-variable when only the VaR is above the truth
(Example 5): an inflated VaR forecast can make the ES e-value large, which matters when the ES e-value of a bank that
over-reports its VaR is read.

## The e-process

Each day the tester bets a share $\lambda_t \in [0, \gamma]$ of its wealth on the e-statistic $X_t$, keeping the rest:

$$
M_0 = 1,\qquad M_t = \prod_{s=1}^{t}\big(1 - \lambda_s + \lambda_s X_s\big),\qquad \lambda_s \text{ a function of } (L_u, v_u, r_u)_{u<s},\, v_s, r_s,
\tag{7}
$$

where $X_s = e^{\mathrm{VaR}}(L_s, v_s)$ or $e^{\mathrm{ES}}(L_s, r_s, v_s)$. Under the null each factor has conditional mean
at most 1 whatever the bet, so $M_t$ is a non-negative supermartingale (equation (7) and Theorem 2) and, by Ville's
inequality,

$$
\mathbb P\Big(\sup_{t} M_t \ge \tfrac{1}{\alpha_{\mathrm{level}}}\Big) \le \alpha_{\mathrm{level}} \tag{Theorem 2}
$$

at any stopping time. The engine reports $\log M_t$ per day (`log_e`), the first day it reaches $\log(1/\alpha_{\mathrm{level}})$
(`rejection_time`) and the anytime-valid p-value $\min(1, 1/\sup_t M_t)$. `alpha_level` is the size of the monitor and defaults
to 0.05 (threshold 20); the paper reads thresholds 2, 5 and 10 as minor, substantial and strong evidence (Remark 3, Section 7),
which are `alpha_level` 0.5, 0.2 and 0.1: not a decisive rejection but an early warning, with a type I error guarantee of
50%, 20% and 10%. The first day has no past and bets nothing, so $M_1 = 1$; a day without a bet is a factor 1 even when its
e-statistic is infinite.

## The betting processes

The power of the test is the choice of $\lambda_t$. The paper's criterion is the growth rate of the log wealth, the Kelly
criterion; its oracle is $\lambda^{\mathrm{GRO}}_t = \arg\max_{\lambda \in [0, \gamma]} \mathbb E_{Q_t}[\log(1 - \lambda + \lambda e(L_t, r, v))]$,
unavailable because the law $Q_t$ of the loss is unknown. Three plug-ins estimate it from the past (Section 5.1), each on all
past days or on the last `window` days:

| Method | Sample the growth rate is averaged over | Paper |
|---|---|---|
| `"GREE"` | the e-statistics of the past days, each on its own day's forecast $e(L_s, r_s, v_s)$, $s < t$ | equation (13) |
| `"GREL"` | the past losses valued at the current forecast $e(L_s, r_t, v_t)$, $s < t$ | equation (14) |
| `"GREM"` (default) | the equal mixture of the wealth of the two: $M_t = (M_t^{\mathrm{GREE}} + M_t^{\mathrm{GREL}})/2$ | Section 5.1, Theorem 3 (iii) |

GREE and GREL are equal when the forecasts are constant (the paper, after (14)); GREE is the better when the e-statistics
are exchangeable and the forecasts say nothing about the next bet, GREL when the losses are exchangeable and the forecast is
informative (Theorem 3, Example 7). GREM is a valid e-process, because an average of two non-negative supermartingales is
one, and it is itself the product (7) of the betting process

$$
\lambda_t^{\mathrm{GREM}} = \frac{M_{t-1}^{\mathrm{GREE}}\,\lambda_t^{\mathrm{GREE}} + M_{t-1}^{\mathrm{GREL}}\,\lambda_t^{\mathrm{GREL}}}{M_{t-1}^{\mathrm{GREE}} + M_{t-1}^{\mathrm{GREL}}}
$$

(Vovk and Wang 2024a, Lemma 1, as quoted by the paper); the engine returns it in `betting` and the two wealth paths in
`component_log_e`. The paper recommends GREM "as a default choice for implementation" (Sections 5.2 and 8): it is asymptotically
optimal whenever either of the two is (Theorem 3 (iii)) and its performance is the more stable across cases.

Two solvers (`solver`) compute the bet of GREE and GREL. The closed form is the second-order Taylor expansion
$\log(1 + y) \approx y - y^2/2$ of the growth rate and its first-order condition, which the paper uses for all its simulations
and its data analysis:

$$
\lambda_t \;\approx\; 0 \,\vee\, \frac{\sum_{s<t}\big(e_s - 1\big)}{\sum_{s<t}\big(e_s - 1\big)^2} \,\wedge\, \gamma,
\qquad e_s = e(L_s, r_s, v_s)\ (\text{GREE}) \text{ or } e(L_s, r_t, v_t)\ (\text{GREL}), \tag{15}
$$

and, for the VaR on a hit series with $k_t$ non-hits among the $t - 1$ past days and $p = 1 - \alpha$, Appendix A gives
$\lambda_t \approx (1 - p)\,((t - 1)p - k_t) / ((t - 1)p^2 + (1 - 2p)k_t)$ (before the cap). The exact solver
(`solver="exact"`) maximises the empirical growth rate $\frac{1}{t-1}\sum_{s<t}\log(1 - \lambda + \lambda e_s)$ over
$[0, \gamma]$, which the paper (equations (13) and (14)) says "can be solved directly via convex programming": it is concave,
so the engine bisects its decreasing derivative (56 halvings). On a hit series it equals $(h - \alpha)/(1 - \alpha)$ with $h$
the share of hits in the sample, the Kelly fraction of a Bernoulli bet.

## What the paper leaves open and the engine fixes

| Choice | Fixed as | Why |
|---|---|---|
| The bet of the first day | 0 | the empirical sample is empty |
| A past sample with mean $e \le 1$ (and one with every $e = 1$) | bet 0 | Proposition 2 (i): the growth-rate optimum is positive only when the mean e-statistic is above 1; equation (15) is clipped at 0 and 0/0 is read as no bet |
| A past e-statistic equal to infinity (an ES forecast equal to the VaR forecast and a loss above it) | the bet is $\gamma$ | the empirical growth rate is infinite for every positive bet, so the optimum is not unique (Section 5.1 notes that the optimiser can be non-unique when the expectation is infinite); the cap is the bet that pays the most if the infinity returns |
| The cap $\gamma$ | $1/2$ | Remark 7: the cap keeps the e-process from collapsing to 0 after one zero e-statistic; the paper reports optimal bets typically below 0.1 for tail risk measures, so the default is harmless. It must lie in $(0, 1)$ |
| The window | all past days; `window=w` keeps the last $w$ | Remark 8 and Section 8 use 250 or 500 days to resist a forecaster who over-reports for a while and then under-reports; the window changes the bet only, so validity does not depend on it |
| The default method | GREM | Section 5.2 and Section 8 |
| An ES forecast below the VaR forecast | refused (`ValueError`) | the paper sets the e-statistic to infinity and calls the case irrelevant; an input in that case is an error of the caller, not evidence |
| The weights of GREM | one half each | the paper allows other weights when the tester has side information on which method works better (Section 5.1); the engine offers the equal weights only |

## Using it

```python
from riskvalidation.validation.sequential import e_backtest_es, e_backtest_var

# losses are positive; var and es are the forecasts made before each loss, at exceedance probability alpha
res = e_backtest_es(loss, var, es, alpha=0.025, alpha_level=0.2, window=500)
res.rejection_time          # the first day the e-process reached 5 (1-based), or None
res.log_e[-1]               # log of today's e-value
res.betting                 # the bets the process made
res.to_test_result()        # the engine's TestResult: red when it rejected, green otherwise
```

The same call for the VaR alone is `e_backtest_var(loss, var, alpha)`. The bets can be read without the process
(`betting_process(loss, var, alpha, es=es, method="GREE")`).

## Assumptions and what is refused

- The null is one-sided: an over-forecast is not rejected, and the supermartingale guarantee holds under the null as the
  paper states it, with no independence or stationarity. The test is conditional: the forecast of day $t$ must be made before $L_t$ is
  known, and the bet of day $t$ must not depend on $L_t$; the engine's bets use the days before $t$ and the forecast of day $t$.
- The ES test reads the ES forecast given the VaR forecast: with a VaR forecast above the truth and an ES forecast at the
  truth the e-statistic is not guaranteed to be an e-variable (Example 5, equation (11)); Proposition 1 gives the form of the
  null that holds for over-forecasts of both.
- A NaN or an infinite loss or forecast, arrays of different lengths or of more than one dimension, `alpha`, `alpha_level`
  or `gamma` outside $(0, 1)$, a `window` that is not a positive whole number, an unknown `method` or `solver` and an ES
  forecast below the VaR forecast raise `ValueError` naming the field. An empty series gives $n = 0$ and the not-evaluated
  form of `to_test_result()`; a series of one day gives $\log M = 0$.
- Cost: GREE with the closed form is linear in the number of days; GREL with forecasts that move is quadratic (computed in
  blocks), as is the exact solver.

## Evidence

The requirements' tests check each definition against the paper's text on hand-computed values, against loops written from
the equations (GREE, GREL, the window, the exact solver against a bounded scalar minimiser), the VaR e-process against
the reference function of the Contraste manuscript to $10^{-12}$, the expectation of each e-statistic by quadrature
(equal to 1 to $10^{-9}$ at the true VaR and ES) and by Monte Carlo, the mean of $M_T$ equal to 1 exactly by enumeration
of all hit sequences of a Bernoulli law and of all loss sequences of a three-point law with $\mathrm{VaR}_{0.9} = 1$ and
$\mathrm{ES}_{0.9} = 3$, and the size and the detection by simulation (`tests/test_sequential_e_backtest.py`).

A run for the page (1,000 series of 1,000 days, standard normal losses, the true VaR and ES of $\alpha = 0.025$ scaled by a
factor, `GREM` with the closed form, the draws `numpy.random.default_rng(SeedSequence([i, 20261116]))` for series $i$) gives the
share of series whose e-process ever reached each threshold and, in brackets, the median first day among those that did:

| Forecast | Test | threshold 2 | threshold 5 | threshold 10 | threshold 20 |
|---|---|---|---|---|---|
| true values (null) | VaR | 0.172 (129) | 0.037 (211) | 0.010 (524) | 0.006 (323) |
| true values (null) | ES | 0.134 (135) | 0.029 (452) | 0.011 (414) | 0.004 (480) |
| 90% of the true values | VaR | 0.857 (209) | 0.697 (401) | 0.556 (511) | 0.440 (589) |
| 90% of the true values | ES | 0.866 (258) | 0.683 (475) | 0.545 (574) | 0.394 (628) |
| 80% of the true values | VaR | 0.999 (81) | 0.999 (158) | 0.998 (219) | 0.992 (273) |
| 80% of the true values | ES | 1.000 (92) | 0.999 (169) | 0.998 (228) | 0.997 (294) |

The null shares are far below the guaranteed 50%, 20%, 10% and 5% (Remark 3: the empirical type I error is usually much
smaller than the bound); a 10% under-forecast of both is detected at threshold 5 in about 70% of the series within four
years, a 20% one in all but one series, at a median of 158 days. With constant forecasts GREE, GREL and GREM give the same VaR paths here.
The `sizepower` test repeats the size study at 4,000 series and also with forecasts that vary by day.

## Sources

- Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991v7: Definitions 1 and 2, equations (1) and (3), Theorems 1,
  2, 3, 4 and 5, Propositions 1 and 2, Remarks 3, 5, 7 and 8, Section 5.1 (GREE, GREL, GREM), equation (15), Sections 7
  and 8, Appendix A. The version read (6 November 2026) corrects the statement of Proposition 6, which the engine does not use.
- Ville (1939), Etude critique de la notion de collectif, Gauthier-Villars (the maximal inequality).
- Ramdas, Grunwald, Vovk and Shafer (2023), Game-theoretic statistics and safe anytime-valid inference, Statistical Science
  38(4):576-601, arXiv 2210.01948.
- Rockafellar and Uryasev (2002), Conditional value-at-risk for general loss distributions, Journal of Banking and Finance
  26(7):1443-1471 (as cited by the paper's proof of Theorem 1; not read, UNVERIFIED as a primary text).
- Vovk and Wang (2024a), Merging sequential e-values via martingales, Electronic Journal of Statistics 18(1):1185-1205
  (Lemma 1 on the betting process of an average of e-processes, as quoted by the paper; not read, UNVERIFIED).
- The reference function against which the VaR case is tested: `wwz_log_e`, CAOS_Contraste, `manuscripts/e-process-sequential-monitors/experiments/common.py`.
