# 08. Adaptive conformal recalibration of VaR and ES (BS-16)

Module `riskvalidation.market.conformal`. Sources: Gibbs and Candes, "Adaptive Conformal Inference Under Distribution Shift",
NeurIPS 34, 2021, arXiv 2106.00170; Christoffersen, International Economic Review 39(4):841-862, 1998,
doi:10.2307/2527341 (independence and conditional coverage); Acerbi and Szekely, Risk, December 2014 (Z2); Fissler and Ziegel
(FZ0, see page 05). Conformal layers on VaR forecasts exist in working papers; what this module adds to the engine is the wrapper
as a tested, model-agnostic function and its comparison, wrapped against unwrapped, on the engine's own back-tests and scores.
Return convention throughout (VaR and ES negative decimals, `es <= var`).

## The method

A model is any callable `forecaster(t, level) -> (var, es)` that uses only the returns before day `t` (for HS-250,
`historical_forecaster(returns, 250)`). The wrapper keeps a working tail level `a_t`, starting at `alpha`, asks the model for its
forecast at `a_t`, and moves the level after the day:

$$a_{t+1}=\operatorname{clip}\Big(a_t+\gamma\,(\alpha-\mathrm{err}_t),\ a_{\min},\ a_{\max}\Big),\qquad \mathrm{err}_t=\mathbb 1\{y_t<\mathrm{VaR}_t(a_t)\}$$

A hit lowers the level by $\gamma(1-\alpha)$ (a deeper VaR), a quiet day raises it by $\gamma\alpha$. The wrapped ES is the model's own
tail mean at $a_t$. Gibbs and Candes prove, for the unclipped recursion and any data order, $\big|\overline{\mathrm{err}}-\alpha\big|\le
(\max(a_1,1-a_1)+\gamma)/(\gamma T)$: a long-run coverage guarantee, with nothing about independence of the hits.

Public names: `aci_update(level, hit, alpha, gamma, level_min, level_max)`, `aci_var_es(returns, forecaster, alpha, gamma, start,
level_min, level_max) -> AciResult`, `historical_forecaster(returns, window)`, `forecast_scores(returns, var, es, alpha) ->
ConformalScores` (Kupiec, independence and conditional-coverage p-values from `validation.market`, mean FZ0 from
`market.scoring`, and $Z_2=1-\sum_t y_t\mathbb 1\{\mathrm{hit}_t\}/(\mathrm{ES}_t\,T\alpha)$), `aci_compare(...) -> AciComparison`
(wrapped, unwrapped and `fz0_change = (FZ0_w - FZ0_u)/|FZ0_u|`, positive is worse). Defaults: `gamma = 0.005`, band `[0.001, 0.25]`.
A p-value the tests cannot evaluate (for example independence with no hit) is `None` and `reasons` says why.

Decisions: D-1, the working level is clipped to the band after each update (the paper lets it leave (0, 1), where the forecast set is
empty or the whole line); the guarantee then holds only while the level stays in the band, and `n_clipped` counts the days it did
not. D-2, no conformal guarantee is claimed for the ES. D-3, a rounding excess `es > var` of one unit in the last place (when
$\alpha N<1$ in HS the two coincide) is set to `es = var`; a larger excess or a non-finite forecast raises `ValueError`.

## Evaluation (hypothesis of the plan, BS-16)

Hypothesis: HS-250, GARCH-t and FHS wrapped pass (Kupiec and conditional coverage, p at least 0.05) in at least one more of the
three stress windows than unwrapped, without mean FZ0 worsening by more than 5 percent. Refuted if the wrapper fails the
independence test in equity-2007-09. `gamma = 0.005` was fixed before any run; nothing was tuned.
Script: `scripts/e16_aci_evaluation.py`.

### Real C11 windows (offline run, one path per window)

Portfolio returns of the six C11 windows (the Contraste pipeline's readers and window definitions), forecast days 504, 104, 249,
251, 64 and 750. HS-250 is the full rolling model; GARCH-t and FHS refit a GARCH(1,1)-t on the last 1,000 returns every 5
forecast days (the Contraste bake refits daily) with a plain filter recursion, so they are close to, not identical with, the bake's
rungs R5 and R6. The French equity series is derived-only: the returns were read from the device data root and only these
aggregates are committed. "Pass" = Kupiec and conditional coverage both at least 0.05; shown unwrapped / wrapped.

| Window | Model @ level | Hit rate | Kupiec p | Independence p | CC p | Pass | FZ0 change |
|---|---|---|---|---|---|---|---|
| equity-2007-09 | HS-250 @ 1% | 0.038 / 0.022 | 0.000 / 0.021 | 0.222 / 0.483 | 0.000 / 0.055 | no / no | -17.2% |
| equity-2007-09 | HS-250 @ 2.5% | 0.071 / 0.034 | 0.000 / 0.233 | 0.371 / 0.275 | 0.000 / 0.271 | no / yes | -14.3% |
| equity-2007-09 | GARCH-t @ 1% | 0.032 / 0.012 | 0.000 / 0.677 | 0.305 / 0.703 | 0.000 / 0.853 | no / yes | -4.1% |
| equity-2007-09 | GARCH-t @ 2.5% | 0.062 / 0.028 | 0.000 / 0.695 | 0.044 / 0.371 | 0.000 / 0.620 | no / yes | -3.6% |
| equity-2007-09 | FHS @ 1% | 0.020 / 0.012 | 0.050 / 0.677 | 0.524 / 0.703 | 0.120 / 0.853 | yes / yes | -2.1% |
| equity-2007-09 | FHS @ 2.5% | 0.046 / 0.028 | 0.008 / 0.695 | 0.138 / 0.371 | 0.010 / 0.620 | no / yes | -0.7% |
| equity-2020 | HS-250 @ 1% | 0.077 / 0.058 | 0.000 / 0.001 | 0.115 / 0.327 | 0.000 / 0.002 | no / no | -168.7% |
| equity-2020 | HS-250 @ 2.5% | 0.125 / 0.067 | 0.000 / 0.022 | 0.754 / 0.468 | 0.000 / 0.055 | no / no | -631.7% |
| equity-2020 | GARCH-t @ 1% | 0.029 / 0.019 | 0.116 / 0.401 | 0.671 / 0.778 | 0.265 / 0.676 | yes / yes | +2.7% |
| equity-2020 | GARCH-t @ 2.5% | 0.067 / 0.048 | 0.022 / 0.180 | 0.468 / 0.209 | 0.055 / 0.185 | no / yes | -2.3% |
| equity-2020 | FHS @ 1% | 0.019 / 0.019 | 0.401 / 0.401 | 0.778 / 0.778 | 0.676 / 0.676 | yes / yes | +3.6% |
| equity-2020 | FHS @ 2.5% | 0.038 / 0.038 | 0.415 / 0.415 | 0.570 / 0.570 | 0.610 / 0.610 | yes / yes | +3.0% |
| rates-2022 | HS-250 @ 1% | 0.032 / 0.016 | 0.005 / 0.377 | 0.495 / 0.754 | 0.016 / 0.644 | no / yes | -5.8% |
| rates-2022 | HS-250 @ 2.5% | 0.056 / 0.032 | 0.007 / 0.490 | 0.213 / 0.495 | 0.011 / 0.624 | no / yes | -3.3% |
| rates-2022 | GARCH-t @ 1% | 0.024 / 0.016 | 0.058 / 0.377 | 0.619 / 0.754 | 0.147 / 0.644 | yes / yes | -0.2% |
| rates-2022 | GARCH-t @ 2.5% | 0.056 / 0.032 | 0.007 / 0.490 | 0.213 / 0.495 | 0.011 / 0.624 | no / yes | -0.3% |
| rates-2022 | FHS @ 1% | 0.020 / 0.020 | 0.160 / 0.160 | 0.685 / 0.685 | 0.343 / 0.343 | yes / yes | +0.5% |
| rates-2022 | FHS @ 2.5% | 0.048 / 0.032 | 0.037 / 0.490 | 0.291 / 0.495 | 0.065 / 0.624 | no / yes | +0.4% |

Windows outside the plan's three (full rows in the output JSON): fx-2008, calm-2004-06 and mixed-2023 change no verdict (every
model that passes unwrapped passes wrapped; the HS-250 rows of fx-2008 gain a pass at both levels; in mixed-2023 no hit occurs at 1%
and the independence test is not evaluated). The FZ0 change of HS-250 in equity-2020 is large and negative (an improvement) because
the unwrapped HS-250 forecasts a VaR near the calm past on a sudden crash.

Verdict on the real windows (stress windows equity-2007-09, equity-2020, rates-2022):

| Model | Extra windows passed, level 1% | Extra windows passed, level 2.5% | Worst FZ0 change | Holds |
|---|---|---|---|---|
| HS-250 | +1 (rates-2022) | +2 (equity-2007-09, rates-2022) | -3.3% (never worse) | yes, both levels |
| GARCH-t | +1 (equity-2007-09) | +3 (all three) | +2.7% | yes, both levels |
| FHS | 0 | +2 (equity-2007-09, rates-2022) | +3.6% | at 2.5% only; at 1% no gain (FHS already passes) |

The refutation condition does not occur: in equity-2007-09 the wrapped independence p-values are 0.275 to 0.703 for all six rows
(at least 0.05). The unwrapped ones were also at least 0.05 in five of six rows (GARCH-t at 2.5%: 0.044), so the independence
test did not discriminate here. Limits of this verdict: one realised path per window, windows of 104 and 64 days with 1 to 8 hits,
asymptotic p-values (Kupiec's small-count caveat applies at 1%), and the wrapped HS-250 still fails Kupiec at 1% in equity-2007-09
(0.021) and both levels in equity-2020. The level reached its lower band edge on up to 8 days (`n_clipped`).

### Generators of known truth (100 replications per scenario)

True process: Student-t(5) GARCH(1,1) (omega 2e-6, alpha 0.08, beta 0.90) after a calm history of 1,000 days, times a volatility
multiplier in the window: `crisis-persistent` (500 days, ramp to 3x then 1.5x), `crash-decay` (100 days, jump to 4x, half-life 30
days), `drift-up` (250 days, to 1.8x). HS-250 rolling; GARCH-t and FHS with parameters fitted once on the calm history (frozen
filter; a per-day refit is not affordable at this size). 300 windows, seed 20261028. Pass rate = share of replications with Kupiec and CC at
least 0.05, unwrapped / wrapped; "indep" = share with independence p at least 0.05.

| Scenario | Model @ level | Pass rate | Indep rate | Mean hit rate | Median FZ0 change |
|---|---|---|---|---|---|
| crisis-persistent | HS-250 @ 1% | 0.18 / 0.87 | 0.89 / 0.95 | 0.024 / 0.014 | -7.4% |
| crisis-persistent | HS-250 @ 2.5% | 0.11 / 0.95 | 0.83 / 0.92 | 0.051 / 0.027 | -6.1% |
| crisis-persistent | GARCH-t @ 1% | 0.44 / 0.97 | 0.99 / 0.97 | 0.020 / 0.014 | -0.9% |
| crisis-persistent | GARCH-t @ 2.5% | 0.31 / 0.99 | 0.98 / 1.00 | 0.044 / 0.030 | -0.7% |
| crisis-persistent | FHS @ 1% | 0.63 / 1.00 | 0.99 / 0.97 | 0.017 / 0.013 | +0.4% |
| crisis-persistent | FHS @ 2.5% | 0.53 / 1.00 | 0.98 / 0.99 | 0.038 / 0.028 | +0.2% |
| crash-decay | HS-250 @ 1% | 0.21 / 0.60 | 0.89 / 0.91 | 0.047 / 0.028 | -30.3% |
| crash-decay | HS-250 @ 2.5% | 0.20 / 0.73 | 0.87 / 0.86 | 0.088 / 0.054 | -15.2% |
| crash-decay | GARCH-t @ 1% | 0.61 / 0.87 | 0.81 / 0.90 | 0.023 / 0.017 | +1.3% |
| crash-decay | GARCH-t @ 2.5% | 0.77 / 0.94 | 0.97 / 0.96 | 0.044 / 0.036 | +0.4% |
| crash-decay | FHS @ 1% | 0.63 / 0.87 | 0.82 / 0.88 | 0.021 / 0.016 | +2.0% |
| crash-decay | FHS @ 2.5% | 0.83 / 0.94 | 0.95 / 0.95 | 0.041 / 0.035 | +0.4% |
| drift-up | HS-250 @ 1% | 0.58 / 0.95 | 0.92 / 0.95 | 0.024 / 0.017 | -2.4% |
| drift-up | HS-250 @ 2.5% | 0.46 / 0.92 | 0.89 / 0.94 | 0.049 / 0.034 | -2.2% |
| drift-up | GARCH-t @ 1% | 0.84 / 0.99 | 0.98 / 0.97 | 0.017 / 0.014 | +1.0% |
| drift-up | GARCH-t @ 2.5% | 0.71 / 0.99 | 0.98 / 1.00 | 0.038 / 0.032 | +0.2% |
| drift-up | FHS @ 1% | 0.91 / 1.00 | 0.98 / 0.99 | 0.016 / 0.014 | +1.0% |
| drift-up | FHS @ 2.5% | 0.78 / 0.99 | 0.98 / 0.99 | 0.037 / 0.031 | +0.4% |

In all 18 rows the wrapped pass rate is higher and the median FZ0 change is at most +2.0 percent. The pass rate of the wrapper is
below 0.9 in `crash-decay` for HS-250 (0.60 and 0.73): a 100-day window dominated by a transient is the case the level shift
repairs least. Independence is not harmed in any row (the share passing falls by at most 0.02).

### Reading

The hypothesis holds on the real windows for HS-250 and GARCH-t at both levels and for FHS at 2.5 percent, and in the generator
study in every scenario. It does not hold for FHS at 1 percent on the real windows (no window to gain: it already passes
equity-2007-09, equity-2020 and rates-2022 at 1 percent). The wrapper repairs coverage, not clustering, and it does not make a
model right: the engine's HS-250 still fails equity-2020.

## Limits

A shorter window means fewer hits, so the p-values are wide; a level pinned at the band edge is reported in `n_clipped`, not hidden;
the guarantee is on the long-run frequency of hits, not on the ES; the forecaster must not read day `t` or later (the wrapper cannot
check this; `tests/market/test_conformal.py::test_no_look_ahead_in_the_levels` tests the wrapper's own part); `gamma` is a free
parameter fixed here at the Gibbs-Candes default and not tuned on these windows.

Reproduce: `python scripts/e16_aci_evaluation.py --reps 100 --out generator.json` (about 35 minutes) and, with the windows file in the
device data root, `python scripts/e16_aci_evaluation.py --source npz windows.npz --refit-every 5 --out real.json` (about 6 minutes).
