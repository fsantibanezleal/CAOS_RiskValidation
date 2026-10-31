# Market risk

`riskvalidation.market` estimates VaR and ES (historical simulation, delta-normal, Cornish-Fisher, EWMA, GARCH, conditional
EVT, CAViaR, GAS-FZ), scores the forecasts (FZ0 and tick losses, Diebold-Mariano, the Model Confidence Set) and holds the FRTB
statistics (MAR32 traffic light, desk limits, the P&L attribution test); `riskvalidation.validation.market` back-tests them
(Kupiec, Christoffersen, duration, Basel zone, Acerbi-Szekely, Du-Escanciano, McNeil-Frey, dynamic quantile, PLA, desk).
The engine computes and states no compliance claim: the alphas of a light, the thresholds and the replication counts are
policy parameters the caller supplies, or Contraste internal gates.

**Return convention in the engine, positive loss numbers in the documents** (dossier 05 A.0). A return `y` is a decimal of the
portfolio value; at a tail level `alpha` (0.01 or 0.025) the engine stores `VaR = v = F^-1(alpha)` and `ES = e = (1/alpha)
int_0^alpha F^-1(u) du`, both negative in practice, with `e <= v`. The case documents store positive loss numbers (`loss = -y`);
the conversion is made once, at the reporting layer, with `conventions.to_loss_numbers`. The functions of
`riskvalidation.validation.market` that take losses (the ES tests, the desk test, the exceedance flags) take positive loss
numbers; the functions of `riskvalidation.market` take and return the return convention, except where a function says otherwise
(`evt.fit_gpd` and `evt.evt_var_es` work on standardised loss residuals, as the formulas of McNeil and Frey do).

| Page | Contents |
|---|---|
| [01. Historical simulation](01_historical-simulation.md) | the sign convention functions, VaR and ES from a sample, HS, age-weighted HS, filtered and volatility-updated HS |
| [02. Parametric and EWMA](02_parametric-and-ewma.md) | delta-normal, Cornish-Fisher and its validity domain, EWMA |
| [03. GARCH and EVT](03_garch-and-evt.md) | GARCH, GJR and EGARCH through `arch`, the innovation quantiles, the peaks-over-threshold tail and the conditional EVT forecast |
| [04. CAViaR and GAS-FZ](04_caviar-and-gas-fz.md) | the CAViaR recursions, the ES extension, the dynamic quantile statistic, the GAS one-factor, two-factor and hybrid models |
| [05. Scoring and comparison](05_scoring-and-comparison.md) | the FZ0 loss and its reduction, the tick loss, Diebold-Mariano, the Model Confidence Set |
| [06. FRTB tests](06_frtb-tests.md) | the binomial table of MAR32.9, the zones and multipliers, the desk limits, the PLA Spearman, KS and zones |
| [07. The back-tests](07_backtests.md) | each back-test as a `TestResult`: statistic, p-value, light, edge cases, the size of each on correct forecasts |
| [08. Conformal recalibration](08_conformal-recalibration.md) | the adaptive conformal recalibration of a VaR model's level (Gibbs and Candes), its ES at the adapted level, wrapped against unwrapped on the C11 windows, and the verdict |
| [09. The PLA complement](09_pla-complement.md) | the scale-and-level complement to the PLA test: Mincer-Zarnowitz Wald, absolute and rolling Spearman, a calibrated decision, its measured power and size |

## Sources read for this unit, and what was not

| Source | Status |
|---|---|
| Dossier 05 sections A.0 to A.2 and B.1 to B.6, dossier 01 section D.4 | transcribed |
| Boudoukh, Richardson and Whitelaw (1998), author PDF https://pages.stern.nyu.edu/~rwhitela/papers/hybrid%20risk98.pdf | read in full text: the weights, the half-weight rule and the linear interpolation (section 3); it gives no ES rule |
| Patton, Ziegel and Chen (2019), arXiv 1707.05108 https://arxiv.org/abs/1707.05108 | read in full text: eqs 9 to 20 and 27, Table 7 |
| Taylor (2019), JBES 37(1):121-133, doi:10.1080/07350015.2017.1281815 | primary text not retrieved; the ES recursion is that restated in Section 3.2 of arXiv 2603.02357 (https://arxiv.org/abs/2603.02357) |
| Engle and Manganelli (2004), NBER w7341 https://www.nber.org/papers/w7341.pdf | the PDF is a scan without a text layer; the `(y)^+`, `(y)^-` convention is checked against the indicator form restated in arXiv 2603.02357 eqs 3 and 4; the DQ regressor set is the contract's and its degrees of freedom are the rank of the regressors (UNVERIFIED against the paper) |
| Christoffersen and Pelletier (2004), doi:10.1093/jjfinec/nbh004 | not read; the Weibull hazard is the standard one and the size of the implemented test is checked by simulation |
| Acerbi and Szekely (2014), Risk, December | gated, not retrieved; the reference point -0.70 and -1.8 is verified only through arXiv 1709.01337 section 6 |
| Zangari (1996) for Cornish-Fisher VaR, McNeil and Frey (2000) for the threshold fraction | not verified (the dossier marks both UNVERIFIED) |

## Decisions

D-1 to D-8 are those of the contract `engine-market-api` 1.0.0, D-9 and D-10 are this unit's (D-10 written in the contract after the review) (the pages that apply them name them).
D-1 the exact binomial p-value (07); D-2 the Bartlett HAC kernel (05); D-3 the empirical VaR and ES of a sample (01); D-4
filtered HS without resampling (01); D-5 the one-sided Du-Escanciano p-value (07); D-6 the Cornish-Fisher validity domain (02);
D-7 the independence test at no exceedance (07); D-8 the minimum of 2 exceedances of the duration and McNeil-Frey tests (07);
D-9 the ES of the age-weighted HS (01); D-10 the two-step estimator of the ES-CAViaR (04), declared in the contract after the review.

## Limits of what the engine claims

A green light is a statistic inside a policy band, never an approval. The engine does not decide which model is right; it
returns the evidence. The estimators are one-step-ahead forecasts from data dated strictly before the forecast day; keeping the
estimation window strictly before the day is the caller's duty, which the functions make possible (they receive no data of the
day they forecast) and the document producer checks.
