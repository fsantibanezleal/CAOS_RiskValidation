- U8 market risk (RV-801 to RV-830): `riskvalidation.market` with the sign convention (`to_loss_numbers`, `var_es_from_sample`,
  `var_es_from_distribution`), historical, age-weighted, filtered and volatility-updated simulation, delta-normal, Cornish-Fisher
  and EWMA, GARCH/GJR/EGARCH through `arch` (optional extra `market`), conditional EVT, CAViaR with the Taylor ES extension
  and the dynamic quantile statistic, the GAS one-factor, two-factor and hybrid FZ0 models, the FZ0 and tick losses,
  Diebold-Mariano and the Model Confidence Set, and the FRTB statistics (MAR32.9 table, zones, multipliers, desk limits, PLA
  Spearman, exact KS step count and zones).
- U8 back-tests (`riskvalidation.validation.market`): Kupiec, Christoffersen independence and conditional coverage, the
  duration test, the Basel traffic light, Acerbi-Szekely, Du-Escanciano unconditional and conditional, McNeil-Frey, the
  dynamic quantile test, the PLA test and the desk back-test, each a `TestResult` with its size checked by simulation.
- Documentation under `docs/market/` (equations, goldens, sources, decisions D-1 to D-9, limits) and the feature SDD
  `docs/design/features/u8-market/requirements.md`.
- U8-E review fixes: the age-weighted HS now has the two knots per return of Boudoukh, Richardson and Whitelaw (1998) and
  reproduces their worked example (VaR 2.63% and 2.34%); the dynamic quantile test takes a Monte Carlo p-value (`n_mc`, `seed`;
  size between 0.03 and 0.07 at (0.01, 250), (0.025, 250) and (0.01, 750)); the CI installs `.[dev,market]`; the ES-CAViaR is
  declared two-step (decision D-10, `caviar_es_fit`); the duration test censors the first spell only when the sample does not
  start with a hit and reports a capped shape; the PLA zone is decided in exact arithmetic (`frtb.compare_spearman`,
  `frtb.pla_zone_exact`).
