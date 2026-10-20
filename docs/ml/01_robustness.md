# 01. Robustness and resilience to shift

Module `riskvalidation.ml.robustness`. Sources: dossier 06 C.7 l.1009-1015 (PiML "Robustness" and "Resilience") and I.1
l.1189-1190; SR 11-7 sensitivity analysis, the historical reference text now superseded by SR 26-2 (06 A.1 l.19-37; "unexpectedly large changes
in outputs in response to small changes in inputs can indicate an unstable model", 06 l.1011); SS1/23 operating boundaries (footnote 7).

## Perturbation robustness

Noise of scale $\epsilon$ proportional to each feature's standard deviation is added to the **evaluation** features only (never
to the training slice) and performance (AUC, Brier) is read against $\epsilon$; a finding if the degradation exceeds a
tolerance at a pre-set $\epsilon$ (06 C.7 l.1009-1012).

$$\tilde x_{ij}=x_{ij}+\epsilon\,s_j\,z_{ij},\quad z_{ij}\sim N(0,1),\qquad
\Delta_{\mathrm{AUC}}(\epsilon)=\mathrm{AUC}(\epsilon)-\mathrm{AUC}(0),\qquad
\text{finding}\iff-\Delta_{\mathrm{AUC}}(\epsilon^*)>\tau$$

Symbols: $s_j$ the sample standard deviation (ddof 1) of column $j$ of the evaluation features unless supplied; $\tau$ the
tolerance in AUC points; $\epsilon^*$ the preset scale, a point of the grid.

- `perturb_features(X, eps, seed, sd=None)` returns a new array; `eps = 0` returns `X` unchanged; the noise of column $j$ has
  standard deviation `eps * sd[j]` (checked within 4 Monte Carlo standard errors on 100,000 rows).
- `robustness_curve(predict, X, y, grid, seed)` perturbs with the **same standard normal draws at every scale** (the noise is a
  function of the seed alone), so the perturbed samples of a grid are nested; the curve always starts at scale 0 (prepended when
  the grid lacks it) and carries AUC, Brier and the AUC delta against scale 0. `predict` must return probabilities of shape `(n,)`
  in [0, 1]; both classes must be present. A constant predictor has AUC 0.5 at every scale; a predictor that ignores its input has
  a zero delta (identities tested).
- `robustness_finding(curve, tolerance, preset_eps)` is true exactly when the loss exceeds the tolerance; a loss equal to the
  tolerance is not a finding; the loss is rounded to 12 decimals first, so a loss that is the tolerance in decimals (0.80 - 0.78
  is `0.020000000000000018` in floats) is green. `preset_eps` must be a grid point.

## The shift score

`shift_score_rank(X_dev, X_eval, seed)` returns, for each evaluation row, the probability that a resemblance classifier gives to
"evaluation" (06 C.7 l.1013-1015: "a shift score"; the resemblance model is the construction of ING probatus
`sample_similarity`, 06 C.7 l.1048-1051). The classifier is the engine core's default logistic fit (NumPy and SciPy only):
ridge-penalised Newton on standardised features with an unpenalised intercept and balanced class weights $n/(2n_c)$, so two
samples that agree score 0.5 on average whatever their sizes.

$$\hat\beta=\arg\min_\beta\ \sum_i w_i\big[\log(1+e^{z_i})-y_i z_i\big]+\tfrac{\lambda}{2}\lVert\beta_{1:p}\rVert^2,\qquad
z_i=\beta_0+\beta_{1:p}^{\top}x_i,\quad \lambda=1$$

Two conventions are fixed here and are not regulatory values: the ridge strength $\lambda = 1$ (the default of scikit-learn's
`C = 1`; no dossier names one) and the 5 cross-fitting folds. The probability of each row comes from folds that did not train on
it; **rows with an identical feature vector are never split across folds** (a row whose exact twin sits in the other sample would
otherwise be scored by a model that saw the twin with the opposite label, which pushes the AUC of two literally identical samples
to about 0.455 instead of 0.5). Deterministic for a seed.

## Worst subsample and out-of-time windows

`worst_subsample_performance(predict, X, y, score, share)` returns AUC and Brier on the $m=\lceil \text{share}\cdot n\rceil$ rows
with the highest shift score (ties broken by row order) beside the full-sample metrics (06 C.7 l.1013-1015; the worst 10 percent
is the dossier's example). $m$ is computed on the exact decimal `share` (`Fraction(str(share))`: `0.1 * 30` is
`3.0000000000000004` in floats and would give 4). `share = 1` returns the full-sample metrics exactly. A subsample that holds a
single class has no AUC: the function raises.

`oot_window_performance(predict, X, y, window)` returns the AUC of each integer window label (keys are the labels as strings) and
the key `omitted`: the number of windows left out because they hold a single class (a window with fewer than two classes never
gets a number).

## The development-only reference of the shift test

Under no shift the evaluation rows with the highest shift score are a region of the feature space on which the model's AUC is lower
by range restriction, so the raw drop $\text{AUC}_{\text{full}}-\text{AUC}_{\text{worst}}$ is positive under the null (finding F4:
the first design, a drop against fixed cut points, had a size of 0.1055). A pooled-permutation reference was tried next and
rejected: it put the shifted evaluation rows into its own null and its power against the covariate-shift plant stayed at 0.28 at any
sample size. The light now reads the observed drop against a development-only reference (`shift_development_reference`, decision G4
of unit U14 as amended; a convention: R = `shift_permutations`, default 199):

1. R times, draw without replacement a random subset of the development rows of the evaluation set's size as the pseudo-evaluation
   set and take the other development rows as the pseudo-development set;
2. refit the shift score exactly as `shift_score_rank` does (cross-fitted, pseudo-development class 0, pseudo-evaluation class 1) and
   compute the drop on the pseudo-evaluation rows with `worst_subsample_performance` at the same share (this needs the development
   labels);
3. $p=\dfrac{1+\#\{\text{null drops}\ge\text{observed drop}\}}{R+1}$.

Under no shift the evaluation rows are exchangeable with the development rows, so the null is valid and holds no shifted row
(Phipson and Smyth 2010, doi:10.2202/1544-6115.1585). A draw whose worst subsample holds a single class has no AUC and is left out
(`n_undefined`); the observed subsample must have an AUC or the function raises. When the development set has fewer than twice the
evaluation rows, or without development labels (`y_dev=None`), the reference carries no null drops and the validation test is
`not_evaluated` with that reason. The light (`ml.shift_resilience`) reads the p-value with the p-value cut points 0.05 and 0.01; the
observed drop is the statistic and the null drops are summarised in `extras` (`null_mean`, `null_p95`). Deterministic for a seed.

## What these checks do not claim

- A robustness curve says how this predictor reacts to noise proportional to the column spread; it does not say the inputs carry
  that noise in production. The tolerance and the preset scale are conventions (the named parameter `perturbation_tolerance`, 0.02
  AUC points, decision V6 of unit U14).
- The worst subsample by shift score is a region of the feature space, not a random subsample: a model is judged on a narrower
  range of its own score there, so even without any shift its AUC there is lower than on the whole sample (page 08, finding F4);
  the development-only reference above is what the light reads, not the raw drop.
- The shift score ranks rows by how far they sit from the development sample; it does not state that the shift matters.
- The engine computes and makes no compliance claim.

## Sources

- Dossier 06 (model risk management and validation) C.7 and I.1; PiML robustness and resilience tests.
- SR 11-7 (Federal Reserve, 2011) on sensitivity analysis, the historical reference text that dossier 06 A.1 l.19-37 records as
  superseded in the US by SR 26-2 / OCC 2026-13 / FDIC FIL-15-2026 (2026-04-17); SS1/23 (PRA, 2023) footnote 7 on operating boundaries.
- ING probatus `sample_similarity`, https://github.com/ing-bank/probatus (dossier 06 D l.1100 and l.1434).
- scikit-learn 1.9.1 `LogisticRegression` (default `C = 1.0`, checked in the installed package), the source of the ridge strength.

## Limits

The AUC and Brier are computed on the rows supplied; the classes must be present in them. `perturb_features` perturbs every column
with a positive standard deviation; a constant column is left untouched. The engine imports no learning library: the model is
whatever the predict callable computes. UNVERIFIED: the dossier gives no value for the shift-score classifier, the ridge strength
or the folds; the choices above are conventions of the engine core, not findings of the dossiers.
