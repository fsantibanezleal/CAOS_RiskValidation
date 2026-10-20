# 04. The overfit gap and the weak spots

Module `riskvalidation.ml.generalisation`. Sources: dossier 06 C.7 l.1016-1018 (PiML "Overfit" and "WeakSpot") and I.1
l.1187-1188; ECB guide to internal models paragraph 43(a) (overfitting); EBA/REP/2023/28 recommendation (i).

## The overfit gap

Train versus test performance gap (06 C.7 l.1016-1018: "overall and per slice"; the engine takes the metrics the pipeline
computed, so the per-slice form is one call per slice):

$$\text{gap}=\begin{cases}M_{\text{train}}-M_{\text{test}}&\text{higher is better (AUC)}\\ M_{\text{test}}-M_{\text{train}}&\text{otherwise (Brier)}\end{cases},\qquad
\text{flag}\iff\text{gap}>\tau$$

`overfit_gap(metric_train, metric_test, tolerance, higher_is_better=True)` returns `OverfitResult(gap, tolerance, flag)`; the flag
equals `gap > tolerance` (a gap equal to the tolerance is not flagged; the dataclass refuses a flag that contradicts it). The
tolerance is the named parameter `overfit_tolerance` (0.05, decision V6 of unit U14), a convention.

## Weak spots

Slices found by binning features (or by a shallow tree on residuals) where the test error exceeds a threshold multiple of the
average (06 C.7 l.1016-1018). `weak_spots(error, bins, k, min_n=30)` takes the per-row loss (Brier terms) and the integer slice
label of each row and returns the slices that hold at least `min_n` rows and whose mean error exceeds `k` times the overall mean:

$$\text{ratio}_s=\frac{\bar e_s}{\bar e},\qquad s\ \text{reported}\iff n_s\ge n_{\min}\ \text{and}\ \text{ratio}_s>k$$

sorted by decreasing ratio (ties by label). A slice below `min_n` is never reported however large its error (the dossier gives no
value for `min_n`: 30 is a convention of the contract); an overall mean of zero leaves nothing to exceed. A planted slice with
three times the error of the rest (overall mean 1.4, ratio 2.14) is returned for `k = 2` and absent for `k = 4` (tested).

## What these checks do not claim

- A small gap does not show the model generalises to a different period or population: the out-of-time and shift tests (page 01)
  answer that.
- The slices are the ones the caller defines; a weak spot found in one slicing says nothing about another.
- The engine computes and makes no compliance claim.

## Sources

- Dossier 06 C.7 and I.1 l.1187-1188; ECB guide paragraph 43(a); EBA/REP/2023/28.
- PiML (the Overfit and WeakSpot tests), https://github.com/SelfExplainML/PiML-Toolbox (dossier 06 l.1431).

## Limits

The multiple `k` and `min_n` are conventions. The error array is the caller's loss per row; for the Brier score it is `(p - y)^2`.
