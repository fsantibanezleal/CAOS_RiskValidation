# 07. Label-free performance estimation and the champion and challenger comparison

Modules `riskvalidation.ml.cbpe` and `riskvalidation.ml.challenger`. Sources: dossier 06 C.7 l.1041-1044 and l.1052-1058 and I.1
l.1196 and l.1199; NannyML CBPE
(https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html); SS1/23 principle 3.3(b)(iii); ECB guide
paragraph 60.

## Confidence-based performance estimation (CBPE)

For a calibrated probability $\hat p_j$ and the predicted label $\hat y_j=\mathbf 1[\hat p_j\ge\theta]$ (06 C.7 l.1052-1058):

$$P(\hat y=y)_j=1-|\hat y_j-\hat p_j|,\qquad
TP=\sum_{\hat y_j=1}\hat p_j,\quad FP=\sum_{\hat y_j=1}(1-\hat p_j),\quad
TN=\sum_{\hat y_j=0}(1-\hat p_j),\quad FN=\sum_{\hat y_j=0}\hat p_j$$

The four expected counts sum to $n$. `cbpe_confusion(p, threshold)` returns them (`yhat = 1` when `p >= threshold`; a probability
equal to the threshold is predicted 1) and reproduces the hand-computed rows: $p=(0.9,0.3,0.6,0.2)$ at $\theta=0.5$ gives
$TP=1.5$, $FP=0.5$, $TN=1.5$, $FN=0.5$.

`cbpe_auc(p, n_thresholds=101)` estimates the ROC AUC by repeating the confusion over thresholds evenly spaced on [0, 1]: the expected
rates $TPR=TP/(TP+FN)$ and $FPR=FP/(FP+TN)$ trace a curve from (1, 1) at threshold 0 to (0, 0) above the largest probability, and
the AUC is the trapezoidal area under it. No label is read. The result is always labelled `estimate` and carries the four
assumptions (calibrated probabilities, no concept drift, no extrapolation into unseen regions, adequate sample). Checks (tested):
on a uniform grid of calibrated probabilities the exact AUC is $5/6$ (positives have density $2p$, negatives $2(1-p)$, so
$\mathrm{AUC}=4\int_0^1p\,(p-p^2/2)\,dp$) and the estimate is within 2e-4 of it; on a calibrated simulated sample it is within Monte
Carlo error of the labelled AUC; and over 4,000 repetitions it is unbiased when the probabilities are calibrated and moves away
from the labelled AUC when they are not (page 08). An early-warning estimate while defaults mature: it complements, never
replaces, back-testing (the SR 11-7 early-warning reading of 06 C.7 l.1052-1058; SR 11-7 is the historical text, superseded in the
US by SR 26-2, 06 A.1).

## Champion and challenger

`paired_brier_bootstrap(p_champion, p_challenger, y, n_boot, seed)` resamples obligors jointly (the same obligors score both
forecasts: the existing `bootstrap_ci` resamples one score and cannot) and returns the Brier difference (challenger minus
champion) with a 95 percent percentile interval (06 C.7 l.1041-1044; SS1/23 3.3(b)(iii); ECB paragraph 60):

$$d_i=(q_i-y_i)^2-(p_i-y_i)^2,\qquad \Delta=\bar d,\qquad [\ell,u]=\big[Q_{2.5}(\bar d^*),\,Q_{97.5}(\bar d^*)\big]$$

where $p$ is the champion's PD, $q$ the challenger's and $\bar d^*$ the means of the $B$ bootstrap resamples. Deterministic for a seed;
identical forecasts give a zero difference and a zero-width interval; the width agrees with the normal-theory width
$2\cdot1.96\,s_d/\sqrt n$ within 8 percent at 4,000 resamples (tested); the interval contains the point estimate for 60 of 60 seeds
at 200 resamples (tested). The level 0.95 is the default of `bootstrap_ci`, a convention.

`grade_disagreement(pd_a, pd_b, edges, notches=2)` is the share of obligors whose grades (the number of edges at or below the PD)
differ by at least `notches`, symmetric in the two vectors; 2 notches is a convention (06 C.7 l.1041-1044 names "2 or more").

## What these checks do not claim

- A CBPE value is an estimate under four assumptions, never evidence of the model's performance: when a label is supplied the
  labelled AUC is read instead (page 09).
- A Brier interval that includes zero does not show the two forecasts are equally good; it shows this sample cannot tell them apart.
- The engine computes and makes no compliance claim.

## Sources

- NannyML documentation, CBPE (the formulas above); dossier 06 C.7 l.1041-1058.
- The percentile interval and the level 0.95 are those of the engine's `riskvalidation.validation.discrimination.bootstrap_ci`
  (which stratifies by default status; the paired form resamples obligors jointly, unstratified, as the contract states).
- SS1/23 (PRA, 2023) principle 3.3(b)(iii); ECB guide paragraph 60 (a complexity must be justified by a gain over a simpler benchmark).

## Limits

The percentile interval is not bias-corrected. UNVERIFIED: the NannyML text is read as dossier 06 transcribes it; the ROC estimate
over thresholds is the dossier's "repeating over thresholds" with the trapezoidal rule, which the dossier does not name.
