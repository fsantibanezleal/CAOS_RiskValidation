# 02. Seed stability and the explanation checks

Modules `riskvalidation.ml.seeds` and `riskvalidation.ml.explanations`. Sources: dossier 06 I.1 l.1191-1193 and C.7 l.990,
l.1006-1008; ECB guide to internal models paragraph 43 (the randomness of the training process; replicability with seeds) and
paragraph 59 (stored seeds), 06 A.3 l.290-299 and B.9 l.647; EBA/REP/2023/28 recommendation 3 (06 A.8 l.480-482); dossier 04 J.2
l.881-883 and ladder P4 l.968.

## Seed stability

`seed_auc_dispersion(train, X, y, seeds)` fits with each of at least two distinct seeds (`train(seed)` returns a predict callable,
which the pipeline supplies) and returns the mean, the sample standard deviation (ddof 1), the minimum and the maximum of the AUC.
A train function that ignores its seed has a zero standard deviation (identity).

$$\bar A=\frac1K\sum_k A_k,\qquad s_A=\sqrt{\frac1{K-1}\sum_k(A_k-\bar A)^2}$$

`grade_assignment_stability(pd_by_seed, edges)` is the share of obligors whose grade differs between at least two seeds. The
grade of a PD is the number of `edges` at or below it (the bins are $[e_k, e_{k+1})$: an edge value belongs to the upper grade);
an obligor counts once; one seed returns 0.

`attribution_rank_stability(attributions_by_seed)` returns the mean and the minimum, over the seed pairs, of the Kendall
$\tau_b$ (`scipy.stats.kendalltau`, which handles ties) between the per-feature mean absolute attributions of two seeds; a
constant row has no rank and raises.

## Reason-code stability and the explainer agreement

`reason_code_overlap(attributions_by_refit, top_k=4)`: the attributions are an array of shape `(n_refits, n, p)` computed in the
pipeline by the pinned `shap` (the engine imports no learning library). The reasons of an obligor in a refit are its `top_k`
features of largest **absolute** attribution (ties by column order). For each obligor the share of the reasons kept between each
later refit and the first is averaged over the later refits; the result carries the mean and the 5th percentile over obligors
(`numpy.percentile`, linear). Identical refits give 1; a planted swap of two columns gives the hand-computed share (a swap of the
fourth and fifth feature for one of two obligors gives a mean of 0.875 and a 5th percentile of 0.7625).

- **UNVERIFIED reading (absolute attribution).** The dossier says "bootstrap rank stability of the top four" (04 J.2 l.881-883)
  without a sign, and the contract does not settle it. A Reg B reason is usually the feature that pushes the score towards denial
  (the largest positive attribution on the PD scale); the engine ranks by magnitude because the function cannot know which sign is
  adverse. The signed variant is one `np.abs` away and needs a contract decision.
- **Finding F1.** The contract states "`p05 <= mean`" as an invariant. It is not a property of a share distribution: 96 obligors at
  share 1 and 4 at share 0 have a 5th percentile of 1 above the mean 0.96 (a test plants it). The function reports the exact
  percentile and `OverlapResult` does not enforce the inequality.

`rank_agreement(attribution_importance, effect_range)` is the Spearman $\rho$ between two per-feature vectors: the mean absolute
SHAP per feature against the range of its ALE or PDP curve (the explainer agreement of C20, 06 C.7 l.1006-1008), or the mean
absolute SHAP on the clean batches against the shocked ones. It is computed as the Pearson correlation of the average ranks,
$\rho = S_{ab}/\sqrt{S_{aa}S_{bb}}$ on the centred ranks (what `scipy.stats.spearmanr` computes), so that equal vectors give exactly
1 and reversed ones exactly -1 (`spearmanr` returns `0.9999999999999999` for equal vectors); a test compares it with `spearmanr`
to 1e-15. A constant vector has no ranks and raises (the case document then carries null and its reason).

## Additivity, sign and monotonicity

The Shapley value of feature $i$ for the prediction $f(x)$ over the feature set $F$ (Lundberg and Lee 2017, arXiv 1705.07874;
dossier 06 C.7 l.979-990) obeys additivity:

$$f(x)=\phi_0+\sum_i\phi_i,\qquad
\text{gap}=\max_{r}\Big|f(x_r)-\big(\phi_0+\sum_i\phi_{ri}\big)\Big|$$

`shapley_additivity_gap(prediction, base_value, attributions)` returns that maximum (prediction units); it is below 1e-12 for the
exact Shapley values of a linear model, $\phi_i=w_i(x_i-\bar x_i)$ and $\phi_0=b+w^{\top}\bar x$ (tested), and equals a planted gap.

`sign_check(effect, expected_sign)` is true when the net change of the ALE or PDP values from the first to the last grid point has
the expected sign (-1 or +1, from the stated economic rationale; 06 I.1 l.1192; EBA recommendation 3) and is nonzero.
`monotonicity_check(effect, expected_sign, tolerance)` returns the largest fall of the curve below its running maximum when a rise
is expected (above its running minimum when a fall is expected); a violation within the tolerance counts as none (0 is returned),
otherwise the violation itself, so a planted bump of known height returns that height.

## What these checks do not claim

- A stable AUC across seeds is not a stable model: the dispersion of grades and attribution ranks is read beside it. The seed set
  is the pipeline's; the engine counts the seeds it is given.
- Agreement of SHAP with ALE or PDP ranks shows the explainers tell a similar story on this model, not that either is right. The
  cut points of the lights are conventions (page 09).
- The engine computes and makes no compliance claim.

## Sources

- Lundberg and Lee (2017), "A Unified Approach to Interpreting Model Predictions", arXiv 1705.07874; Lundberg et al. (2020),
  Nature Machine Intelligence 2(1):56-67, doi:10.1038/s42256-019-0138-9.
- `scipy.stats.kendalltau` (tau-b, the variant that handles ties) and `scipy.stats.spearmanr`, the references of the tests.
- Dossier 06 C.7 and I.1; dossier 04 J.2 and ladder P4; ECB guide paragraphs 43, 59, 62 and 64 to 66; EBA/REP/2023/28.

## Limits

`reason_code_overlap` needs at least two refits and `top_k` at most the number of features. Attributions are arrays: the engine
does not compute them. UNVERIFIED: the sign reading of the top reasons and the invariant of finding F1, above.
