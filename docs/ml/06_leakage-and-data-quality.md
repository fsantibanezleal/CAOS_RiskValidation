# 06. The leakage screen and the data-quality counts

Modules `riskvalidation.ml.leakage` and `riskvalidation.ml.dataquality`. Sources: dossier 06 C.7 l.1045-1051 (data quality and
leakage detection) and I.1 l.1197-1198; dossier 07 D pattern 3, l.597-600 (leakage and sample-integrity tests: mercury-robust
`LabelLeakingTest`, `SampleLeakingTest`, `NoDuplicatesTest`; the probatus resemblance model); ECB guide paragraph 52(h)(vii) and
Overarching paragraph 11. The leakage screens are conventions: no regulatory text prescribes them (06 C.7).

## Leakage

- **Temporal order.** `temporal_order_violations(feature_dates, observation_date)` counts the feature values dated after the
  observation date of their row (`datetime64[D]` arrays of shape `(n, p)` and `(n,)`). NaT is never a violation and a date equal
  to the observation date is not one either. A single violation is a leak by definition (06 C.7 l.1048-1051).
- **Single-feature AUC.** `single_feature_auc(X, y, threshold=0.95)` returns the largest $\max(\mathrm{AUC}_j,1-\mathrm{AUC}_j)$
  over the columns, computed with the engine's `disc_auc`, its column (the first on a tie) and whether it exceeds the near-perfect
  threshold strictly (a convention): a lone near-perfect feature is a suspect. A leak that ranks the other way is found by the
  $1-\mathrm{AUC}$ branch.
- **Entity overlap.** `entity_overlap(train_ids, test_ids)` is the share of distinct test entities present in training,
  identifiers compared as strings; an empty test set raises.
- **Resemblance AUC.** `resemblance_auc(X_dev, X_oot, seed)` is the cross-fitted AUC, with the ECB variance of
  `auc_with_variance`, of a classifier separating development rows (class 0) from out-of-time rows (class 1): near 0.5 when the
  samples agree (ING probatus `sample_similarity`). The classifier is the engine core's ridge logistic fit with the 5-fold
  cross-fit of page 01; rows with an identical feature vector stay in one fold.

$$\mathrm{AUC}_{\mathrm{res}}\approx\tfrac12\ \text{when the samples agree},\qquad
\text{suspect}\iff\max_j\max(A_j,1-A_j)>0.95$$

## Data quality

Six counted checks feeding a `TestResult`; each returns `DataQualityCount(check, n_checked, n_failed, share_ok)` with
$\text{share\_ok}=1-n_{\text{failed}}/n_{\text{checked}}$.

| Function | What is counted | Convention |
|---|---|---|
| `dq_completeness(X)` | NaN cells of `X` (also the count per column, in the subclass `CompletenessCount.by_column`); maps to SGCD shock F1 | `n_checked` is the number of cells |
| `dq_validity(X, lo, hi)` | cells outside `[lo, hi]` per column, bounds inclusive | NaN is not a violation and is not checked: `n_checked` is the number of non-NaN cells |
| `dq_uniqueness(ids)` | repeated identifiers after the first occurrence | the first occurrence is kept |
| `dq_consistency(a, b)` | rows where two views of the same field disagree (SGCD F4 reads both views) | equal-length sequences |
| `dq_timeliness(dates, as_of, max_age_days)` | rows older than `max_age_days` at `as_of` (ISO dates; ECB 52(h)(vii): the age of the financial statements) | an age equal to the maximum, or a date after `as_of`, is not stale |
| `dq_label_consistency(X, y)` | rows whose feature vector equals, exactly, that of a row with a different label (06 C.7: labelling consistency) | a planted contradictory pair counts both rows |

## What these checks do not claim

- A passed screen is not proof of no leakage: the four screens find the leaks they were written for (a future-dated feature, a
  near-perfect single feature, an entity in both slices, a development against out-of-time shift). A leak through a combination of
  features or through the labelling process is not found by them.
- The counts say how many cells or rows fail a rule, not whether the failure matters; the rules (ranges, maximum age) are the
  caller's.
- The engine computes and makes no compliance claim.

## Sources

- Dossier 06 C.7 l.1045-1051 and I.1; dossier 07 D pattern 3; ECB guide paragraphs 52(h)(vii), 52(h)(ix) and Overarching paragraph 11.
- ING probatus `sample_similarity` (resemblance model), https://github.com/ing-bank/probatus (cited by dossier 06 and 07).

## Limits

UNVERIFIED: the contract's "counts per column" of `dq_completeness` is returned as the `by_column` field of a subclass of
`DataQualityCount`, because the contract's four-field dataclass has no place for it; the subclass is still a `DataQualityCount`.
The label-consistency count reads exact equality of float rows, so two rows that differ in the last bit are different rows.
