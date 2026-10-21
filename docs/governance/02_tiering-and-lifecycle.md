# 02 Tiering and lifecycle

Modules `riskvalidation.governance.tiering` and `riskvalidation.governance.lifecycle`. Sources: dossier 06 B.3 (l.553-574) and B.4 (l.576-583);
the U15 research addendum D2 (the tiering inputs of the nineteen built cases); decision W15-3. SS1/23 Principle 1.3, SR 26-2 and OSFI E-23
require a materiality and a complexity rating and prescribe neither a number of tiers nor a formula (verified for SS1/23 and SR 26-2 by reading
the full text); the rubric below is a **convention**, transparent and configurable.

## Exposure band

`exposure_band(share, cuts=EXPOSURE_BAND_CUTS)` maps the share of the stated balance to a band 1 to 4. The cut points `EXPOSURE_BAND_CUTS =
(0.25, 0.10, 0.02)` are a convention (06 B.3 says only "banded at configurable cut-offs"): a share of 25 percent or more is 4, 10 percent 3,
2 percent 2, below that 1, and `None` (no balance exists) is the floor 1. Goldens: `exposure_band(1.0) = 4`, `0.25 -> 4`, `0.2499 -> 3`,
`0.1 -> 3`, `0.0999 -> 2`, `0.02 -> 2`, `0.0199 -> 1`, `0.0 -> 1`. The band is non-decreasing in the share; a share outside [0, 1] and cut
points that are not decreasing in (0, 1] raise `ValueError`.

## Materiality and complexity

`materiality(band, regulatory_use, decision_criticality)` is the maximum of the three scores, each 1 to 4 (06 l.563-566). Regulatory use has
two readings, both conventions (`REGULATORY_USE_GRADES`): **graded** 4 where the output is itself the reported or binding figure, 3 where it
is a parameter or a back-test that feeds one, 2 for compliance monitoring, 1 for none; **literal** 4 whenever the output feeds capital,
provisioning or reporting. The literal grade is never below the graded one. Decision criticality is at least 3 for an automated credit decision
on natural persons. Goldens: `materiality(2, 3, 3) = 3` (C01, graded), `materiality(2, 4, 3) = 4` (C01, literal), `materiality(1, 1, 1) = 1` (C22).

`complexity(scores, aggregation="mean", rounding="half-up")` aggregates the scored factors (methodology, data, explainability, dynamic
updating, dependency depth, reproducibility, history against horizon), each 1 to 4; an unscored factor is left out of the list and never counted
as 1. The aggregation `COMPLEXITY_AGGREGATION` is a convention: the mean (06 l.567; the rounding is not in the text, half up is used) or the
maximum. The mean is rounded half up in integer arithmetic, $\lfloor (2 \sum s + n) / (2n) \rfloor$, so no floating point decides a boundary.
Goldens: `complexity([4, 2, 3, 1, 1, 1]) = 2` (mean 2.00), `[2, 2, 1, 1, 2, 1] -> 2` (1.50 rounds up), `[2, 2, 1, 1, 1, 1] -> 1` (1.33),
`[3, 3, 3, 2, 2, 2] -> 3` (2.50 rounds up), and the maximum of the first list is 4.

## The matrix, the override and the four readings

`matrix_tier(m, c)` is the 4 by 4 matrix of 06 l.571-574: Tier 1 if M is 4 or (M is 3 and C is 3 or more); Tier 2 if M is 3 or (M is 2 and C is
3 or more); Tier 3 if M is 2 or (M is 1 and C is 3 or more); Tier 4 otherwise.

| | C 1 | C 2 | C 3 | C 4 |
|---|---|---|---|---|
| M 4 | 1 | 1 | 1 | 1 |
| M 3 | 2 | 2 | 1 | 1 |
| M 2 | 3 | 3 | 2 | 2 |
| M 1 | 4 | 4 | 3 | 3 |

The tier never worsens when M or C rises. `apply_override(computed_tier, override)` returns the computed tier or the override's `to_tier`; an
override without a rationale or without an approver raises `ValueError("an override needs a rationale and an approver")`, never silent
(SS1/23 Principle 1.3(d) and (e)). The computed tier stays in the document beside the final tier.

`tier_sensitivity(share, regulatory_use_literal, regulatory_use_graded, decision_criticality, scores, cuts)` returns the tier under the four
readings of the rubric's two open inputs (what counts as regulatory use; the mean or the maximum for C): A literal and mean, **B graded and mean
(the adopted reading, `ADOPTED_READING`, W15-3)**, C literal and maximum, D graded and maximum. Each row is (reading, regulatory use,
complexity aggregation, M, C, tier). Reading A never gives a higher tier number than reading B. For the nineteen built cases of the addendum
D2 the tiers A, B, C, D reproduce the contract's table, and the counts of Tier 1, 2, 3 and 4 per reading are:

| Reading | Tier 1 / 2 / 3 / 4 |
|---|---|
| A (literal, mean) | 18/0/0/1 |
| B (graded, mean), adopted | 12/6/0/1 |
| C (literal, maximum) | 18/0/0/1 |
| D (graded, maximum) | 16/2/0/1 |

What the computation shows about the rubric (addendum 3.5): regulatory use saturates (every risk model is Tier 1 under the literal reading),
the mean never reaches 3 so the tier is a function of M alone, and the rubric has no materiality term for fan-out, so C22, whose error would
mis-size every test of every case, keeps Tier 4. `tier_counts(tiers)` counts the entries in tiers 1 to 4: `tier_counts([1, 2, 1, 4]) = (2, 1, 0, 1)`.

## The lifecycle

`allowed_next(state)` and `transition(state, to, exception=None)` implement the twelve states of B.4, a **convention** synthesised from SS1/23
para 2.10, OSFI E-23 and SR 11-7 (`LIFECYCLE_GRAPH`): `proposed`, `in_development`, `submitted_for_validation`, `in_validation`, `approved`,
`approved_with_conditions`, `in_use`, `under_monitoring_alert`, `material_change_pending`, `restricted`, `in_remediation`, `decommissioned`.
The allowed transitions are:

| From | To |
|---|---|
| `proposed` | `in_development` |
| `in_development` | `submitted_for_validation` |
| `submitted_for_validation` | `in_validation` |
| `in_validation` | `approved`, `approved_with_conditions`, `in_development` |
| `approved`, `approved_with_conditions` | `in_use` |
| `in_use` | `under_monitoring_alert`, `material_change_pending`, `in_validation`, `restricted`, `decommissioned` |
| `under_monitoring_alert` | `in_use`, `restricted`, `in_remediation` |
| `restricted` | `in_remediation`, `in_use`, `decommissioned` |
| `in_remediation` | `in_validation`, `restricted`, `decommissioned` |
| `material_change_pending` | `submitted_for_validation`, `in_use` |
| `decommissioned` | none (terminal) |

Use before validation, from `in_development`, `submitted_for_validation` or `in_validation` to `in_use`, is allowed only with an exception record
(type, approver, expiry date), all three present (SR 11-7 page 9; SR 26-2 Section V; SS1/23 Principle 5.3, where exceptions are "temporary").
Every other pair is refused with `cannot move from 'proposed' to 'in_use'`, naming both states. A decommissioned model may serve as a challenger
but does not re-enter the lifecycle. The state names are the vocabulary of the model owner's process: Contraste records a state, it never
moves a model into one.

## Tests

`tests/governance/test_governance_tiering.py` (the 19-case table, the whole matrix, the boundaries) and `tests/governance/test_governance_lifecycle.py` (every
allowed pair and every refused pair of the 144 ordered pairs).
