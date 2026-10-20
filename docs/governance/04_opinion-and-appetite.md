# 04 Opinion and appetite

Modules `riskvalidation.governance.opinion` and `riskvalidation.governance.appetite`. Sources: dossier 06 B.6 (l.601-612), B.7 (l.620-623);
ECB Instructions for reporting the validation results of internal models (February 2019), section 2.4.2; decisions W15-7, W15-11 and W15-14. The
**rule** that turns findings into a level, the recommendation and every appetite limit are **conventions** (item 30); the 5 and 10 percent cut
points of an RWEA anchor are the ECB's.

## The ECB scale and the finding level

The overall assessment of a validation has four levels (ECB 2019, section 2.4.2): 1 adequate, no deficiencies; 2 adequate, minor deficiencies;
3 major deficiencies (RWEA impact of at least 5 percent and below 10 percent); 4 severe deficiencies (RWEA impact of at least 10 percent).

`finding_level(severity, rwea_share=None)` is the level one open finding gives: **the maximum of its severity level and its RWEA level**.
The severity level is S1 4, S2 3, S3 2 and S4 2; the RWEA level is 3 from 5 percent of RWEA, 4 from 10 percent and none below 5 percent. An
anchor can therefore only raise a level, and a critical finding is never downgraded by a small share. Goldens: `finding_level('S1') = 4`, `S2 -> 3`,
`S3 -> 2`, `S4 -> 2`; `finding_level('S4', rwea_share=0.0499) = 2` and `0.05 -> 3`; `finding_level('S3', rwea_share=0.0999) = 3` and `0.10 -> 4`;
`finding_level('S1', rwea_share=0.0499) = 4`; `finding_level('S2', rwea_share=0.10) = 4`. A negative share or an unknown severity raises `ValueError`.

`opinion_level(open_findings)` takes the open findings as (severity, RWEA share or `None`) pairs and returns the maximum of their levels, 1 with
none. The caller passes only the open findings: `open`, `action_plan_agreed`, `in_progress`, `submitted_for_closure` and `overdue` rows; a
`risk_accepted` or `validated_closed` finding is not one. Goldens: `opinion_level([]) = 1`, `[S4, S3] -> 2`, `[S3, S2] -> 3`, `[S2, S1] -> 4`,
`[(S3, 0.0499), (S4, None)] -> 2`, `[(S3, 0.05)] -> 3`, `[(S4, 0.10)] -> 4`, `[(S1, 0.01)] -> 4`.

`recommendation(level)` returns (outcome, text, label): level 1 `approve`, level 2 `approve with conditions`, levels 3 and 4
`remediate before use`. The label is always "the rule's output on the evidence; Contraste approves no model". The text states the rule's output
and never an approval of a model by Contraste; the engine produces a recommendation to the approval authority of the client, nothing more.

## The seven appetite metrics

`appetite_metrics(entries, rows, as_of, thresholds=APPETITE_THRESHOLDS)` returns the seven records of dossier 06 B.6 (the model risk appetite,
SS1/23 Principle 2.1(c)), each with its `value`, `unit`, `numerator`, `denominator`, `register_rows`, `threshold` (labelled `convention`, with
its direction, green limit and amber limit and the basis) and `light`. `light(value, direction, green_limit, amber_limit)` returns `green`,
`amber`, `red` or `not_evaluated` (no value); the limits are inclusive on the good side.

| Metric | Function | Direction, green limit, amber limit | Fixture value (4 entries, 81 rows) |
|---|---|---|---|
| `tier1-validated-on-time` | `tier1_validated_on_time` | higher, 0.9, 0.75 | (2, 2, 1.0) green |
| `open-high-severity` | `open_high_severity` | lower, 0, 3 | (4, 39, 4, 8 rows) red |
| `overdue-remediation` | `overdue_remediation` | lower, 0, 3 | (3, 39, 3, 7 rows) amber |
| `entries-under-exception` | `entries_under_exception` | lower, 0, 1 | (0, 4, 0) green |
| `entries-in-breach` | `entries_in_breach` | lower, 0, 1 | (3, 4, 3) red |
| `pma-magnitude` | `pma_magnitude` | lower, 0.01, 0.03 | (324129997.0, 97592002519.53, 0.0033213) green |
| `owner-and-complete-record` | `owner_and_complete_record` | higher, 0.9, 0.5 | (0, 4, 0.0) red |

- `tier1_validated_on_time`: the Tier 1 entries whose next review date is on or after the as-of date, over the Tier 1 entries; the value is
  `None`, and the light `not_evaluated` with a reason, when there is no Tier 1 entry.
- `open_high_severity` and `overdue_remediation` count **distinct (entry, id) findings** (W15-14), which is what B.6 counts: a finding repeated in n
  variants of an entry counts once, over the distinct findings of the register. The number of register rows behind the numerator is the secondary
  number in the same metric object (`register_rows`).
- `entries_under_exception`: the entries whose lifecycle holds an exception record. An entry in use before validation without a record is a
  refused lifecycle state, not a count.
- `entries_in_breach`: the entries with at least one red test row on any variant (attribute A36), counted once. An entry with no light counts
  raises `ValueError` naming it, rather than being counted as not in breach.
- `pma_magnitude`: the sum of the post-model adjustments (attribute A26 with an `overlays_total`) over the sum of the model outputs they adjust,
  in one currency; amounts in two currencies raise `ValueError`, nothing is converted; the value is `None` when no entry holds a PMA with an amount.
- `owner_and_complete_record`: the entries whose owner (A28) is present, derivable or convention and none of whose forty attributes is
  `missing` (`not-applicable` counts as complete), over the entries.

`appetite_population(entries, rows)` returns the population the document states: `entries`, `findings` (register rows), `distinct_findings`
and `tier1_entries`.

## Tests

`tests/governance/test_governance_schedule_opinion.py` (every golden above and the invariant that an anchor only raises a level) and
`tests/governance/test_governance_appetite.py`; the seven metrics of the appetite fixture are recomputed by `tests/governance/test_governance_fixtures.py`.
