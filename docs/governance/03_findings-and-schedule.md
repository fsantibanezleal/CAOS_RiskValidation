# 03 Findings and schedule

Modules `riskvalidation.governance.findings` and `riskvalidation.governance.schedule`. Sources: dossier 06 B.7 (l.614-628) and B.8 (l.630-640);
decisions W15-2 (the severity direction), W15-5 (the status map), W15-11 (the RWEA anchor), W15-13 (the as-of date) and W15-14 (the counts).
The deadline table, the finding transition graph and the frequency table are **conventions**: no source prescribes days, 06 l.624-628 names the
seven states and no transition, and SS1/23 and OSFI state the review frequency as "consistent with the model tier" with no number.

## Severity and status

The product keeps its direction S1 critical to S4 minor (W15-2); `severity_to_ecb` returns the ECB TRIM F equivalent (06 l.616-617, "F1 to F4,
where F1 refers to a low impact and F4 to a very high impact"): S1 is F4, S2 F3, S3 F2, S4 F1. `map_product_status` maps the product's three
statuses to the B.7 states (W15-5): `open` to `open`, `accepted` to `risk_accepted`, `closed` to `validated_closed`. `overdue` is never returned
by it: it is computed from the due date and the as-of date by `is_overdue`.

A `Finding` is keyed `case_id/variant_id/id` because the same finding repeats in each variant of a case; its evidence is non-empty and its
`status` is the mapped product status unless the workflow has moved it.

## Deadline, ageing and overdue

`deadline_days(severity, tier, table=DEADLINE_DAYS)` returns the days from the raised date (a **convention**, rows S1 to S4, columns tier 1 to 4):

| Severity | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| S1 | 30 | 45 | 60 | 90 |
| S2 | 60 | 90 | 120 | 180 |
| S3 | 90 | 120 | 180 | 270 |
| S4 | 180 | 270 | 365 | 365 |

With the default table the days are non-decreasing in the tier number and in the severity number. A row that does not hold four values, an
unknown severity and a tier outside 1 to 4 raise `ValueError`. `due_date(raised, days)` adds calendar days: `due_date(2026-10-12, 30) =
2026-11-11` and `due_date(2026-10-16, 60) = 2026-12-15`. `ageing_days(raised, as_of)` counts the whole days: `ageing_days(2026-10-12,
2026-12-31) = 80`; an as-of date before the raised date raises `ValueError`.

`is_overdue(status, due, as_of)` is true only for the five open states (`open`, `action_plan_agreed`, `in_progress`, `submitted_for_closure`,
`overdue`) and only after the due date: on the due date a finding is not overdue (`is_overdue('open', 2026-11-11, 2026-11-11)` is false, one
day later true), and a `risk_accepted` or `validated_closed` finding is never overdue.

## The seven states

`transition_finding(state, to, risk_acceptance=None)` implements the graph (`FINDING_GRAPH`, a convention): sixteen of the 49 ordered pairs of the
seven states are allowed, thirty-three refused (`cannot move from 'validated_closed' to 'open'`).

| From | To |
|---|---|
| `open` | `action_plan_agreed`, `risk_accepted`, `overdue` |
| `action_plan_agreed` | `in_progress`, `risk_accepted`, `overdue` |
| `in_progress` | `submitted_for_closure`, `risk_accepted`, `overdue` |
| `submitted_for_closure` | `validated_closed`, `in_progress`, `overdue` |
| `overdue` | `in_progress`, `submitted_for_closure`, `risk_accepted` |
| `risk_accepted` | `open` (when its expiry or its basis lapses) |
| `validated_closed` | none (terminal; a reopened finding is a new finding) |

Entering `risk_accepted` needs an acceptance record with an `approver` field and an `expiry` field; the register writes `"not-applicable"` and
`None`, each with a reason, for a stated limit of a synthetic case.

## The anchor and the RWEA share

`anchor_from_test_row(evidence, tests)` gives a finding its quantitative anchor. Only evidence of the form `test_id@model_id` or
`test_id@model_id@segment` is cited, the first one in the list. The anchor is the first non-null of the cited row's `statistic` and `metric`,
with the unit and the pointer it was read from (`<document>#/tests/<i>/<field>`); a statistic is in `test statistic (dimensionless)`, and a
metric is in the unit its test defines, never guessed (a row may state its own `unit`). Evidence such as `design:b1-minimum-normal-individual`
or `rate:rating.migration_ztests@r2-n500`, or a cited row whose statistic and metric are both null, gives `(None, reason)`.

`rwea_share_of_anchor(anchor, rwea_total=None)` returns the share of risk-weighted exposure amounts an anchor states, non-null only where the
anchor's unit is `share-of-rwea` (the value) or `rwea-amount` (the value over a positive `rwea_total`): `(0.08, None)` for 0.08 of RWEA and
`(0.1001, None)` for 750000 over 7492853. Any other unit, or no anchor, gives `(None, reason)`: a statistic never becomes a share. The share
feeds [`finding_level`](04_opinion-and-appetite.md).

## Register counts

`register_counts(rows)` returns `total`, `by_severity` (S1 to S4), `by_status` (the seven states), `overdue` and `by_case` (sorted by case id,
with the same four counts); each of `by_severity` and `by_status` sums to `total`. For the fixture register of four entries: 81 rows, S1 2, S2
14, S3 41, S4 24, `open` 49, `validated_closed` 4, `risk_accepted` 28, 7 overdue.

## The review schedule

`add_months(d, n)` adds calendar months and clamps the day to the end of the target month: `add_months(2026-08-31, 6) = 2027-02-28`,
`add_months(2024-01-31, 1) = 2024-02-29`. `next_dates(tier, last_validation, table=FREQUENCY, profile="default")` returns the next review, the
next full revalidation (`None` where the table states none) and the next monitoring date (`None` where none). The frequency table (a
**convention**, 06 B.8 l.636-640; months for review, full revalidation and monitoring) is:

| Tier | Review | Full revalidation | Monitoring |
|---|---|---|---|
| 1 | 12 | 36 | 3 |
| 2 | 12 | 36 | 6 |
| 3 | 24 | none | 12 |
| 4 | 36 (an attestation) | none | none |

From a last validation of 2026-10-12, tier 1 gives (2027-10-12, 2029-10-12, 2027-01-12) and tier 4 gives (2029-10-12, None, None). With the
profile `ecb` the IRB minimums apply: a review at most every 12 months and a full revalidation at most every 36 months, the minimum applying even
where the table states none (06 l.633-636; Delegated Regulation 2022/439 Article 11), so tier 3 under `ecb` gets a full revalidation at 36
months.

## Tests

`tests/governance/test_governance_findings.py` (every allowed and refused pair, the arithmetic at `due = as_of`, the anchor and the share) and
`tests/governance/test_governance_schedule_opinion.py`; the 81 register rows are recomputed field by field by `tests/governance/test_governance_fixtures.py`.
The `gov.` tests [`gov.findings_overdue`](06_validation-tests.md) and `gov.validation_due` read these functions.
