# 06 The `gov.` validation tests

Module `riskvalidation.validation.governance`. Sources: dossier 06 B.2 (l.539-551), B.3 (l.574-575), B.7 (l.614-628), B.8 (l.630-640), I.2;
SS1/23 Principles 1.2 and 1.3(d) and (e); OSFI E-23 Appendix 1; SR 11-7 on model dependencies.

Each test returns one `TestResult` that validates against `common/test-result`, is registered in the family `governance` and has a test id
prefixed `gov.`. `model_id` and `segment` are `None`: the rows are about the inventory, not about a model. `statistic` and `p_value` are `None`;
`metric` carries the measurement, and the light comes from a policy row keyed by the test id. Every cut point is a **convention** (item 30),
recorded in `extras["thresholds"]` with the label `convention`; none is a regulatory threshold. A test with nothing to evaluate returns the light
`not_evaluated` with its reason in `notes` and no number in the serialised row. A `policy` argument of `None` means the packaged default policy
with the `gov.` rows added where it holds none (`gov_default_policy`); a policy that lacks the key gives `not_evaluated`, never a colour.

| Test | H0 | `metric`, `n`, `n_events` | Light (policy row) |
|---|---|---|---|
| `gov.inventory_completeness` | every one of the 40 attributes of every entry has a status other than `missing` | the share of (entry, attribute) pairs not `missing`; the pairs; the missing pairs | `lower_bad`, strict: amber below 0.9, red below 0.7 |
| `gov.findings_overdue` | no open finding is past its due date | the share of open-state rows that are overdue; the open-state rows; the overdue rows | `upper_bad`, strict: amber above 0, red above 0.25 |
| `gov.validation_due` | no entry is past its next review date | the number of entries whose next review date is before the as-of date; the entries; the same count | `upper_bad`: amber at 1, red at 2 or more |
| `gov.tier_override_documented` | every tier override has a rationale and an approver | the number of overrides without a rationale or an approver; the overrides; the same count | `upper_bad`: red at 1 or more |
| `gov.edges_acyclic` | the dependency graph has no cycle and every edge joins two entries | the entries on a cycle plus the edges to an unknown entry; the edges; the same count | `upper_bad`: red at 1 or more |

A strict row states a cut that "below" or "above" triggers: it is stored one float beyond the stated value (`math.nextafter`), so a metric equal
to the stated value is not past it (`Threshold.classify` is inclusive at its cut points). A metric of exactly 0.9 is green for
`gov.inventory_completeness`; 1 of 4 open-state rows overdue (0.25) is amber, not red, for `gov.findings_overdue`.

## Golden values on the fixture

| Test | n | n_events | metric | Light |
|---|---|---|---|---|
| `gov.inventory_completeness` | 160 | 46 | 0.7125 | amber |
| `gov.findings_overdue` (as of 2026-12-31) | 49 | 7 | 0.1429 | amber |
| `gov.validation_due` (as of 2026-12-31) | 4 | 0 | 0 | green |
| `gov.tier_override_documented` (no override) | 0 | 0 | 0 | green |
| `gov.edges_acyclic` | 2 | 0 | 0 | green |

A cycle among three entries gives `gov.edges_acyclic` a metric of 3, light red, and the cycle named in `notes` the way `build_inventory` names it
(`dependency cycle: A, B, C, A`). An inventory with no entry returns `not_evaluated` for the completeness test with its reason; with no override,
`gov.tier_override_documented` is green with `n` 0, not `not_evaluated`.

## What a light says

The lights say where the **record** is incomplete, late or inconsistent. They say nothing about the quality of a model, and a green light is not
an approval: Contraste approves no model. The cut points are the engine's defaults for the synthetic bank of the product, set by the main session;
an institution would set its own.

## The policy rows

The rows live in `riskvalidation.validation.governance.GOV_ROWS` (the five keys above, each with its kind, stated cuts, whether it is strict
and its source, which starts with the word `convention`) and `gov_default_policy()` adds them to the packaged default policy. Once
`policies/default.json` holds them the two are the same object, as for the `fair.` rows. The packaged policy rows to add (the stated cut
points; the strict rows are stored one float beyond them by `gov_threshold`):

| Key | Kind | Amber | Red |
|---|---|---|---|
| `gov.inventory_completeness` | `lower_bad` | 0.9 (strict) | 0.7 (strict) |
| `gov.findings_overdue` | `upper_bad` | 0 (strict) | 0.25 (strict) |
| `gov.validation_due` | `upper_bad` | 1 | 2 |
| `gov.tier_override_documented` | `upper_bad` | 1 | 1 |
| `gov.edges_acyclic` | `upper_bad` | 1 | 1 |

## Tests

`tests/governance/test_governance_validation.py`: the goldens above, every boundary, the not-evaluated cases, the serialisation into `common/test-result`
rows and a run in a suite under the governance policy.
