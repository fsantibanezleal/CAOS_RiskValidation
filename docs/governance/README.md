# Governance: scope, inventory, tiering, lifecycle, findings, schedule, opinion, appetite, policy, report and the `gov.` tests

`riskvalidation.governance` computes the model-risk-governance record of a set of models: the **scope test** of a model against four
jurisdiction profiles, the **inventory** of entries with the forty attributes of dossier 06 B.2 and the dependency graph between them, the
**tier** of each entry from a materiality by complexity matrix with the tier under four readings of the rubric, the **lifecycle**
transitions, the **findings workflow** (severity, deadline, ageing, overdue, the seven states, the quantitative anchor), the **review
schedule**, the **overall opinion** on the ECB four-level scale, the **model risk appetite metrics**, the **policy identity** of a test row
and the twelve-section **validation report**. `riskvalidation.validation.governance` returns five `gov.` tests as `TestResult` rows with a
traffic light. The contract is `engine-governance-api` 1.0.0 (Contraste `docs/contracts/engine/governance-api.md`, branch `task/u15-s` at
5dce395, after the Opus review's fix round).

The engine **computes; it approves nothing and certifies nothing** (SDD non-goal 1; dossier 06 I.6). Every tier count, band, deadline,
frequency and threshold is a stated **convention**: a named parameter that carries the label `convention` and its basis, passed to its
function as an argument with the default stated on these pages, and never presented as a value taken from a regulation (item 30 of the
dossier's UNVERIFIED list: "tier counts, traffic-light p-value bands, remediation deadlines and revalidation frequencies are proposals
(conventions), not sourced requirements").

## Pages

| Page | Contents |
|---|---|
| [01 Scope and inventory](01_scope-and-inventory.md) | the B.1 scope test, the forty B.2 attributes, entries, components, edges, the acyclic graph, completeness |
| [02 Tiering and lifecycle](02_tiering-and-lifecycle.md) | exposure bands, materiality, complexity, the 4 by 4 matrix, the override, the four readings, the twelve lifecycle states |
| [03 Findings and schedule](03_findings-and-schedule.md) | severity and status maps, deadlines, ageing and overdue, the seven finding states, the anchor, the register counts, the review dates |
| [04 Opinion and appetite](04_opinion-and-appetite.md) | the opinion level with the 5 and 10 percent RWEA cut points, the recommendation, the seven appetite metrics |
| [05 Policy and report](05_policy-and-report.md) | the policy identity of a test row and the twelve sections of the validation report |
| [06 The `gov.` validation tests](06_validation-tests.md) | the five tests, their H0, units, lights and policy rows |

## Conventions of every page

- **Purity.** No function reads the clock, the environment or a file: the as-of date and every table come in as arguments. Inputs are never
  mutated; the same inputs give the same output. The package imports only the standard library (the validation tests add NumPy through the
  engine core).
- **Dates and units.** Dates are `datetime.date`; the documents write them as ISO strings. Shares and ratios are decimals (5 percent is `0.05`);
  a month count is an integer of calendar months (an addition clamps the day to the end of the target month); money is never converted.
- **Missing data.** A function that cannot compute a value raises `ValueError` naming the reason, or returns `None` with a reason, or a test
  returns the light `not_evaluated` with its reason in `notes`; nothing returns a placeholder number.
- **Severities and tiers.** Severities are the product's `S1` (critical) to `S4` (minor), with the ECB F equivalent beside them (S1 F4, S2 F3,
  S3 F2, S4 F1). Tiers are integers 1 to 4, 1 the highest.
- **As-of date.** Every document carries `as_of` and `as_of_basis` (label `convention`); the production rule is the date of the newest release
  heading of `CHANGELOG.md` when the bake runs, never earlier than a raised date.

## The named conventions

Each is an argument with this default, labelled `convention` in `riskvalidation.governance.CONVENTIONS` with its basis; none is regulatory.

| Convention | Default | Where |
|---|---|---|
| `EXPOSURE_BAND_CUTS` | share of the stated balance 25, 10 and 2 percent | [02](02_tiering-and-lifecycle.md) |
| `DEADLINE_DAYS` | days from the raised date by severity and tier, S1 30 to 90 up to S4 180 to 365 | [03](03_findings-and-schedule.md) |
| `FREQUENCY` | review, full revalidation, monitoring months by tier | [03](03_findings-and-schedule.md) |
| `APPETITE_THRESHOLDS` | direction, green limit and amber limit per metric | [04](04_opinion-and-appetite.md) |
| `REGULATORY_USE_GRADES` | graded 4, 3, 2, 1; literal 4, 2, 1 | [02](02_tiering-and-lifecycle.md) |
| `COMPLEXITY_AGGREGATION` | the mean rounded half up, or the maximum | [02](02_tiering-and-lifecycle.md) |
| `ADOPTED_READING` | reading B (graded regulatory use, mean complexity) | [02](02_tiering-and-lifecycle.md) |
| `LIFECYCLE_GRAPH` | the twenty-four allowed transitions of twelve states | [02](02_tiering-and-lifecycle.md) |
| `FINDING_GRAPH` | the sixteen allowed transitions of seven states | [03](03_findings-and-schedule.md) |
| `RECOMMENDATION_RULE` | level 1 approve, 2 approve with conditions, 3 and 4 remediate before use | [04](04_opinion-and-appetite.md) |

The one pair of values in this package that is not a convention is the 5 and 10 percent RWEA cut points of an anchor, which are the ECB's
(Instructions for reporting the validation results of internal models, February 2019, section 2.4.2). The rule that turns findings into a
level is a convention all the same.

## What the engine does not claim

The list of dossier 06 I.6 (l.1261-1277), kept as the contract of what any text, page or document of this package may state:

- No regulatory approval, supervisory endorsement or "compliance" of any model or of Contraste itself. A passed test is evidence, not
  compliance; the ECB says implementing all listed analyses "does not necessarily mean that the validation requirements according to CRR
  have been fulfilled". Contraste approves no model and certifies none: the recommendation of the report is "the rule's output on the
  evidence; Contraste approves no model".
- SR 11-7 is not current US guidance (superseded 2026-04-17 by SR 26-2), and SR 26-2 does not cover generative or agentic AI: under the
  profile `us-2026` such a model is out of scope.
- PSI 0.10/0.25, four-fifths, p < 0.05, tier counts and remediation deadlines are conventions or policy choices, never regulatory
  thresholds. The bands, deadlines, frequencies and appetite limits of this package are the same kind of value.
- The four-fifths rule is an employment benchmark, not a credit-law standard.
- A calibration test does not prove a PD correct: no really powerful calibration test exists under default correlation, and a one-year test
  does not validate a long-run average PD.
- No EU AI Act conformity is claimed; conformity is a provider or deployer process, not a software output.
- Model-risk quantification outputs (Glasserman-Xu bounds) are not capital requirements.
- No equivalence to ECB reporting templates beyond the formulas reproduced; the templates themselves were not reviewed.

**UNVERIFIED**, carried from the dossier and the U15 research addendum D6: the SS1/23 2026 text against the 2023 text (not diffed); the OSFI
2017 guideline's rescission (no OSFI page opened states a replacement; OSFI's transition is "an 18-month transition period" in OSFI's own
words); the reading of EU AI Act Annex III 5(b); a post-2019 ECB reporting instruction; the EBA draft RTS on material model changes. The
rules of the scope test are a reading of the B.1 table and are UNVERIFIED beyond it. A packaged policy file per battery is backlog BL-085.

## Sources

Dossier 06 of the Contraste initiative: B.1 to B.8 (l.523-640), I.2, I.3 (l.1219-1237), I.6 (l.1261-1277), UNVERIFIED item 30 (l.1321-1322);
the U15 research addendum D1 to D6 (the B.2 attribute map, the tiering inputs of the nineteen built cases, the findings, the dependencies, the
report against what Contraste holds, the regulation check); the unit sizing decisions W15-1 to W15-15; the references named on each page
(SS1/23, SR 26-2, OSFI E-23, ECB Instructions 2019, Delegated Regulation 2022/439).
