# 05 Policy and report

Modules `riskvalidation.governance.policy` and `riskvalidation.governance.report`. Sources: dossier 06 I.3 (the validation report template,
l.1219-1237, SR 11-7 and SS1/23 structure, Delegated Regulation 2022/439 Article 13 content), I.5 ("no light is shown without a policy-file
version"), I.6; the U15 research addendum D5; decisions W15-6, W15-12 and W15-13.

## The policy identity of a test row

`resolve_policy(policy_version, test_id, alpha_amber=None, alpha_red=None)` returns `(identity, reason)`, exactly one of them `None`. An identity
is `{"label", "kind", "kind_reason", "direction", "direction_reason", "alpha_amber", "alpha_red", "parameters", "version", "source"}`:

| Input | Source | Version | Parameters |
|---|---|---|---|
| a packaged label, `riskvalidation-default 2026.10.1` | `packaged-policy`: resolved through the packaged policy file (`default_policy`) | `2026.10.1` | none |
| an inline battery string, `contraste-ifrs9-battery-1.0.0; nl_amber=1.0; nl_red=3.0` | `inline-battery`: parsed | `1.0.0` | `{"nl_amber": 1.0, "nl_red": 3.0}` |
| anything else, such as `x` | none: `(None, reason)` | | |

- The **kind** is `spec(test_id).decision`, `p_value` or `metric` (W15-12); where the test is not registered (`spec` raises `KeyError`; at the
  batch 3 head the `ifrs9`, `stress`, `aggregation` and `portfolio` tests are not) it is `None` with its `kind_reason`; there is no fallback
  to a threshold. A label with the suffix `+gov-` (the governance policy, `gov_default_policy`) resolves through the `gov.` rows.
- The **direction** (`upper_bad` or `lower_bad`) is read from the policy file where one resolves; a p-value threshold has none, and a test with
  no packaged threshold has none, with the reason stated. For an inline battery string it is `None` with the reason "no policy file (BL-085)".
- A parameter value that is not a number stays a string (`rule=red-on-any`). The family of a test is the part of the id before the first dot.
- The report states the source of every threshold. A packaged policy file per battery is backlog item BL-085, done when the cases are next
  re-baked (U16). Until then an inline battery string is a **convention** the product wrote, not a file the engine can check.

## The validation report

`build_report(entry, inventory, register, variants, as_of)` returns the I.3 report of one inventory entry as a structured document that
validates against the schema `governance-report`: twelve sections, each **filled** from the inputs it names or **absent** with its reason. A
section is filled exactly when it names at least one input. The inputs are the inventory entry, the inventory document (for its provenance),
the register rows of the entry, the variant documents of the entry (the first is the default) and the as-of date. The function is
deterministic (the same inputs give a byte-identical document), never mutates its inputs and refuses an as-of date earlier than a raised or
validation date. The texts are English and Spanish.

| Section | Filled from | Notes |
|---|---|---|
| 1 Executive summary | the case question, the register rows, the test rows of the default variant | the light counts (`executive_counts`) and the opinion level of section 11 |
| 2 Model description and intended use | the models document, the inventory entry, the tier block | components, the tier under the four readings; intended use is missing as data |
| 3 Conceptual soundness | the models document (the rungs), the Methodology page | no written assessment or rating exists per case |
| 4 Data | the manifest sources, the provenance of the default variant | representativeness and bias are examined only by drift tests |
| 5 Implementation and process verification | seed, code version and engine version | absent when the variants record none |
| 6 Outcomes analysis | every test row of the default variant | each with its definition and its policy identity (`resolve_policy`) |
| 7 Sensitivity, robustness and stress | the variants as perturbations | test rows and red rows per variant |
| 8 Ongoing monitoring | the post-model adjustments (attribute A26) | absent with its reason when the entry holds none |
| 9 Limitations and compensating controls | the register rows with status `risk_accepted` | the stated limits of the synthetic case |
| 10 Findings register | the entry's register rows | key, severity, ECB F, status, due date, overdue |
| 11 Overall opinion and recommendation | `opinion_level` and `recommendation` over the open findings | with the label "the rule's output on the evidence; Contraste approves no model" |
| 12 Appendices | the sha256 of every document read for the entry | the reproducibility manifest |

`executive_counts(rows)` counts the lights `green`, `amber`, `red`, `not_evaluated` and `error` the way `SuiteRun.counts` does: for the default
variant of C18/banco-andino (17 rows) it is `{"green": 6, "amber": 2, "red": 6, "not_evaluated": 3, "error": 0}`. For the fixture entries the
opinion is level 4 for C18/banco-andino (an S1 finding is open, 29 open findings) and level 3 for C07/banco-andino (an S2 finding is open); the
report of C07 fills section 8 from the PMA overlays and the report of C18 states it absent.

No section states an approval, a certification or a compliance of a model (06 I.6), and none calls a convention a requirement of a regulator
or supervisor. Every section lists what is missing and why: the report is as honest about the evidence Contraste does not hold (conditions of use,
a written soundness assessment, a monitoring pack, a remediation plan) as about the evidence it does.

## Tests

`tests/governance/test_governance_policy.py`, `tests/governance/test_governance_report.py` (the goldens, the schema and the contract checker, the text rules) and
`tests/governance/test_governance_fixtures.py`, which rebuilds the two fixture reports field by field.
