# 01. The result contract

Every test in the package returns one immutable `TestResult`. Validation is reviewed by people who were not in
the room when it ran (an approval committee, internal audit, a supervisor), so a result has to carry its own
explanation.

| Field | Why it exists |
|---|---|
| `test_id` | stable identity across releases; the key into the registry and the policy file |
| `h0`, `alternative` | what was tested; a p-value is meaningless without its null hypothesis |
| `statistic`, `p_value`, `metric` | the evidence; descriptive measures (AUC, PSI) use `metric` |
| `reference` | the primary source that defines the test (paper, supervisory document and section) |
| `policy_version`, `alpha_amber`, `alpha_red`, `light` | which policy judged the evidence, with which cut points, and the outcome |
| `n`, `n_events` | sample size and events; low counts change what a test can show |
| `model_id`, `segment` | what was tested |
| `inputs_hash` | a SHA-256 of the exact inputs, so the result is tied to one sample (see page 03) |
| `notes`, `extras` | caveats, intervals, critical values and other auxiliary numbers |

## Lights

| Light | Meaning |
|---|---|
| `green`, `amber`, `red` | the statistic fell in the policy's band |
| `not_evaluated` | the policy holds no thresholds for this test, or the statistic is undefined for the sample (for example no obligors in a grade); no colour is shown |
| `error` | the test raised; the exception is recorded in `notes` and the suite continues |

A result that is not an error must carry at least one of statistic, p-value or metric; the record rejects empty
evidence, empty references and p-values outside [0, 1] at construction.
