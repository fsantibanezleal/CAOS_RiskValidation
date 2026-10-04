# 02. Thresholds are policy, not regulation

## What the primary sources say

- The ECB's *Instructions for reporting the validation results of internal models: IRB Pillar I models for
  credit risk* (February 2019) define the tests and statistics banks report (the Jeffreys test, the AUC
  comparison, migration and concentration statistics, LGD and CCF back-tests) but set no pass or fail thresholds
  ([ECB](https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf)).
- BCBS Working Paper 14, *Studies on the validation of internal rating systems* (revised May 2005), on
  calibration: "at present no really powerful tests of adequate calibration are currently available. Due to the
  correlation effects that have to be respected there even seems to be no way to develop such tests." It
  describes the binomial test as "kind of an early warning system", because its true type I error "can be much
  larger than the nominal level of the test if default events are correlated"
  ([BIS](https://www.bis.org/publ/bcbs_wp14.htm)).
- The population stability index bands (below 0.10 little change, 0.10 to 0.25 moderate, 0.25 or more
  significant) "are used without reference to statistical type I or type II error rates". Yurdakul and Naranjo
  show that, with sample sizes $n$ and $m$ and $B$ fixed bins, $(1/n + 1/m)^{-1}\,\mathrm{PSI}$ is approximately
  $\chi^2_{B-1}$; the 0.10 band has too large a type I error unless both samples exceed about 400, and the 0.25
  band is too conservative above about 200 (Yurdakul and Naranjo 2020, *Journal of Risk Model Validation*
  14(4), 89 to 100, [doi:10.21314/JRMV.2020.227](https://doi.org/10.21314/JRMV.2020.227)).

So a traffic light is an institution's choice. The package keeps every threshold in a versioned JSON policy,
records the policy on every result, and the packaged default states where each convention comes from.

## The three threshold kinds

| Kind | Reads | Red | Amber | Green |
|---|---|---|---|---|
| `p_value` | the p-value | $p < r$ | $r \le p < a$ | $p \ge a$ |
| `upper_bad` (larger is worse, for example PSI) | the metric | $m \ge r$ | $a \le m < r$ | $m < a$ |
| `lower_bad` (smaller is worse, for example AUC) | the metric | $m \le r$ | $r < m \le a$ | $m > a$ |

with amber cut $a$ and red cut $r$. A missing or non-finite value gives `not_evaluated`.

## Writing an institution's policy

```json
{
  "policy_id": "bank-x-credit",
  "version": "2027.01.0",
  "description": "Approved by the model risk committee on 2027-01-15.",
  "thresholds": {
    "pd.jeffreys": {"kind": "p_value", "amber": 0.05, "red": 0.01, "source": "internal policy MRM-12"}
  }
}
```

`load_policy(path)` reads it; pass it to any test or to `run_suite`.
