# riskvalidation, software design document

Version: 0.01.000 design, 2026-10-04. Requirements with named gates live in `docs/design/features/<unit>/`; each
unit's requirements are written before its code and its `tasks.md` closes with a convergence verdict.

## 1. Problem and non-goals

An independent validator of financial risk models re-implements three things the model owner also uses: the
validation tests (discrimination, calibration, stability, backtesting, robustness, fairness), the regulatory
formulas that turn a model into capital, provisions or liquidity metrics, and reference or challenger engines to
benchmark the model. Each needs its exact definition, its primary source and a reproducible result. No
maintained open-source library implements, for example, the ECB's 2019 IRB validation-reporting tests as
specified, a permissive CreditRisk+ or an open bank ALM engine. `riskvalidation` is that toolkit.

Non-goals:

1. Deciding that a model is compliant or approved. A test result is evidence under a stated policy; the light is a
   policy choice, never a regulatory verdict.
2. Fitting heavy machine-learning models. Gradient boosting, survival forests or neural challengers are trained
   by the consuming application; this package validates their outputs.
3. Market data access, data storage or a database of models.
4. Proprietary methodologies of any institution. Published methods are re-implemented from their primary sources
   and cited.

## 2. Contracts

- **`TestResult`** (`riskvalidation.core.result`): one immutable record per test, model, segment and sample:
  `test_id`, `h0`, `reference`, `policy_version`, `light`, at least one of `statistic`, `p_value`, `metric`, plus
  `alternative`, thresholds applied, `n`, `n_events`, `model_id`, `segment`, `inputs_hash`, `notes`, `extras`.
  `to_dict()` is JSON-safe (non-finite numbers become null).
- **Policy file** (`riskvalidation.core.policy`): JSON with `policy_id`, `version`, `description` and
  `thresholds[test_id] = {kind, amber, red, source}`; `kind` in `p_value`, `upper_bad`, `lower_bad`. Thresholds
  never appear in test code.
- **Registry** (`riskvalidation.core.registry`): every test registers `test_id`, `family`, `name`, `h0`,
  `reference` and its `decision` quantity.
- **Inputs**: NumPy arrays and Python scalars; no data frames in the core.

## 3. Lanes

The core imports NumPy and SciPy only (gate RV-006), so the same wheel runs offline in a Python pipeline and live
in the browser under Pyodide, where NumPy and SciPy ship with the distribution. There is no service lane.

## 4. Method ladder and acceptance

| Method family (unit) | Accepted when |
|---|---|
| Core contracts (U0) | RV-001 to RV-007 pass. |
| PD calibration and discrimination, rating-system and stability tests (U1) | each test reproduces a published worked example (BCBS WP14 critical values, ECB 2019 formulas, Yurdakul-Naranjo benchmarks) or an independent SciPy reference to a stated tolerance. |
| IRB, LHP, low-default and calibration methods (U2) | BCBS CRE99 Table 1, Pluto-Tasche tables and Tasche (2013) Table 1 reproduced. |
| Known-truth generators and the size/power harness (U3) | each generator recovers its parameters; each test holds its size under the null within Monte Carlo error. |
| Transition estimators (U4), provisions (U5), LGD/CCF (U6), portfolio models (U7), market (U8), regulatory calculators (U9), IRRBB/ALM (U10), liquidity (U11), operational risk (U12), aggregation and stress (U13), ML and fairness tests (U14), governance (U15) | the golden values listed in each unit's feature SDD are reproduced, and every displayed regulatory parameter carries its paragraph reference and in-force date. |

## 5. Cases

The package's cases are its fixtures: published worked examples (the strongest oracle) and seeded generator
scenarios with known parameters. Each fixture records its source and the tolerance it is reproduced to.

## 6. Oracles

1. Published worked examples and tables from the primary source of each method.
2. Independent reference computations (SciPy distributions, exact enumeration on small cases).
3. Known-truth generators: estimators recover parameters; tests hold their size under the null.
4. Identities and invariants (row sums, Euler allocations, reconciliation roll-forwards).

## 7. Deploy driver

A Python library published to PyPI by OIDC trusted publishing on a GitHub release (ADR-0061). The repository is
private until the first publish and public from then on. Driver: the consuming product installs a pinned version,
and the browser lane needs a pure wheel; publishing waits for the one-time PyPI pending-publisher registration.

## 8. Risks and kill criteria

| Risk | Criterion |
|---|---|
| A test cannot reproduce its published worked example | it is not shipped until it does |
| A dependency beyond NumPy and SciPy enters the core | RV-006 fails the build |
| Thresholds drift into code | RV-004 and review of every `make_result` call |
| A primary source is ambiguous (for example the ECB generalised-AUC standard deviation scaling) | the ambiguity is documented on the test's page and in its result notes, not silently resolved |
