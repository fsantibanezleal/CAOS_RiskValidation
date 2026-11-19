# riskvalidation, software design document

Version: design of 0.04.000 (2026-10-06), extended unit by unit through 0.09.000 and the public API of 1.0.0 (U16, 2026-10-21). Requirements with named gates live in `docs/design/features/<unit>/`; each
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
| IRB, LHP, low-default and calibration methods (U2) | BCBS CRE99 Table 1, the Pluto-Tasche tables and Tasche (2013) Tables 2 to 9 reproduced from their inputs (RV-201 to RV-214). |
| Known-truth generators and the size/power harness (U3) | each generator reproduces its exact truth on its own stream; the exact rejection probabilities agree with the simulation; BCBS WP14 Tables 7 and 8 (143 of 144 cells), Yurdakul and Naranjo Table 4 (53 of 54) and Demler et al. (2012) reproduced, every cell that is not listed with its evidence; every p-value test's size measured, a test above the bound documented (RV-301 to RV-316). |
| Rating transitions (U4) | the cohort, duration, Aalen-Johansen and EM estimators recover a known generator; the embedding diagnostics and adjustments reproduce Israel, Rosenthal and Wei's nine distances (eight from the printed matrices, the ninth from the printed JLT generator); the time-homogeneity, order, reference-matrix and momentum tests hold their size in the form chosen by measurement, and their power against momentum and a regime is measured; Schuermann and Hanson's Table 5 and Engelmann's section 4 reproduce to the printed digits (RV-401 to RV-414). |
| Provisions (U5), LGD/CCF (U6), portfolio models (U7), market (U8), regulatory calculators (U9), IRRBB/ALM (U10), liquidity (U11), operational risk (U12), aggregation and stress (U13), ML and fairness tests (U14), governance (U15) | the golden values listed in each unit's feature SDD are reproduced, and every displayed regulatory parameter carries its paragraph reference and in-force date. |
| E-backtests of VaR and expected shortfall (U20) | the e-statistics are the paper's equations (1) and (3) on hand-computed values and with expectation 1 by quadrature at the true VaR and ES; the VaR e-process equals the Contraste manuscript's reference to 1e-12; the betting processes GREE, GREL and GREM equal loops written from equations (13) to (15); the mean of the e-process is exactly 1 by enumeration, its size at any stopping time is held and a planted under-forecast is detected (RV-2101 to RV-2115). |

## 5. Cases

The package's cases are its fixtures: published worked examples (the strongest oracle) and seeded generator
scenarios with known parameters. Each fixture records its source and the tolerance it is reproduced to.

## 6. Oracles

1. Published worked examples and tables from the primary source of each method.
2. Independent reference computations (SciPy distributions, exact enumeration on small cases).
3. Known-truth generators: estimators recover parameters; tests hold their size under the null.
4. Identities and invariants (row sums, Euler allocations, reconciliation roll-forwards).

## 7. Deploy driver

A Python library published to PyPI by OIDC trusted publishing on a GitHub release (ADR-0061). The package has been on
PyPI since 0.8.0. Driver: the consuming product installs a pinned version, and the browser lane needs a pure wheel. From
1.0.0 the public API is the one in `docs/api-policy.md`, under semantic versioning, and the wheel ships `py.typed`.

## 8. Risks and kill criteria

| Risk | Criterion |
|---|---|
| A test cannot reproduce its published worked example | it is not shipped until it does |
| A dependency beyond NumPy and SciPy enters the core | RV-006 fails the build |
| Thresholds drift into code | RV-004 and review of every `make_result` call |
| A primary source is ambiguous (for example the ECB generalised-AUC standard deviation scaling) | the ambiguity is documented on the test's page and in its result notes, not silently resolved |
