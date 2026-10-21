# 05. The IFRS 9 validation battery

Module: `riskvalidation.validation.ifrs9`. Source: research dossier 02, sections I.2 and I.5; contract
`engine-provisions-api` 1.0.0.

Each test returns one `TestResult` of the engine: `test_id`, `h0`, `alternative`, `reference`, `policy_version`, `light` and
the statistic, p-value or metric. The acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds:
each function states its light rule, and its parameters travel in `policy_version` (and in `alpha_amber` and `alpha_red`
where a p-value or a band is used) so that a view can explain a light. A test that cannot be evaluated returns the light
`not_evaluated` with its reason in `notes`; its statistic is undefined (NaN, serialised as null), never a fabricated number.
The tests do not read the default policy file: their rules are the contract's.

| `test_id` | Question | Metric | Light rule (Contraste internal gate) | Reference |
|---|---|---|---|---|
| `ifrs9.lifetime_pd_backtest` | does the observed cumulative default follow the predicted lifetime PD? | largest absolute gap over observed horizons | with $\rho$: red outside the Vasicek band at $\alpha = 0.01$, amber at 0.05; without: worst relative gap, green below 0.25, amber below 0.50 | EBA/Rep/2023/36 paras 114 to 118 |
| `ifrs9.sicr_effectiveness` | does the Stage 2 flag select the accounts that default? | $DR_{S2} / DR_{S1}$ | green at 2.0 or more, amber from 1.0, red below; amber if more than half of Stage 3 arrivals came directly from Stage 1 | EBA/Rep/2023/36 paras 94 to 99; d350 A38 |
| `ifrs9.sicr_threshold_stability` | does the threshold loosen when conditions worsen? | periods with a rising multiplier and a rising condition index | 0 green, 1 or more amber, red from half of the periods | EBA/Rep/2023/36 paras 29 to 33 |
| `ifrs9.stage_migration_backtest` | do predicted stage transitions match the observed ones? | chi-square statistic and p-value | green from $p = 0.05$, amber from 0.01, red below | EBA/Rep/2023/36 para 97 |
| `ifrs9.ecl_backtest` | do discounted realised losses match the ECL? | $R_H = \sum \mathrm{PV(realised)} / \sum \mathrm{ECL}(H)$ | green 0.8 to 1.25, amber 0.5 to 2.0, red outside | IFRS 9 B5.5.52; EBA/Rep/2023/36 paras 100 to 104 |
| `ifrs9.scenario_sensitivity` | how non-linear is the ECL in the scenarios? | $NL$ | green $0 \le NL \le 1$, amber above 1 or below 0, red above 3 | d350 para 36(a); GPPC 2.8.1.2 |
| `ifrs9.nonlinearity` | the same at PD level, with a numerical cross check | $NL$ of the PD | green if the Gauss-Hermite error is at most 1e-9 and $NL \ge 0$, red otherwise | dossier 02 B.4, C.1 |
| `ifrs9.overlay_diagnostics` | how large and how consistent are the overlays? | overlay share of the total allowance | green up to 0.10 and direction consistent, amber to 0.25 or inconsistent, red above | ECB July 2024; d350 paras 50, 51 |
| `ifrs9.reconciliation_identities` | do the IFRS 7 35H and 35I tables close? | largest absolute residual | green if every identity closes within $10^{-6}\max(1, G)$, red otherwise | IFRS 7 35H, 35I |

## Lifetime PD backtest

Per origination cohort and horizon $k$: the observed Kaplan-Meier cumulative default $\hat F(k)$ against the predicted
$\bar F(k)$ (scenario-weighted); the gap $\hat F(k) - \bar F(k)$ and the Brier score by horizon.

$$\mathrm{Brier}(k) = \hat F(k)\,\bigl(1 - \bar F(k)\bigr)^2 + \bigl(1 - \hat F(k)\bigr)\,\bar F(k)^2.$$

*Equation 1. The Brier score of a predicted cumulative PD against the observed default share, from aggregates.* When
prepayment competes, Kaplan-Meier would treat prepayment as censoring and estimate $1 - S_D$, not the incidence; the
observed series is then the Aalen-Johansen cumulative incidence and the prediction comes from `competing_risks` (the result
names its `estimator`). The Vasicek acceptance band (Vasicek 2002) replaces the binomial one so that systematic default-rate
swings are not mistaken for miscalibration: it is the central interval of the default count among `n_at_risk` obligors whose
default probability is the predicted PD with asset correlation $\rho$, taken as a rate, two-sided with $\alpha/2$ in each tail
(`band_low`, `band_high` at 0.05 and `band_low_red`, `band_high_red` at 0.01). `extras` carries `gap`, `brier` and the bands per
observed horizon; unobserved horizons are excluded and none observed gives `not_evaluated`.

## SICR effectiveness and threshold stability

The default-rate ratio $DR_{S2}/DR_{S1}$ is the metric (log-scale interval). `extras` carries the ROC AUC of the stage flag
(ties one half, with an interval from the ECB variance), the hit rate (the share of defaulters at least a minimum number of
months in Stage 2 before default), the share of Stage 3 arrivals directly from Stage 1, the mean months in Stage 2 before
default, and the share of Stage 2 that is there by the backstop only; proportions carry Wilson intervals. A reference with no
Stage 1 defaults returns `not_evaluated`. Golden value: Stage 1 1000 accounts with 10 defaults, Stage 2 100 accounts with 30
defaults give a ratio of 30, green. The EBA expects thresholds not to loosen when conditions worsen: the threshold stability
test counts the periods where the implied multiplier rises while the condition index rises (golden: multipliers $[2.0, 2.5]$,
index $[0, 1]$ gives 1, not green).

## Stage migration and ECL backtests

$$\chi^2 = \sum_{i}\sum_{j:\,p_{ij}>0} \frac{\bigl(O_{ij} - n_i\,p_{ij}\bigr)^2}{n_i\,p_{ij}}, \qquad \mathrm{df} = \sum_{i:\,n_i > 0}\bigl(\#\{j: p_{ij} > 0\} - 1\bigr).$$

*Equation 2. The chi-square on the stage transition counts (Anderson and Goodman 1957, equation (3.1), summed over rows).*
Symbols: $O_{ij}$ are the observed counts, $n_i$ the row totals and $p_{ij}$ the predicted row-stochastic matrix; rows with no
observations are skipped and counted in `extras`. Counts exactly proportional to the predicted matrix give $\chi^2 = 0$, $p =
1$, green.

$$R_H = \frac{\sum_i \mathrm{PV}(\text{realised loss}_i)}{\sum_i \mathrm{ECL}_i(H)}.$$

*Equation 3. The like-for-like ECL backtest ratio.* For a reporting-date cohort, discounted realised losses within horizon $H$
against the ECL for the same accounts, overlays excluded (IFRS 9 B5.5.52). Golden value: realised 90 against ECL 100 gives
0.9, green. A zero ECL gives `not_evaluated`.

## Scenario sensitivity, non-linearity, overlays and reconciliation

`scenario_sensitivity` returns $NL$ ([page 03](03_measurement.md), Equation 4) with the 100 per cent single-scenario ECLs and the
weight sensitivity against each reference scenario in `extras`; they are sensitivities, never the IFRS 9 measure. Golden
value: scenario ECLs $0.4 \times (0.005255059, 0.014287387, 0.073424082)$, weights $(0.25, 0.5, 0.25)$, ECL at the mean path
$0.4 \times 0.017997448$ give $NL = 0.4898489$ (to 1e-6), green. `nonlinearity_test` gives the same index at PD level and the
Gauss-Hermite error (golden: weighted 0.026813479, at the mean factor 0.017997448, each to 1e-8).

`overlay_diagnostics` reports the overlay share of the total allowance (the model allowance by stage plus the overlays), the
share by stage and by segment (an overlay with no stage or segment under `portfolio`), the amounts by target, the
parameter-level (targets `pd`, `lgd`, `ead`), staging-level (`staging-share`) and total-level (no segment) split, and a
`direction_consistent` flag: a positive overlay in an improving scenario is flagged. Golden value: overlays 5 against model ECL
75 give 0.0625, green.

`reconciliation_identities` reports the largest absolute residual of the roll-forward identities and lists the failing
identity names; the stage closing against the stage ECL (the overlays carried in a stage read from the waterfall) is judged on
the same tolerance and its largest gap is `stage_ecl_gap_max`. Golden value: the C06 reporting-date fixture gives a metric of
0 (to rounding) and green.

## Size and power of the p-value tests

The p-value tests of this page, measured on a generator where each null holds at the boundary (U16-E2, BL-088; the harness is `riskvalidation.harness`, the null and the alternative of each are named in `riskvalidation.generators.size_studies`, the runs are the `sizepower` suites under `tests/`, a local release gate that CI deselects). Size is the rejection rate at the nominal level over 4,000 repetitions, accepted when it is within the binomial bound alpha plus 3.09 standard errors; power is the rejection rate at 5% against one stated alternative, with no pass threshold. The figures below are the numbers of the run; the suite fails when a page states other ones.

**Size and power of `ifrs9.stage_migration_backtest`** (three stages, 4000, 1500 and 500 accounts; seed 5202, 4000 repetitions). Assumptions met: independent accounts, one multinomial draw per stage row from the predicted matrix (rows of 4,000, 1,500 and 500 accounts), so the smallest expected cell count is 10, the chi-square approximation's assumption. The measured size is 5.15% at 5% (bound 6.06%) and 1.12% at 1% (bound 1.49%). Power at 5% against the alternative that the stage 1 to stage 2 migration is 9.5%, not the predicted 8% (stage 1 staying 88.5%) (seed 5302, 4000 repetitions): 72.88%.

## What must not be claimed

These limits come from dossier 02, section I.5, and bind every use of this page and of the engine:

- That any synthetic portfolio, parameter set, scenario or overlay reflects any bank, any supervisor's view or any actual
  disclosure.
- That multipliers, absolute floors, probation lengths, behavioural lives, mean-reversion schedules or acceptance gates are
  IFRS 9 requirements. Only the 30 days past due (5.5.11) and 90 days past due (B5.5.37) rebuttable presumptions are in the
  standard; the threefold backstop is an EBA supervisory expectation (EBA/Rep/2023/36 para 32), not IFRS 9 and not law.
- That the engine is an IFRS 9 compliant or audited implementation for any entity, or that its outputs are IFRS 7 disclosures:
  the engine computes and makes no compliance claim.
- That Chilean bank loan provisions are IFRS 9 ECL (they follow CMF CNCB Chapters B-1 to B-3).
- That overlays are validated models, or that a model "captures" climate risk when climate only enters through GDP paths.
- That a backtest on one dataset validates a model for any other market.

## References

- EBA (2023). IFRS 9 implementation by EU institutions, 2023 Monitoring report, EBA/Rep/2023/36. doi:10.2853/269836.
- EBA (2021). IFRS 9 implementation by EU institutions, Monitoring report, EBA/Rep/2021/35. doi:10.2853/631076.
- BCBS (2015). Guidance on credit risk and accounting for expected credit losses (d350). https://www.bis.org/bcbs/publ/d350.htm.
- GPPC (2016). The implementation of IFRS 9 impairment requirements by banks.
  https://www.globalpublicpolicycommittee.org/wp-content/uploads/2022/12/gppc-ifrs9-implementation-considerations-20160617.pdf.
- ECB Banking Supervision (2024). IFRS 9 overlays and model improvements for novel risks. doi:10.2866/034127.
- ECB Banking Supervision (2019). Instructions for reporting the validation results of internal models: IRB Pillar I models for
  credit risk, sections 2.5.3.1 and 2.5.4.
  https://www.bankingsupervision.europa.eu/banking/tasks/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf.
- Anderson, T.W. and Goodman, L.A. (1957). Statistical inference about Markov chains. Annals of Mathematical Statistics 28(1),
  89-110. doi:10.1214/aoms/1177707039.
- Vasicek, O. (2002). The distribution of loan portfolio value. Risk 15(12), December 2002.
- Tasche, D. (2003). A traffic lights approach to PD validation. arXiv:cond-mat/0305038.
- IASB, IFRS 9 B5.5.52 and IFRS 7 35H, 35I, as endorsed by Commission Regulation (EU) 2016/2067.
  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R2067.

Previous: [04. Overlays and reconciliation](04_overlays-and-reconciliation.md). Index: [README](README.md).
