# 06. The validation tests

Module `riskvalidation.validation.ccr`. Sources: dossier 05, the counterparty ladder (acceptance: the Python EE matches ORE within Monte Carlo error at each grid point, EEPE recomputed
from EE, CVA recomputed, convergence in the number of paths, exposure backtesting of PFE exceedances); the U9 research addendum R5 (BCBS 185, CRE53.33, Formenti et al.) and R9 (CRE53.12,
CRE53.13). Each function returns one `TestResult` whose `test_id` is prefixed `ccr.` and whose row validates against `common/test-result`. Acceptance gates are Contraste internal gates, stated
as such, not regulatory thresholds. A test that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value once serialised, never
a placeholder number.

| Test id | What it checks | Light |
|---|---|---|
| `ccr.ee_analytic_vs_simulated` | the analytic EE of a single swap against the simulated one at the reset dates; statistic the largest abs(analytic - simulated) / mc_se, metric the largest absolute difference | green within `k_se` (3), red beyond |
| `ccr.martingale_discounted_bonds` | the sample mean of the discounted bond price against $P(0,T)$; statistic the largest abs(mean - P(0, T)) / mc_se | green within `k_se` (3), red beyond |
| `ccr.eepe_recomputed` | the reported EEPE against the effective EPE recomputed from EE; statistic the absolute residual, metric the relative one | green at most `rel_tol` (1e-9) |
| `ccr.cva_recomputed` | the reported CVA against sum PD LGD EPE; statistic the absolute residual, metric the relative one | green at most `rel_tol` (1e-9) |
| `ccr.path_convergence` | the change of the EE profile and of the CVA between the two largest path counts, against the Monte Carlo standard errors the caller passes (`ee_se_by_path_count`, `cva_se_by_path_count`, keyword arguments); statistic the largest abs(change in EE) / se of the smaller count, metric abs(change in CVA) / its se | green within `k_se` (3) standard errors of the smaller count, amber within `2 k_se`, red beyond; `not_evaluated` for fewer than two counts |
| `ccr.reconcile_ee_profile` | the engine's EE profile against a reference (ORE's) within the combined error; `extras["agreement"]` lists the per-point difference and combined error | green within `k_se` (2) combined errors, red beyond |
| `ccr.pfe_exceedance_backtest` | the exceedances of the PFE forecast by the realised mark-to-market; statistic the Kupiec likelihood ratio, `p_value` its chi-square(1) value | green at $p\ge0.05$, amber below 0.05, red below 0.01 |
| `ccr.pit_exposure_backtest` | the PITs of the realised values in the simulated distributions; statistic the Anderson-Darling $W^2$ | green at or below the simulated 5 percent value, amber to the 1 percent value, red above; `not_evaluated` without simulated critical values |

## The differences within the Monte Carlo error

For `ee_analytic_vs_simulated`, `martingale_discounted_bonds` and `reconcile_ee_profile` the statistic is the largest of $|x-y|/se$ over the grid (for the reconciliation, the combined error
$\sqrt{se_x^2+se_y^2}$). A point with no Monte Carlo error counts 0 for a zero difference and infinity for any other (the EE after the maturity is zero with zero error). `extras["thresholds"]`
carries `k_se`. The analytic EE and the simulated one are on the same measure: the analytic identity of page 03 holds for the risk-neutral measure only, so the discounted simulated EE is compared
with $P(0,t)$ times the analytic profile.

## EEPE and CVA recomputed

The recomputation uses `exposure.effective_ee` from a zero current exposure and `exposure.effective_epe` over the first year, and `cva.cva`. Golden: grid `[0.25, 0.5, 1.0]` and EE `[1, 3, 2]` with a
reported 2.5 is green; a reported 2.0 is red (2.0 is the plain EPE, not the effective one). The `cva` golden (1.8) reported as 1.8 is green, 1.9 is red.

## Path convergence

The statistic is the largest absolute change of the EE profile between the two largest path counts divided by the peak of the larger count's profile; the metric is the change of the CVA relative to the
larger count's CVA. The inputs carry no standard error, so the relative Monte Carlo error implied by the smaller count $n$ is taken as $1/\sqrt n$, a unit coefficient of variation: this is a convention
of the test (an EE of an at-the-money swap has a coefficient of variation near 1.5 at its peak, so the green band is about two standard deviations of the change). UNVERIFIED as a standard. Green when the
larger of the two changes is within `k_se` times that error, amber within twice, red beyond. Golden: identical profiles at every count are green with statistic 0; fewer than two path counts is
`not_evaluated` with the reason.

## The PFE exceedance backtest

An exceedance is a realised value strictly above its forecast. With $N$ dates, $x$ exceedances and the expected rate $p=1-q$, the Kupiec likelihood ratio of unconditional coverage is

$$LR=-2\ln\!\big[(1-p)^{N-x}p^{x}\big]+2\ln\!\Big[\big(1-\tfrac xN\big)^{N-x}\big(\tfrac xN\big)^{x}\Big],\qquad p\text{-value}=\Pr[\chi^2_1>LR]$$

Golden: $N=100$, $q=0.95$ and 5 exceedances give $LR=0$ and green; 0 exceedances give $LR=-2N\ln q=10.2587$ and red. BCBS 185 paragraph 27 (a caveat the row carries in `extras["caveat"]`):
criteria based on exception counting often assume independence, and the data may overlap, so a count over dates closer than the horizon is not an independent sample. Backtest several horizons and
several quantiles (paragraph 31), never one (BCBS 185 paragraphs 18 to 25 and CRE53.33).

## The PIT-based exposure backtest

For a backtest date with $P$ simulated values $\hat r^{(1)}<\dots<\hat r^{(P)}$ at the horizon and the realised value $r$, the probability integral transform is (Formenti, Spadafora, Terraneo and Ramponi,
arXiv 1505.04593, page 2)

$$F(r)=\begin{cases}\frac1{P+2},&r<\hat r^{(1)}\\ \frac{i+1}{P+2},&\hat r^{(i)}<r<\hat r^{(i+1)}\\ \frac{P+1}{P+2},&r>\hat r^{(P)}\end{cases}$$

`pit_from_simulated(realised, simulated)` counts the simulated values strictly below the realised one. The PITs over the backtest dates should be uniform if the model matches the realised values (BCBS 185
paragraph 25). The statistic is the Anderson-Darling distance (equations 1 and 2)

$$W^2=-n-\sum_{k=1}^n\frac{2k-1}{n}\Big[\ln F(x_k)+\ln\big(1-F(x_{n+1-k})\big)\Big]$$

`anderson_darling_w2(pits)`; golden: the PITs $0.25$, $0.5$, $0.75$ give $W^2=0.269431$. The iid table values for the upper-tail probabilities 0.25, 0.15, 0.10, 0.05, 0.01, 0.005, 0.001 are 1.248, 1.610,
1.933, 2.492, 3.880, 4.500, 6.000 (Table 1 of the paper, from Anderson and Darling; `IID_CRITICAL_VALUES`). They hold for independent PITs, so the rule is not the iid table: the critical values come
from simulating the whole backtest under the model itself, at the same dates and with the same overlapping horizons (B5). `simulate_pit_critical_values(pit_sampler, n_backtests, seed, max_lag=None)`
takes a sampler that returns the PITs of one whole simulated backtest, and returns `CriticalValues` (`p05`, `p01`, `effective_sample_size`, `n_backtests`, `w2_samples`): the 0.95 and 0.99 inverse
empirical quantiles of $W^2$ over `n_backtests` (at least 100) statistics, the effective sample size of the first simulated series, reproducible for the seed. A sampler of independent uniforms returns
`p05` near 2.492 and `p01` near 3.880 (the test accepts a band of 0.15 and 0.35 at 4,000 backtests).

`pit_exposure_backtest(realised, simulated, critical_simulated=None, effective_sample_size=None)`: without `critical_simulated` the result is `not_evaluated` with the reason "no simulated critical
values" and a null statistic, p-value and metric (the iid table is reported beside a simulated rule, never in its place, and no placeholder number stands in). With `{"p05": ..., "p01": ...}`: green at or
below the simulated 5 percent value, amber to the 1 percent value, red above; `extras` carry the PITs, the iid values, the simulated ones and the effective sample size. Under the model itself the
rejection rate at the 5 percent value is the nominal level (600 simulated backtests of 60 dates, within three binomial standard errors), and a realised path with half the forecast sigma is rejected
at the 1 percent value in at least 90 percent of backtests (200 backtests).

### The effective sample size

BCBS 185 paragraph 25 allows overlapping horizons without presuming independence and requires only that the effective size be stated. `effective_sample_size(pits, max_lag=None)` uses
$z_i=\Phi^{-1}(PIT_i)$ and the sample autocorrelations $\rho_k$ of $z$:

$$\mathrm{ESS}=\frac{n}{1+2\sum_{k=1}^{L}\big(1-\tfrac{k}{L+1}\big)\rho_k},\qquad L=\max\_lag\ \text{(default the integer part of }n^{1/3}),\quad \mathrm{ESS}\in[1,n]$$

Bartlett weights. Uncorrelated PITs give about $n$ and fully overlapping horizons give less. Golden: $z=[1,0,-1,0]$ has $\rho_1=0$ and with `max_lag = 1` gives ESS 4. This is a convention of the
contract, UNVERIFIED as a backtesting standard. Equal PITs (no variance) are one observation.

## Size and power of the p-value tests

The p-value tests of this page, measured on a generator where each null holds at the boundary (U16-E2, BL-088; the harness is `riskvalidation.harness`, the null and the alternative of each are named in `riskvalidation.generators.size_studies`, the runs are the `sizepower` suites under `tests/`, a local release gate that CI deselects). Size is the rejection rate at the nominal level over 4,000 repetitions, accepted when it is within the binomial bound alpha plus 3.09 standard errors; power is the rejection rate at 5% against one stated alternative, with no pass threshold. The figures below are the numbers of the run; the suite fails when a page states other ones.

**Size and power of `ccr.pfe_exceedance_backtest`** (PFE at 95%, 1000 dates; seed 5201, 4000 repetitions). Assumptions met: independent dates with a date-specific scale, realised values normal and the forecast the true 95% quantile, so the exceedances are independent Bernoulli(5%) draws (about 50 per run), the regime where the Kupiec chi-square(1) law is accurate. The measured size is 4.88% at 5% (bound 6.06%) and 1.00% at 1% (bound 1.49%). Power at 5% against the alternative that the PFE forecast is 90% of the true 95% quantile (exceedance frequency 6.94%, not 5%) (seed 5301, 4000 repetitions): 71.30%.

## Limits

- The three tests that compare with a reference (`ee_analytic_vs_simulated`, `reconcile_ee_profile` and `martingale_discounted_bonds`) say whether the engine agrees with a closed form or with ORE within the
  Monte Carlo error; they say nothing about whether the model is right for the market.
- The Anderson-Darling critical values for the overlapping-horizon backtest come from simulating the backtest under the model; the Ruiz (2014) and Anfuso et al. (2014) procedures were not read (UNVERIFIED), and the
  work of Kyung and Tsumagari on overlapping data is a lead, not a source. The test is "more sensitive with respect to the tails" and has low power for small $n$ (the paper needs about 50 observations for a
  coefficient of variation below 10 percent).
- A green light is a statistic inside a band, not a statement that the model is accepted or compliant; the engine states no compliance claim.
