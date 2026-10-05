# 01. PD calibration

Calibration asks whether the default rates that occurred are consistent with the PDs the model assigned. The
honest starting point is BCBS Working Paper 14 (2005): "at present no really powerful tests of adequate
calibration are currently available. Due to the correlation effects that have to be respected there even seems
to be no way to develop such tests" ([BIS](https://www.bis.org/publ/bcbs_wp14.htm)). The tests below therefore
differ in what they assume, and a validator reads them together.

![Calibration tests by scope and dependence assumption](../assets/calibration-map.svg)

Notation: a grade (or portfolio) has $N$ obligors at the start of a one-year period, $D$ defaults during it, and
PD applied $\mathrm{PD}$; across grades $i$ these are $n_i$, $d_i$, $p_i$.

## Jeffreys test (`pd.jeffreys`)

The test the ECB prescribes for reporting IRB PD back-testing, per grade and for the portfolio ($K+1$ tests)
(ECB, *Instructions for reporting the validation results of internal models: IRB Pillar I models for credit
risk*, February 2019, section 2.5.3.1). With the Jeffreys prior $\mathrm{Beta}(1/2, 1/2)$ the posterior of the
default probability is $\mathrm{Beta}(D + 1/2,\, N - D + 1/2)$ and

$$p = F_{\mathrm{Beta}(D+\frac12,\,N-D+\frac12)}(\mathrm{PD}),$$

the posterior probability that the true PD is below the PD applied.

- $H_0$: the PD applied at the start of the period is greater than or equal to the true PD.
- Reject (PD underestimated) when $p < \alpha$.
- Source of the interval logic: Brown, Cai and DasGupta (2001), *Interval estimation for a binomial
  proportion*, Statistical Science 16(2), 101 to 117, [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286).
- Implementation choice: the portfolio PD is the obligor-weighted average of the grade PDs (the average PD across
  customers). The result also reports the posterior 2.5% and 97.5% quantiles.

## Binomial test (`pd.binomial`)

Under independence $D \sim \mathrm{Bin}(N, \mathrm{PD})$ and the one-sided p-value is
$p = P(\mathrm{Bin}(N,\mathrm{PD}) \ge D)$. WP14 warns that the true type I error "can be much larger than the
nominal level of the test if default events are correlated" and suggests reading the test as "kind of an early
warning system".

## Vasicek-corrected binomial test (`pd.binomial_vasicek`)

WP14's appendix corrects for correlation with the one-factor model: asset value
$R_i = \sqrt{1-\rho}\,\varepsilon_i + \sqrt{\rho}\,X$ with independent standard normals $X$, $\varepsilon_i$,
default when $R_i \le \Phi^{-1}(\mathrm{PD})$. Conditional on $X = x$ defaults are independent with PD

$$\mathrm{PD}(x) = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}) - \sqrt{\rho}\,x}{\sqrt{1-\rho}}\right),$$

so the default count is a binomial mixture:

$$P(D \ge k) = \int_{-\infty}^{\infty} P\big(\mathrm{Bin}(N, \mathrm{PD}(x)) \ge k\big)\,\varphi(x)\,dx .$$

The package evaluates the integral by Gauss-Hermite quadrature with 256 nodes. For large $N$ the critical
default rate at confidence $q$ is approximately
$\Phi\big((\Phi^{-1}(\mathrm{PD}) + \sqrt{\rho}\,\Phi^{-1}(q))/\sqrt{1-\rho}\big)$, reported as
`large_n_critical_count`.

**Critical counts, and a reconciliation.** WP14 tabulates, for PD 1%, $N = 1{,}000$ and $q = 99\%$, critical
counts of 19, 35, 49, 63 and 77 for $\rho$ = 0%, 5%, 10%, 15% and 20%. These are reproduced exactly by
$k^* = \min\{k : P(D \ge k) \le 1 - q\}$, with rejection when $D \ge k^*$; the 99% quantile of $D$ itself is one
lower (18, 34, 48, 62, 76). The package uses the definition that reproduces the published table (test
`test_wp14_critical_values`). The asset correlation is a modelling assumption that must be documented; the IRB
supervisory correlations are one option.

## Chi-square test over grades (`pd.chi2_grades`)

$$T = \sum_{i}\frac{(n_i p_i - d_i)^2}{n_i p_i (1 - p_i)},$$

asymptotically $\chi^2$ with as many degrees of freedom as grades under $H_0$ (every grade PD is the true one,
defaults independent within and between grades). WP14: likely to understate the true type I error under default
dependence, and slow to converge for small PDs.

## Chi-square test of the default profile (`pd.default_profile`)

The test Tasche (2013) uses to rank PD curve calibrations
([doi:10.21314/jcr.2013.169](https://doi.org/10.21314/jcr.2013.169), sections 3.3 and 4.2.5). Given the number of
defaults $D = \sum_i d_i$, the default numbers by grade are multinomial with the implied default profile
$q_i = n_i p_i / \sum_j n_j p_j$ under $H_0$, and Pearson's statistic is

$$T = \sum_{i}\frac{(d_i - D\,q_i)^2}{D\,q_i}.$$

- $H_0$: given their number, the defaults fall on the grades as the PD curve implies. The test sees the **shape** of
  the curve, not its level; read it with a level test such as `pd.binomial`.
- p-value: Monte Carlo, $(1 + \#\{T^* \ge T\})/(B + 1)$ over $B$ = `n_sim` multinomial samples drawn with `seed`
  (both recorded), because expected counts in good grades are far below the chi-square rule of thumb. The Monte Carlo
  standard error and the asymptotic p-value ($k - 1$ degrees of freedom) are kept in `extras`.
- Reproduces the p-values of Tasche (2013) Table 7 on the S&P data within their Monte Carlo error (see
  [PD curve calibration](../engines/03_pd-curve-calibration.md)).

## Hosmer-Lemeshow test (`pd.hosmer_lemeshow`)

Obligors are grouped by quantiles of predicted PD into $G$ groups with observed defaults $O_g$, expected
defaults $E_g = \sum_{j \in g} p_j$ and size $m_g$:

$$\mathrm{HL} = \sum_{g=1}^{G}\frac{(O_g - E_g)^2}{E_g\,(1 - E_g/m_g)} .$$

The original statistic, for a logistic model fitted on the same data, is compared with $\chi^2_{G-2}$ (Hosmer
and Lemeshow 1980, Communications in Statistics, Theory and Methods 9(10), 1043 to 1069,
[doi:10.1080/03610928008827941](https://doi.org/10.1080/03610928008827941)). On an independent hold-out sample
some authors use $G$ degrees of freedom; the package defaults to $G - 2$, accepts `dof`, and records the choice.

## Spiegelhalter test (`pd.spiegelhalter`)

At obligor level, with forecasts $f_j$ and outcomes $d_j$, under $H_0$: $p_j = f_j$ for every $j$
(Spiegelhalter 1986, Statistics in Medicine 5(5), 421 to 433,
[doi:10.1002/sim.4780050506](https://doi.org/10.1002/sim.4780050506); formula as documented in the
[Stata `brier` manual](https://www.stata.com/manuals/rbrier.pdf)):

$$E(\mathrm{BS}) = \frac1N\sum_j f_j(1-f_j),\quad \mathrm{Var}(\mathrm{BS}) = \frac{1}{N^2}\sum_j f_j(1-f_j)(1-2f_j)^2,\quad Z = \frac{\mathrm{BS} - E(\mathrm{BS})}{\sqrt{\mathrm{Var}(\mathrm{BS})}},$$

with $\mathrm{BS}$ the observed Brier score; the p-value is the upper normal tail. No binning is needed.

## Normal test over several periods (`pd.normal_multiperiod`)

For years $t = 1, \dots, T$ with forecast $\mathrm{PD}_t$ and observed default rate $d_t$ (WP14):

$$\hat\sigma^2 = \frac{1}{T-1}\left(\sum_{t}(d_t - \mathrm{PD}_t)^2 - \frac1T\Big(\sum_t (d_t - \mathrm{PD}_t)\Big)^2\right),\qquad Z = \frac{\sum_t (d_t - \mathrm{PD}_t)}{\sqrt{T}\,\hat\sigma}.$$

$H_0$: none of the true PDs in years $1..T$ exceeds its forecast; reject when $Z$ exceeds the upper normal
quantile. Cross-sectional dependence is admissible; independence over time is assumed; power is moderate for
short series such as $T = 5$.

## Traffic-lights test (`pd.traffic_lights`)

Blochwitz, Hohl and Wehn, as specified in WP14. Each year's standardised default rate
$R_t = (D_t - N_t p_t)/\sqrt{N_t p_t (1 - p_t)}$ is coloured green, yellow, orange or red at the cut-offs
$\Phi^{-1}(q_g)$, $\Phi^{-1}(q_g + q_y)$, $\Phi^{-1}(q_g + q_y + q_o)$; WP14's simulation used
$(q_g, q_y, q_o, q_r) = (0.5, 0.3, 0.15, 0.05)$, the package default. Under $H_0$ the colour counts
$(A_g, A_y, A_o, A_r)$ over $T$ years are multinomial. WP14 orders profiles by
$V = 1000A_g + 100A_y + 10A_o + A_r$ and rejects when $V \le v_\alpha$, the greatest $v$ with
$P[V \le v] < \alpha$. The package reports $p = P[V \le V_{obs}]$ by exact multinomial enumeration (for any $T$,
using the lexicographic order of the counts, which equals the order of $V$ when $T \le 9$). WP14's simulation
found the traffic lights "more powerful than the normal" test and the normal test "slightly more robust".

## Brier score and Murphy decomposition (`pd.brier`, descriptive)

$$\mathrm{BS} = \frac1N\sum_j (f_j - d_j)^2 = \underbrace{\frac1N\sum_k n_k(\bar f_k - \bar d_k)^2}_{\text{reliability}} - \underbrace{\frac1N\sum_k n_k(\bar d_k - \bar d)^2}_{\text{resolution}} + \underbrace{\bar d(1 - \bar d)}_{\text{uncertainty}}$$

(Brier 1950, [doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2);
Murphy 1973, [doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)).
The identity is exact when forecasts are constant within each group $k$ (rating grades) and approximate after
binning continuous scores; the result flags which. For low default rates the trivial forecast already has a small
Brier score, so the Brier skill $1 - \mathrm{BS}/\big(\bar d(1-\bar d)\big)$ is reported (WP14).

## Expected calibration error (`pd.ece`, descriptive)

$$\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\,\big|\bar y(B_m) - \bar p(B_m)\big|$$

(Naeini, Cooper and Hauskrecht 2015, [doi:10.1609/aaai.v29i1.9602](https://doi.org/10.1609/aaai.v29i1.9602);
Guo et al. 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Binning-dependent; the package defaults to
quantile bins because PDs cluster near zero, where equal-width bins would put almost everyone in the first bin.
The bins are returned for the reliability diagram.

## Reading calibration results

The ECB tests one-year default rates while IRB PDs are long-run averages (CRR Article 180; EBA/GL/2017/16
chapter 5). A one-year rejection in a downturn year is expected for a through-the-cycle calibration, so a
calibration result should always be read next to the model's rating philosophy and the year's macro conditions.
