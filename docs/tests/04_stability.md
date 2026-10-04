# 04. Population and characteristic stability

Stability asks whether the population the model now scores still resembles the one it was built on. A model can
be well calibrated on its development sample and wrong on a population that has moved.

## Population stability index (`stability.psi`) and characteristic stability index (`stability.csi`)

For $B$ fixed bins with development proportions $e_i$ and current proportions $a_i$:

$$\mathrm{PSI} = \sum_{i=1}^{B}(a_i - e_i)\ln\frac{a_i}{e_i},$$

the symmetrised Kullback-Leibler divergence. The CSI is the same formula on one model input's bins (pass the
characteristic's name as `segment`).

**The conventional bands.** PSI below 0.10 "little change", 0.10 to 0.25 "moderate change", 0.25 or more
"significant change, action required". These benchmarks "are used without reference to statistical type I or
type II error rates" (Yurdakul and Naranjo 2020, *Statistical properties of the population stability index*,
Journal of Risk Model Validation 14(4), 89 to 100, [doi:10.21314/JRMV.2020.227](https://doi.org/10.21314/JRMV.2020.227)).

**The benchmark that controls the error.** With development and current sample sizes $n$ and $m$, Yurdakul and
Naranjo show that $(1/n + 1/m)^{-1}\,\mathrm{PSI}$ is approximately $\chi^2_{B-1}$, so $E[\mathrm{PSI}] \approx
(1/n + 1/m)(B - 1)$ and a test of $H_0$: identical binned distributions rejects when

$$\mathrm{PSI} > \Big(\frac1n + \frac1m\Big)\,\chi^2_{1-\alpha,\,B-1}.$$

Their Table 2 ($B = 10$, $\alpha = 5\%$, $\chi^2_{0.95, 9} = 16.92$) gives 0.338, 0.169, 0.085 and 0.034 for
$n = m$ = 100, 200, 400 and 1,000, reproduced by `psi_benchmark` (test `test_psi_chi2_benchmark`). Their findings:
the 0.10 rule has too large a type I error unless both samples exceed about 400; the 0.25 rule is too
conservative above about 200; the power of fixed-threshold rules falls as samples grow.

![PSI critical value against sample size, with the conventional bands](../assets/psi-benchmark.svg)

The package reports both: the light follows the chi-square p-value, and the conventional band appears in the
result's notes, labelled as a rule of thumb.

**Implementation choices, stated.**

- Bins must be fixed before the current sample is seen. `quantile_edges` places them at the development
  sample's quantiles, with open outer edges so every current value falls in a bin; Yurdakul and Naranjo note
  that bins defined on base percentiles make PSI asymmetric in the two samples.
- An empty bin would make the logarithm infinite. Its count is floored at half an observation of its own sample
  (a continuity correction); the result counts how many bins were floored.

## Chi-square test of homogeneity (`stability.chi2`)

$X^2 = \sum_{g \in \{dev, cur\}}\sum_i (O_{gi} - E_{gi})^2/E_{gi}$ on the $2 \times B$ table, $\chi^2_{B-1}$ under
$H_0$. Asymptotically it plays the same role as the PSI benchmark. Computed with SciPy's `chi2_contingency`
without continuity correction.

## Two-sample Kolmogorov-Smirnov (`stability.ks`)

For a continuous variable, $\sup_x |F_{dev}(x) - F_{cur}(x)|$ with SciPy's `ks_2samp` p-value; $H_0$: both
samples come from one continuous distribution.

## Wasserstein-1 distance (`stability.wasserstein`, descriptive)

$W_1(F, G) = \int |F(x) - G(x)|\,dx$, in the variable's own units. No regulatory source sets a threshold for it.

## How open-source monitors set defaults (for comparison, not adopted)

Evidently uses KS or chi-square with $p \le 0.05$ for samples up to 1,000 rows and Wasserstein or Jensen-Shannon
distances of at least 0.1 above that ([docs](https://docs.evidentlyai.com/metrics/explainer_drift)); ING's popmon
colours the standardised deviation from a reference, green below 4, yellow 4 to 7, red above 7
([docs](https://popmon.readthedocs.io/en/latest/comparisons.html)). Both are conventions; this package keeps
thresholds in the policy file.
