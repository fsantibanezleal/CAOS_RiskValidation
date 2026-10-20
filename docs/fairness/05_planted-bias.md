# 05. Planted bias

Module `riskvalidation.fairness.plants`. Sources: the SDD l.335 (each test is validated on its planted defect); dossier 07 D pattern 6
l.611-616 (plant a defect of known size, measure the test's size under the null and its power against the defect); the generator
contract of `riskvalidation.generators` (a frozen configuration, `draw(rng)`, an exact `truth()`, a `verify()`); Morris, White and
Crowther (2019), "Using simulation studies to evaluate statistical methods", Statistics in Medicine 38(11), doi:10.1002/sim.8086. All
three plants are built on `generators.LogisticPortfolio`, whose linear predictor $\eta$ is exactly normal, so the selection rates have
a closed form.

Conventions of a draw: `group` 1 is the protected group and 0 the reference; `decision` 1 is favourable (the model approves the
obligors of low $\eta$); `defaults` is the default indicator and `y = 1 - defaults` the outcome aligned with the decision, both as
observed; `pd` is the model's calibrated PD. The `seed` of a plant is the stream of `draw_default()` and of the default `verify()`.

## `ThresholdBiasPlant(n, gap, seed)`

The model is the same for both groups but the protected group is approved below a cutoff lower by `gap` (log-odds), a group-dependent
decision threshold:

$$\text{approve} \iff \eta \le c_g,\qquad c_0=\mu_\eta+\sigma_\eta\Phi^{-1}(r_0),\quad c_1=c_0-\text{gap}$$

$$r_0=\text{base rate},\qquad r_1=\Phi\!\left(\Phi^{-1}(r_0)-\frac{\text{gap}}{\sigma_\eta}\right),\qquad \text{truth: } r_0-r_1$$

Symbols: $\mu_\eta,\sigma_\eta$ the mean and standard deviation of the linear predictor; $r_0$ the reference selection rate (the base
rate, default 0.7). `truth()` gives `selection_rate_reference`, `selection_rate_protected`, `selection_rate_gap` (exactly 0 at gap
0, the null) and `null`. Golden: with the defaults `gap = 0.3` gives a true gap of about 0.091 and `gap = 0.5` of about 0.156;
`verify()` reproduces both rates within Monte Carlo error.

## `LabelNoiseBiasPlant(n, noise, seed)`

The model decides identically for both groups, but each observed label of the protected group is flipped with probability `noise`
(in $[0, 0.5]$), so the error rates measured against the observed outcome differ between groups although the model treats them
alike. `truth()` gives the `flip_rate` (and `flipped_group` 1); `draw` returns both the observed (`y`, `defaults`) and the unflipped
(`y_true`, `defaults_true`) labels. `verify()` reproduces the flip rate of the protected group, zero flips in the reference group
and the selection rate.

## `ProxyFeaturePlant(n, proxy_strength, seed)`

A feature, the proxy, has correlation $\rho$ = `proxy_strength` with the protected indicator and the decision depends on the
covariates and the proxy only, never on the protected indicator:

$$\text{proxy}=\rho\,\frac{A-\pi}{\sqrt{\pi(1-\pi)}}+\sqrt{1-\rho^{2}}\,\varepsilon,\qquad
\operatorname{logit}P(\hat Y=1)=\beta_0+x^{\top}\beta+\gamma\,\text{proxy}$$

Symbols: $\pi$ the protected share (0.5), $\gamma$ the `proxy_effect` (1.0), $x$ the portfolio covariates. The correlation of the
proxy with $A$ is exactly $\rho$. `truth()` names the proxy (`proxy`, its column `proxy_index` in `controls_with_proxy`) and gives
`protected_effect_given_proxy = 0.0`: a decision model that controls for the covariates but not the proxy finds a protected effect
(the disparity is transmitted by the proxy), and once the proxy is a control that effect is 0 exactly. `draw` returns `controls` (the
covariates) and `controls_with_proxy`. Golden: at $\rho = 0.7$ and 60,000 rows the uncontrolled log-odds effect has an interval above 0
and a coefficient above 0.5, and the controlled one is within three standard errors of zero; at $\rho = 0$ both vanish.

Limits: a plant is an idealisation (one binary protected attribute, a normal linear predictor, independence between the group and the
covariates in the first two plants) and a test that passes its plants has shown only that it detects those defects at those sizes; it
says nothing about a real portfolio. The base rate, the gap, the noise and the proxy strength are named parameters, not findings about
credit. The size and power measured on these plants are in [the validation page](06_validation-tests.md). The engine states no legal
claim. UNVERIFIED: no dossier prints the size of a planted bias to use; the values chosen are conventions of the test, labelled so.
