# 05. CVA, DVA and wrong-way risk

Module `riskvalidation.ccr.cva`. Sources: dossier 05 C.2 (ORE `xva.tex`, "Value Adjustments"); the U9 research addendum R1 (Hull and White, "CVA and wrong way risk",
Financial Analysts Journal 68(5):58-69, 2012, DOI 10.2469/faj.v68.n5.6, SSRN 2151507; the author's draft of 6 July 2012, read from the Internet Archive copy of
`https://www-2.rotman.utoronto.ca/~hull/downloadablepublications/WrongWayRisk.pdf`; page numbers are the draft's).

## The credit triangle and the default probabilities

$$\lambda=\frac{s}{LGD},\qquad S(t)=\exp\!\Big[-\frac{s\,t}{LGD}\Big]$$

`survival_from_spread(spread, t, lgd)`: the spread per year as a decimal, $LGD\in(0,1]$ (Hull and White: $e^{-s_it_i/(1-R)}$ "is a good estimate of the (risk-neutral)
probability of no default", and the average hazard is $s(t)/(1-R)$, page 13). $S(0)=1$, non-increasing in $t$, 1 for a zero spread. Golden: $s=0.012$, $LGD=0.6$, $t=5$ gives
$e^{-0.1}=0.904837$. `default_probabilities(survival)` takes the (K+1,) curve at $t_0=0,\dots,t_K$:

$$PD(t_{i-1},t_i)=S(t_{i-1})-S(t_i)$$

non-negative and summing to $S(0)-S(t_K)$; golden `[1, 0.9, 0.8]` gives `[0.1, 0.1]`.

## CVA and DVA

$$\mathrm{CVA}=\sum_iPD(t_{i-1},t_i)\,LGD\,\mathrm{EPE}(t_i),\qquad \mathrm{DVA}=\sum_iPD_{Bank}(t_{i-1},t_i)\,LGD_{Bank}\,\mathrm{ENE}(t_i)$$

`cva(epe_discounted, pd, lgd)` and `dva(ene_discounted, pd_bank, lgd_bank)` return `(value, terms)` with `terms` the bucket terms in money (the `cva_dva.buckets` of the case
document). This is the unilateral, discretised formula with independence between exposure and default, the "advanced" discretisation of ORE (dossier 05 C.2; Brigo and
Mercurio). CVA is a positive cost; with a zero spread (`pd = 0`) it is exactly 0; it is linear in the LGD and in the EE. Golden: `epe = [10, 20]`, `pd = [0.1, 0.1]`, `lgd = 0.6`
gives $0.6+1.2=1.8$; the same numbers with the ENE give a DVA of 1.8. The simple bilateral form is $\mathrm{BCVA}=\mathrm{CVA}-\mathrm{DVA}$. The paper's own
discretisation, $\mathrm{CVA}\approx(1-R)\sum q_iv_i$ with the mid-point $t_i^*=0.5(t_{i-1}+t_i)$, is the same sum with the exposure read at the mid-point.

## The wrong-way hazard model

Hull and White model the hazard rate as an exponential function of the portfolio value $w$ (equations 8 and 9, pages 15 and 16):

$$h(t)=\exp\big[a(t)+b\,w(t)+\sigma\varepsilon\big]\ (8),\qquad h(t)=\exp\big[a(t)+b\,w(t)\big]\ (9),\quad\frac{\Delta h}{h}=b\,\Delta w$$

$b$ measures the amount of wrong-way ($b>0$) or right-way ($b<0$) risk, $a(t)$ is a function of time, and the noise term makes little difference unless $\sigma$ is very large
relative to $w$, so equation (9) is the working model; `wwr_calibrate_a` accepts `sigma_noise` and the draws `eps` for the full form. `b_from_two_points(w1, hazard1, w2, hazard2)` is the
subjective route of page 16: with a flat hazard term structure, $b=(\ln h_2-\ln h_1)/(w_2-w_1)$; the paper's example ($w=3$ and 20 USD million, hazards 5 and 10 percent) gives
$0.040773$ (printed 0.0408), per USD million.

Calibration of $a(t)$, Appendix page 22: on simulation trial $j$ the hazard is $h_{ij}=\exp[a(t_i^*)+b\,w_{ij}+\sigma\varepsilon_{ij}]$ and, for each $k$, the average survival over the
$m$ trials equals the spread-implied one:

$$\frac1m\sum_{j=1}^m\exp\Big(-\sum_{i=1}^kh_{ij}\Delta t_i\Big)=\exp\Big(-\frac{s_kt_k}{1-R}\Big)$$

`wwr_calibrate_a(w, survival_target, dt, b, sigma_noise=0.0, eps=None)` solves it sequentially by a root search: $k=1$ from the $w_{1j}$, then $k=2$ from the $w_{2j}$ and the $h_{1j}$, and so
on. It returns `(a, h)` with `a` (K,) and the hazards `h` (P, K). With $b=0$ every path has the same hazard. The largest calibration error, the distance between the average survival
and its target, is at most 1e-10 (the `a_calibration_max_error` of the case document). A target that nonnegative hazards cannot reach (the survival would have to rise) is refused with
the field named.

**Golden, the Appendix three-path example** (copied inputs): $\sigma=0$, $m=3$, spread 1 percent at every maturity, recovery 0, $\Delta t=0.5$, $b=0.01$; $w$ at step 1 is
$[100,200,300]$ and at step 2 $[100,300,400]$. The paper prints $a(t_1^*)=-6.9128$, $h_1=[0.00270,0.00735,0.01998]$, $a(t_2^*)=-7.8509$, $h_2=[0.00106,0.00782,0.02126]$. Recomputed on the printed
equations: $a_1=-6.91283$, $a_2=-7.85077$, $h_1=[0.002705,0.007352,0.019984]$, $h_2=[0.001059,0.007822,0.021263]$. The engine is held to the recomputed values at 5e-6 for $a$ and 5e-7 for
$h$; against the printed digits the tolerance is 2e-4 for $a$ (the printed $a_2$ is 1.3e-4 from the recomputed one) and 1e-5 for $h$. A plain-Python bisection in the test solves the same
equations and agrees to 1e-9.

`wwr_cva(w, exposure_discounted, survival_target, dt, lgd, b)` returns `(cva, mc_se)`:

$$\mathrm{CVA}=LGD\sum_i\frac1m\sum_jPD_{ij}E_{ij},\qquad PD_{ij}=S_{ij}(t_{i-1})-S_{ij}(t_i)$$

with $S_{ij}=\exp(-\sum_{i'\le i}h_{i'j}\Delta t_{i'})$ the path-wise survival from the calibrated hazards and $E_{ij}$ the discounted positive exposure; `mc_se` is the standard error of the path
mean. With $b=0$ the hazard does not depend on the path and the value is that of the independent `cva` on the same paths, computed by it (bit for bit: the same arithmetic). Wrong-way risk
raises CVA for a positively correlated exposure (the test sweeps $b$ from -0.2 to 0.4 and the value rises at every step); the sweep over a named grid of $b$ is the case document's
`wrong_way_risk.sweep`. The paper's own results (Tables 1 and 2, page 24: CVA at $b=0$ of 0.048, 0.036, 0.011 and 0.002 for a one-year forward FX contract and the percentage impacts at
$b=0.03$) are for an FX forward and are not reproduced here: UNVERIFIED as a swap check. The paper says swap portfolios gave "similar results" (page 20) but publishes no swap numbers.

## Limits

- ORE's wrong-way example (`Examples/Exposure/run_wwr.py`, tag 1.8.17.0) is a different model: a correlated credit LGM with an instantaneous correlation of -0.2 between the credit factor and
  the rate factor. It has no relationship with the hazard model here, so no ORE golden applies to `wwr_cva` and the wrong-way variant has no ORE reconciliation. Hull and White's footnote 22
  gives an alternative form $h=\ln(1+\exp[a+bw])$ with "very little difference" in practice; it is not implemented.
- The first-to-default form of bilateral CVA (each term multiplied by the survival probability of the other party; Brigo, Morini and Pallavicini 2013) is not implemented: the simple
  $\mathrm{BCVA}=\mathrm{CVA}-\mathrm{DVA}$ only.
- The hazard is linked to the portfolio value, the paper's choice of three; a link to an outside variable (the 5-year spread, a stock price) is not implemented. The paper notes that linking
  to the exposure itself works worse because the exposure is zero on many paths.
- The spread curve is the caller's. No open CDS spread series is known (sizing R5); the base credit curve of the case is the HQM minus Treasury proxy of addendum R10, a floor for lower
  ratings.
