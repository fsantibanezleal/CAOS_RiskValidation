# 03. Conditional disparity and replication

Modules `riskvalidation.fairness.conditional` and `riskvalidation.fairness.replication`. Sources: dossier 08 l.839 (the question: the
disparity "after controlling for permitted underwriting factors" and whether the finding is stable year to year); decision Q4 of the
Contraste U14 sizing (option a); the U14 research addendum (H1 the HMDA field names, section 2.4 underwriting and pricing fields;
H3 the legal tests, section 4.3). Primary texts: the Interagency Fair Lending Examination Procedures (August 2009),
https://www.federalreserve.gov/boarddocs/caletters/2009/0906/09-06_attachment.pdf; the HMDA data dictionary and Data Browser,
https://ffiec.cfpb.gov/documentation/; McCullagh and Nelder (1989), Generalized Linear Models, 2nd ed., for the iteratively
reweighted least squares of the logistic model; statsmodels Logit and OLS (the independent references).

## The decision model

The approval (1) or denial (0) of a row is modelled on the protected indicators $a_{i1},\dots,a_{iq}$ (columns against the reference
group) and the permitted controls $c_{i1},\dots,c_{ic}$:

$$\operatorname{logit}P(\hat Y_i=1)=\beta_0+\sum_{j=1}^{q}\beta_j a_{ij}+\sum_{l=1}^{c}\gamma_l c_{il}$$

`fit_decision_model(decision, protected, controls)` maximises the likelihood by Newton's method, which for the logit link is iteratively
reweighted least squares, from the zero start with step halving:

$$\beta^{(t+1)}=\beta^{(t)}+(X^{\top}W X)^{-1}X^{\top}(y-\mu),\qquad W=\operatorname{diag}\bigl(\mu_i(1-\mu_i)\bigr),\qquad
\widehat{\operatorname{Cov}}(\hat\beta)=(X^{\top}\hat W X)^{-1}$$

Symbols: $X$ the design with the intercept first, then the protected columns, then the controls; $\mu_i$ the fitted approval
probability. The columns are standardised inside the fit for conditioning and the coefficients and standard errors are returned in the
original units. The result is a frozen `DecisionModel(coefficients, standard_errors, n, converged)`: coefficients are **log-odds**,
`coefficients[0]` is the intercept and `coefficients[1 + j]` the effect of protected indicator $j$. It stops when the largest step is
below 1e-11 (100 steps at most); on complete separation, a standardised coefficient beyond 50 or no convergence it returns
`converged = False` with NaN coefficients and standard errors and raises nothing. A decision that is not 0 and 1 with both classes,
non-finite entries, rows that do not match, a constant column and a rank-deficient design raise `ValueError` naming the field. Reference
(the gate): statsmodels `Logit` agrees on the coefficients and the standard errors to 1e-6.

## The rate-spread model

`fit_rate_spread_model(rate_spread, protected, controls)` is linear least squares on the same terms for the loans that carry a rate
spread (action taken 1 or 2), in percentage points:

$$\text{spread}_i=\beta_0+\sum_{j}\beta_j a_{ij}+\sum_{l}\gamma_l c_{il}+\varepsilon_i,\qquad \widehat{\operatorname{Cov}}(\hat\beta)=\hat\sigma^2 (X^{\top}X)^{-1},\ \hat\sigma^2=\frac{\text{RSS}}{n-k}$$

Rows with a missing (NaN) rate spread are dropped and never imputed; the count dropped is the rows given minus `n` of the result.
Reference: statsmodels `OLS` agrees to 1e-8.

## The protected effect

`protected_effect(model, index, level=0.95)` reads one coefficient and returns a frozen `Effect(coefficient, odds_ratio, lo, hi,
level)`:

$$\hat\beta\ \pm\ z_{1-(1-\ell)/2}\ \widehat{\operatorname{se}}(\hat\beta),\qquad \text{odds ratio}=e^{\hat\beta}$$

`index` is the position in `coefficients`: protected indicator $j$ is index $1+j$, and index 0, the intercept, is refused with
`ValueError("index 0 is the intercept, not a protected indicator; indicator j is index 1 + j")` (contract 1.1.0; in 1.0.0 it was
returned as if it were an effect). The interval equals the coefficient
plus or minus $z\,\text{se}$ (identity) and always satisfies lo <= coefficient <= hi; an index outside the coefficients, a level
outside $(0, 1)$ and a model whose standard error is not finite raise `ValueError`. The odds ratio is meaningful for the decision
model; for the linear model read the coefficient. The level 0.95 is a convention.

## Replication

`replicate_effect(fit, confirm)` compares the estimate fitted on year $t$ with the same estimate refitted on year $t+1$, by sign and
size only (decision Q4): `same_sign` is the equality of the signs of the two coefficients and `size_ratio` is
$\hat\beta^{(t+1)}/\hat\beta^{(t)}$ (None when the fit coefficient is 0; negative when the sign flips). No verdict beyond sign and
size is returned. Golden: equal estimates give ratio 1 and `same_sign` True; $0.5$ then $0.25$ gives ratio $0.5$; $0.3$ then $-0.15$
gives `same_sign` False and ratio $-0.5$.

## The C21 conventions this module is called with

The field names are the snapshot spellings of the addendum's H1 (`derived_race`, `derived_ethnicity`, `derived_sex`, `applicant_age`,
`action_taken`, `debt_to_income_ratio`, `combined_loan_to_value_ratio`, `income`, `loan_amount`, `property_value`, `loan_type`,
`rate_spread`). The approval is a named convention: `action_taken` 1 or 2 is approval and 3 is denial; codes 4 to 8 have no decision
and are left out. The mixed exact and binned debt-to-income field is recoded to its published bins before it enters the controls.
Public HMDA has no credit score, so every controlled estimate names it as a missing control.

Limits: the controlled disparity is an **approximation of comparative evidence**, never a finding of a violation and no legal claim.
Regulation B as amended effective 2026-07-21 does not apply the effects test under ECOA (12 CFR 1002.6(a)); the Fair Housing Act is
a separate statute under which disparate-impact claims are cognizable (91 FR 21620, recital); the Interagency Procedures state that
gross HMDA approval or denial disparities are not appropriate for adverse impact analysis (addendum H3). The model omits whatever the
caller does not supply (the credit score, the pricing rules of the lender), so a persisting effect may be confounded by it. The
Wald interval is a large-sample approximation. UNVERIFIED: no dossier prints the formula of the controlled disparity or the
replication (addendum, decision Q4: the estimate and the interval are the construction chosen, with statsmodels as the reference);
the approval and denial classification of `action_taken` (see [01. Metrics](01_metrics.md)). Golden values: no dossier prints one; the
checks are the independent references above and the planted-bias runs of [05. Planted bias](05_planted-bias.md).
