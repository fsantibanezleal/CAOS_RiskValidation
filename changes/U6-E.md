- engine U6-E: new package `riskvalidation.lgd_ead` (workout LGD with the EU discount rate convention, range flags,
  incomplete workout treatment and the LGD distribution; the LGD rungs OLS, fractional logit, beta, hurdle and Tobit with
  `predict_lgd`; the cohort downturn add-on; the EU-convention CCF with three reference-date designs and four treatments,
  the direct EAD and utilisation change models; the IRB LGD and EAD floors, the partly secured floor formula documented for the caller and the
  supervisory CCFs) and the six tests of `riskvalidation.validation.lgd_ead`, reproducing every golden value of the contract
  `engine-lgd-ead-api` 1.0.0 (requirements RV-601 to RV-625, pages under `docs/lgd_ead/`).
