"""Operational risk: the loss distribution approach (LDA) and the Basel III standardised approach (SMA).

Modules are imported by path (for example ``from riskvalidation.oprisk import aggregate``); nothing is re-exported here.

- ``distributions``: frequency and severity laws (lognormal, Weibull, generalised Pareto and the lognormal body with
  a GPD tail splice), their truncated versions above the collection threshold with the frequency rescaled, and their
  maximum likelihood fits.
- ``aggregate``: the compound annual loss by Monte Carlo, Panjer recursion and FFT, and what is read from it.
- ``sla``: the single loss approximation, and its mean-corrected form, which is UNVERIFIED.
- ``scenarios``: scenario analysis and its combination with loss data.
- ``sma``: the OPE25 standardised approach (business indicator, BIC, loss component, ILM).

The engine computes and states no compliance claim: the LDA is the economic-capital and validation tool of
dossier 05 F.1, never a Pillar 1 model, and collection thresholds, quantile levels and light parameters are policy
parameters the caller supplies. Money is a float in one currency (the caller's); the engine never converts. A tail
index of 1 or more has no mean: no function returns a finite mean-based number for it. The validation tests are in
``riskvalidation.validation.oprisk``. Contract: ``engine-oprisk-api`` 1.0.0.
"""
