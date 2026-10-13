# LGD and EAD

`riskvalidation.lgd_ead` computes the realised (workout) LGD, the LGD rungs, the downturn add-on, the CCF and the EAD models,
and the IRB floors; `riskvalidation.validation.lgd_ead` validates them. The engine computes and makes no compliance claim:
discount rates, downturn periods, thresholds and light parameters are policy parameters the caller supplies or Contraste
internal gates, not regulatory requirements. An LGD or a CCF outside [0, 1] is kept and flagged with `range_flag`, never
silently clipped, and a value that cannot be computed (a fully drawn line, an empty sample) is undefined (NaN) or a
`not_evaluated` result, never a placeholder number. The CCF is in the EU convention (the share of the undrawn amount drawn
by default, `convention.name = "eu-undrawn"`; the LEQ in US terms): every page that shows a CCF names it.

| Page | Contents |
|---|---|
| [01. Workout LGD](01_workout-lgd.md) | the discounted workout LGD, the discount rate convention, the range flag, incomplete workouts, the training cut-off, the LGD distribution |
| [02. LGD models](02_lgd-models.md) | the rungs: OLS, fractional logit, beta regression, hurdle, Tobit, the cure decomposition; the endpoint transform and its sensitivity |
| [03. Downturn LGD](03_downturn.md) | the downturn period from a macro path, the cohort add-on, the coverage shortfall |
| [04. CCF and EAD](04_ccf-and-ead.md) | the EU convention and the US terms, the reference-date designs, the treatments of a CCF outside [0, 1], the direct EAD and utilisation change models |
| [05. Floors](05_floors.md) | the IRB LGD floor cells, the partly secured floor formula, the EAD floor, the supervisory CCFs |
| [06. The validation tests](06_validation-tests.md) | the six tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.lgd_ead import workout` and
`from riskvalidation.validation import lgd_ead`; nothing is re-exported from the package root. Requirements `RV-601` to
`RV-625` are in `docs/design/features/u6-lgd-ead/requirements.md`, each naming its test. Sources: dossier 04 of the Contraste
initiative (sections F.1 to F.4 and G.1 to G.3) and dossier 01 (section A.5), cited with their DOIs on each page. An item the
dossiers mark UNVERIFIED is marked UNVERIFIED here and is not silently decided.
