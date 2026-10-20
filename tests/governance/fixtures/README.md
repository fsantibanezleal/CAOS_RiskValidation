# Governance fixtures

Copied unchanged from CAOS_Contraste, branch `task/u15-s`, commit 5dce3952591f0a9fa6df91c6d39e55e5f7a7736c:
`docs/contracts/fixtures/governance/*-fixture.json`, `docs/contracts/schemas/governance/*.schema.json` (in `schemas/`),
`tests/contracts_u15/semantics.py` (here `semantics.pysrc`, so the linter does not read it) and
`tests/contracts_u15/snapshots/source-extract.json` (the facts of the four fixture cases read from the committed Contraste documents).
`inventory-fixture.json` is copied again from commit a8170a5 of the same branch: the osfi-2027 and ecb scope rationale is the
engine's sentence (the U15-E review, F2).

The fixtures are labelled `fixture:u15-s-contracts` and are never published as results. The tests recompute every computed field of
them through `riskvalidation.governance`.
