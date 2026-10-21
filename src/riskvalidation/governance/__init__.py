"""Model governance: the scope test, the inventory and its dependency graph, the tiering rubric, the lifecycle, the
findings workflow, the review schedule, the overall opinion, the model risk appetite metrics, the policy identity
of a test row and the validation report (contract `engine-governance-api` 1.0.0; dossier 06 B and I).

Modules are imported by path (for example ``from riskvalidation.governance.tiering import matrix_tier``). The
engine computes; it approves nothing and certifies nothing. Every tier count, band, deadline, frequency and
threshold is a stated **convention**: a named parameter that carries the label ``convention``
(:data:`CONVENTIONS`), passed to its function as an argument with the default of
:mod:`riskvalidation.governance.conventions`, and never presented as a value taken from a regulation. No function
reads the clock, the environment or a file: the as-of date and every table come in as arguments. The ``gov.``
validation tests are in :mod:`riskvalidation.validation.governance`.
"""

from riskvalidation.governance.conventions import CONVENTIONS

__all__ = [
    "CONVENTIONS",
    "appetite",
    "findings",
    "inventory",
    "lifecycle",
    "opinion",
    "policy",
    "report",
    "schedule",
    "scope",
    "tiering",
    "conventions",
]
