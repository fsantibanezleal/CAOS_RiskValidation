"""A battery of e-processes: several monitors read as one anytime-valid verdict (BS-15).

`eprocess_battery` combines named `EProcessResult` components that cover the same time steps. Two combinations, both
e-processes (so Ville's bound keeps the size at `alpha_level` whatever the number of components and however often the
battery is read):

- `"average"` (default): the mean of the components. An e-process whatever the dependence between the components, so
  it is the right combination for monitors fed the same series (the hit-rate and the clustering monitors of one VaR).
  The price is that a signal in one component is divided by the number of components.
- `"product"`: the product. An e-process when the components are independent given the past (for example monitors of
  different portfolios, or of different grades); a signal in one component passes whole and signals add in the log.

The null of the battery is the intersection of the component nulls; a rejection says that at least one fails and the
component paths say which one moved (`extras["components"]`). Sources: Ramdas, Grunwald, Vovk and Shafer (2023),
Statistical Science 38(4):576-601, arXiv 2210.01948 (combining e-values: averaging is always valid, multiplying needs
conditional independence); Ville (1939).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.validation.sequential._core import check_alpha_level
from riskvalidation.validation.sequential.monitor import EProcessResult

REFERENCE = (
    "Ramdas, Grunwald, Vovk and Shafer (2023), Statistical Science 38(4):576-601, arXiv 2210.01948; Ville (1939)"
)


def eprocess_battery(components: Mapping[str, EProcessResult], *, combine: str = "average",
                     alpha_level: float | None = None) -> EProcessResult:
    """Combine named e-processes of equal length into one.

    `components` maps a name to an `EProcessResult` (all with the same number of steps, at least one; a non-record
    raises `TypeError`, a length mismatch or an empty mapping `ValueError`). `combine` is `"average"` or `"product"`
    (see the module docstring). `alpha_level` defaults to the first component's. Returns an `EProcessResult` whose
    `extras` list each component's name, final `E`, supremum and first crossing.
    """
    if combine not in ("average", "product"):
        raise ValueError(f"combine must be 'average' or 'product', got {combine!r}")
    if not components:
        raise ValueError("a battery needs at least one component")
    for name, comp in components.items():
        if not isinstance(comp, EProcessResult):
            raise TypeError(f"component {name!r} must be an EProcessResult, got {type(comp).__name__}")
    names = list(components)
    lengths = {components[k].n for k in names}
    if len(lengths) != 1:
        raise ValueError(f"the components must cover the same steps, got lengths {sorted(lengths)}")
    level = check_alpha_level(alpha_level if alpha_level is not None else components[names[0]].alpha_level)
    paths = np.array([components[k].log_e for k in names])
    if combine == "product":
        log_e = paths.sum(axis=0)
    else:
        log_e = np.logaddexp.reduce(paths, axis=0) - np.log(len(names))
    detail = {k: {"final_log_e": float(components[k].log_e[-1]) if components[k].n else None,
                  "max_log_e": components[k].max_log_e if components[k].n else None,
                  "rejection_time": components[k].rejection_time} for k in names}
    return EProcessResult(
        monitor_id="sequential.battery", log_e=log_e, alpha_level=level,
        h0="every component null holds (the intersection of the component nulls)", reference=REFERENCE,
        inputs_hash=inputs_hash(paths, level), notes=f"components combined by {combine}",
        extras={"combine": combine, "components": detail})


__all__ = [
    "eprocess_battery",
]
