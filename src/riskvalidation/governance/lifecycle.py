"""The B.4 lifecycle (dossier 06 B.4 l.576-583): twelve states and their allowed transitions, a **convention**
synthesised from SS1/23 para 2.10, OSFI E-23 and SR 11-7.

Use before validation (from ``in_development``, ``submitted_for_validation`` or ``in_validation`` to ``in_use``)
is allowed only with an exception record of a type, an approver and an expiry date (SR 11-7 page 9; SR 26-2
Section V; SS1/23 Principle 5.3, where exceptions are temporary). A refused transition names both states.
``decommissioned`` is terminal: a decommissioned model may serve as a challenger but does not re-enter the
lifecycle.
"""

from __future__ import annotations

from datetime import date

from riskvalidation.governance.conventions import LIFECYCLE_GRAPH

PRE_VALIDATION = ("in_development", "submitted_for_validation", "in_validation")


def _known(state: str) -> str:
    if state not in LIFECYCLE_GRAPH:
        raise ValueError(f"unknown lifecycle state {state!r}")
    return state


def allowed_next(state: str) -> tuple[str, ...]:
    """The states reachable in one step without an exception record."""
    return LIFECYCLE_GRAPH[_known(state)]


def transition(state: str, to: str, exception: tuple[str, str, date] | None = None) -> str:
    """The state after moving from ``state`` to ``to``; ``exception`` is (type, approver, expiry date)."""
    _known(state)
    _known(to)
    if to in LIFECYCLE_GRAPH[state]:
        return to
    if to == "in_use" and state in PRE_VALIDATION and exception is not None:
        kind, approver, expiry = exception
        if not str(kind).strip() or not str(approver).strip() or not isinstance(expiry, date):
            raise ValueError("an exception record needs a type, an approver and an expiry date")
        return to
    raise ValueError(f"cannot move from {state!r} to {to!r}")

__all__ = [
    "allowed_next",
    "transition",
]
