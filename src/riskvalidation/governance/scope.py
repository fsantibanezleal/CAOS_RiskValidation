"""The B.1 scope test (dossier 06 B.1 l.523-537): four answers and a jurisdiction profile give the in-scope decision
with its rationale.

The rules are a reading of the B.1 table and are UNVERIFIED beyond it: ``us-2026`` (SR 26-2) is in scope when
theory is applied and the output is an estimate and the model is not generative or agentic; ``pra`` (SS1/23
Principle 1.1) when theory is applied, the output need not be quantitative; ``osfi-2027`` (E-23) and ``ecb`` when
theory is applied and the output is an estimate. The rationale is English; the governance bake adds the Spanish
text.
"""

from __future__ import annotations

from dataclasses import dataclass

PROFILES = ("us-2026", "pra", "osfi-2027", "ecb")


@dataclass(frozen=True)
class ScopeAnswers:
    """The four answers of the scope test, all recorded, including the one a profile does not read."""

    applies_theory: bool  # (a) statistical, economic, financial or mathematical theory is applied
    output_is_estimate: bool  # (b) the output is an estimate
    deterministic: bool  # (c) a deterministic rule set
    generative_or_agentic: bool  # (d) a generative or agentic AI model


@dataclass(frozen=True)
class ScopeDecision:
    profile: str
    in_scope: bool
    rationale: str


def _rationale_us(a: ScopeAnswers) -> tuple[bool, str]:
    if a.generative_or_agentic:
        return False, "Out of scope: it is a generative or agentic AI model (SR 26-2)."
    if not a.applies_theory:
        extra = " and it is a deterministic rule set" if a.deterministic else ""
        return False, f"Out of scope: it applies no statistical or economic theory{extra} (SR 26-2)."
    if not a.output_is_estimate:
        return False, "Out of scope: its output is not an estimate (SR 26-2)."
    return True, (
        "In scope: it applies statistical or economic theory, its output is an estimate and it is not generative or "
        "agentic (SR 26-2)."
    )


def _rationale_pra(a: ScopeAnswers) -> tuple[bool, str]:
    if a.applies_theory:
        return True, "In scope: it applies a quantitative method (SS1/23 Principle 1.1)."
    return (
        False,
        "Out of scope: it applies no statistical, economic, financial or mathematical theory (SS1/23 Principle 1.1).",
    )


def _rationale_estimate(a: ScopeAnswers, source: str) -> tuple[bool, str]:
    if not a.applies_theory:
        return False, f"Out of scope: it applies no statistical, economic, financial or mathematical theory ({source})."
    if not a.output_is_estimate:
        return False, f"Out of scope: its output is not an estimate ({source})."
    return True, f"In scope: it applies theory and its output is an estimate ({source})."


def scope_test(answers: ScopeAnswers, profile: str) -> ScopeDecision:
    """The in-scope decision of one profile, with the answers that decided it named in the rationale."""
    if profile == "us-2026":
        ok, why = _rationale_us(answers)
    elif profile == "pra":
        ok, why = _rationale_pra(answers)
    elif profile == "osfi-2027":
        ok, why = _rationale_estimate(answers, "OSFI E-23")
    elif profile == "ecb":
        ok, why = _rationale_estimate(answers, "ECB")
    else:
        raise ValueError(f"unknown scope profile {profile!r}; expected one of {PROFILES}")
    return ScopeDecision(profile=profile, in_scope=ok, rationale=why)


def scope_record(answers: ScopeAnswers) -> tuple[ScopeDecision, ...]:
    """The four decisions in the order ``us-2026``, ``pra``, ``osfi-2027``, ``ecb`` (the inventory field
    ``scope.profiles``).
    """
    return tuple(scope_test(answers, p) for p in PROFILES)

__all__ = [
    "ScopeAnswers",
    "ScopeDecision",
    "scope_record",
    "scope_test",
]
