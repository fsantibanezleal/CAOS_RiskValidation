"""The classifiers a packaged policy entry names in its ``rule``: the light of a result row from its own fields.

Most Contraste battery tests turn a statistic into a light with two cut points on one value, but some compare a
different field, read two fields (the worse of two metrics), a field of ``extras`` (a count, a flag, a list) or a
band that is itself data. An entry names the rule that reproduces the test function's own light rule (the
function of ``riskvalidation.validation.<family>`` it is transcribed from is named in each docstring), and the
cut points come from the entry (``amber``, ``red`` and ``parameters``), never from the row. A rule reads only the
fields of the row (``statistic``, ``p_value``, ``metric``, ``segment`` and ``extras``), so a committed document
can be re-classified without the inputs of the test. A missing value the rule needs gives ``not_evaluated``.

A rule is ``name`` or ``name:field``; generic rules take the field they read (``statistic``, ``metric`` or
``extras.<key>``) after the colon. ``EPS`` is the rounding tolerance the test functions add to a gate comparison.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from riskvalidation.core.result import Light

EPS = 1e-12

G, A, R, N = Light.GREEN, Light.AMBER, Light.RED, Light.NOT_EVALUATED


def _get(row: Any, name: str) -> Any:
    head, *rest = name.split(".")
    value = row.get(head) if isinstance(row, Mapping) else getattr(row, head, None)
    for key in rest:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _num(row: Any, name: str) -> float | None:
    value = _get(row, name)
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _p(th: Any, name: str) -> float:
    """A named cut point of the effective entry: ``amber`` and ``red`` are the pair, the rest are parameters."""
    if name == "amber":
        return float(th.amber)
    if name == "red":
        return float(th.red)
    try:
        return float(th.parameters[name])
    except KeyError as exc:
        raise KeyError(f"the entry has no parameter {name!r} (its parameters are {sorted(th.parameters)})") from exc


def _worse(*lights: Light) -> Light:
    order = (G, A, R)
    return order[max(order.index(x) for x in lights)]


def _upper_le(value: float, green: float, amber: float) -> Light:
    if value <= green + EPS:
        return G
    return A if value <= amber + EPS else R


def _lower_ge(value: float, green: float, amber: float) -> Light:
    if value >= green - EPS:
        return G
    return A if value >= amber - EPS else R


# generic rules, ``name:field`` ----------------------------------------------------------------------------------

def upper_le(th: Any, row: Any, field: str) -> Light:
    """Green at or below ``amber``, amber at or below ``red``, red above (the test functions' three-light gate)."""
    v = _num(row, field)
    return N if v is None else _upper_le(v, th.amber, th.red)


def lower_ge(th: Any, row: Any, field: str) -> Light:
    """Green at or above ``amber``, amber at or above ``red``, red below."""
    v = _num(row, field)
    return N if v is None else _lower_ge(v, th.amber, th.red)


def abs_upper_le(th: Any, row: Any, field: str) -> Light:
    """``upper_le`` on the absolute value."""
    v = _num(row, field)
    return N if v is None else _upper_le(abs(v), th.amber, th.red)


def red_if_positive(th: Any, row: Any, field: str) -> Light:
    """Red when the field (a count or a violation) is above zero, green otherwise."""
    v = _num(row, field)
    return N if v is None else (R if v > 0.0 else G)


def red_if_nonempty(th: Any, row: Any, field: str) -> Light:
    """Red when the field (a list of failures) is not empty, green when it is an empty list."""
    v = _get(row, field)
    return N if v is None else (R if len(v) else G)


def always_green(th: Any, row: Any, field: str = "") -> Light:
    """A reference row shown for context, never a gate."""
    return G


# aggregation ----------------------------------------------------------------------------------------------------

def aggregation_sum_vc(th: Any, row: Any, field: str = "") -> Light:
    """``aggregation.simple_sum_and_variance_covariance``: red above the simple sum (``tolerance``), amber when the
    variance-covariance ratio is further than ``amber`` from 1."""
    ratio_sum, stat = _num(row, "metric"), _num(row, "statistic")
    if ratio_sum is None or stat is None:
        return N
    if ratio_sum > 1.0 + _p(th, "tolerance"):
        return R
    return A if abs(stat - 1.0) > th.amber + EPS else G


def aggregation_corr_stress(th: Any, row: Any, field: str = "") -> Light:
    """``aggregation.correlation_stress``: red when the capital falls (below ``-tolerance``), amber above ``amber``."""
    stat = _num(row, "statistic")
    if stat is None:
        return N
    if stat < -_p(th, "tolerance"):
        return R
    return A if stat > th.amber + EPS else G


def aggregation_diversification(th: Any, row: Any, field: str = "") -> Light:
    """``aggregation.diversification_sign``: red when the largest factor is above ``1 + tolerance``."""
    stat = _num(row, "statistic")
    return N if stat is None else (R if stat > 1.0 + _p(th, "tolerance") else G)


# ccr ------------------------------------------------------------------------------------------------------------

def ccr_path_convergence(th: Any, row: Any, field: str = "") -> Light:
    """``ccr.path_convergence``: the larger of the EE change (statistic) and the CVA change (metric) within ``k_se``
    green, within ``2 k_se`` amber."""
    s, m = _num(row, "statistic"), _num(row, "metric")
    if s is None or m is None:
        return N
    return _upper_le(max(s, m), _p(th, "k_se"), 2.0 * _p(th, "k_se"))


def ccr_pit_simulated(th: Any, row: Any, field: str = "") -> Light:
    """``ccr.pit_exposure_backtest``: the Anderson-Darling statistic against the row's own simulated 5 and 1 percent
    critical values (``extras.simulated_critical_values``, data of the case, never policy)."""
    w2 = _num(row, "statistic")
    crit = _get(row, "extras.simulated_critical_values")
    if w2 is None or not isinstance(crit, Mapping):
        return N
    return _upper_le(w2, float(crit["p05"]), float(crit["p01"]))


# ifrs9 ----------------------------------------------------------------------------------------------------------

def ifrs9_lifetime_pd(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.lifetime_pd_backtest``: with a Vasicek band, red if an observed horizon is outside the ``alpha_red``
    band, amber outside the ``alpha_amber`` band (the observed rates are ``extras.observed``); without one, the
    worst relative gap against ``gap_amber`` and ``gap_red``."""
    x = _get(row, "extras") or {}
    if "band_low" in x:
        obs = x.get("observed")
        if obs is None:
            return N
        out_red = any(o < lo or o > hi for o, lo, hi in zip(obs, x["band_low_red"], x["band_high_red"], strict=True))
        out_amb = any(o < lo or o > hi for o, lo, hi in zip(obs, x["band_low"], x["band_high"], strict=True))
        return R if out_red else A if out_amb else G
    gaps = x.get("relative_gap")
    if not gaps:
        return N
    worst = max(gaps)
    return G if worst < _p(th, "gap_amber") else A if worst < _p(th, "gap_red") else R


def ifrs9_nonlinearity(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.nonlinearity``: green when the Gauss-Hermite error is within its tolerance and the index is >= 0."""
    m, err = _num(row, "metric"), _num(row, "extras.gauss_hermite_error")
    if m is None or err is None:
        return N
    return G if err <= _p(th, "gauss_hermite_tolerance") and m >= 0.0 else R


def ifrs9_sicr_effectiveness(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.sicr_effectiveness``: the default-rate ratio against ``green_ratio`` and ``amber_ratio``; amber at
    least when the direct Stage 3 share is above ``direct_share_amber``."""
    m = _num(row, "metric")
    if m is None:
        return N
    light = G if m >= _p(th, "green_ratio") else A if m >= _p(th, "amber_ratio") else R
    share = _num(row, "extras.share_stage3_direct_from_stage1")
    if share is not None and share > _p(th, "direct_share_amber") and light is G:
        light = A
    return light


def ifrs9_sicr_stability(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.sicr_threshold_stability``: no loosening period green; red at ``red_share_of_periods`` of the periods
    or more; amber otherwise."""
    loosening, periods = _get(row, "extras.loosening_periods"), _num(row, "extras.periods")
    if loosening is None or periods is None or periods < 2:
        return N
    count = len(loosening)
    if count == 0:
        return G
    return R if count >= periods * _p(th, "red_share_of_periods") else A


def ifrs9_scenario_sensitivity(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.scenario_sensitivity``: red above ``nl_red``, amber above ``nl_amber`` or below zero, else green."""
    nl = _num(row, "metric")
    if nl is None:
        return N
    return R if nl > _p(th, "nl_red") else A if (nl > _p(th, "nl_amber") or nl < 0.0) else G


def ifrs9_ecl_backtest(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.ecl_backtest``: R_H inside the green range, else inside the amber range, else red."""
    r = _num(row, "metric")
    if r is None:
        return N
    if _p(th, "r_green_low") <= r <= _p(th, "r_green_high"):
        return G
    return A if _p(th, "r_amber_low") <= r <= _p(th, "r_amber_high") else R


def ifrs9_overlay(th: Any, row: Any, field: str = "") -> Light:
    """``ifrs9.overlay_diagnostics``: the overlay share above ``share_red`` red, above ``share_amber`` or an
    inconsistent direction amber, else green."""
    m = _num(row, "metric")
    if m is None:
        return N
    if m > _p(th, "share_red"):
        return R
    consistent = _get(row, "extras.direction_consistent")
    return A if (m > _p(th, "share_amber") or consistent is False) else G


# irrbb ----------------------------------------------------------------------------------------------------------

def irrbb_eve_with_without(th: Any, row: Any, field: str = "") -> Light:
    """``irrbb.eve_with_without_behavioural``: red when the behavioural assumptions hide a breach, else the
    three-light gate on ``|statistic|`` (``diff_amber``, ``diff_red``)."""
    stat = _num(row, "statistic")
    if stat is None:
        return N
    if _get(row, "extras.hidden_breach"):
        return R
    return _upper_le(abs(stat), _p(th, "diff_amber"), _p(th, "diff_red"))


def irrbb_srp31_92(th: Any, row: Any, field: str = "") -> Light:
    """``irrbb.srp31_92_goldens``: green when every deviation is within its tolerance (``tolerance_scalar`` for
    ``alpha_short``, ``tolerance_bp`` for the shocks)."""
    dev = _get(row, "extras.deviations")
    if not isinstance(dev, Mapping):
        return N
    ok = all(
        float(d) <= _p(th, "tolerance_scalar" if k == "alpha_short" else "tolerance_bp") + EPS for k, d in dev.items()
    )
    return G if ok else R


# lgd_ead, portfolio ----------------------------------------------------------------------------------------------

def p_value_negative_red(th: Any, row: Any, field: str = "") -> Light:
    """``portfolio.crp_recursion_vs_mc``: red whatever the p-value when the recursion has a negative entry, else the
    p-value bands (green at or above ``amber``, amber at or above ``red``)."""
    if (_num(row, "extras.negative_entries") or 0) > 0:
        return R
    p = _num(row, "p_value")
    return N if p is None else (G if p >= th.amber else A if p >= th.red else R)


def portfolio_size_power(th: Any, row: Any, field: str = "") -> Light:
    """``portfolio.correlation_size_power``: a size cell reads the p-value against the entry's ``amber`` and ``red``;
    a power cell reads the rejection rate against ``power_green`` and ``power_amber``."""
    role = _get(row, "extras.role")
    if role == "power":
        rate = _num(row, "metric")
        if rate is None:
            return N
        return _lower_ge(rate, _p(th, "power_green"), _p(th, "power_amber"))
    p = _num(row, "p_value")
    return N if p is None else (G if p >= th.amber else A if p >= th.red else R)


# market, oprisk, stress, liquidity ---------------------------------------------------------------------------------

def market_pla_zone(th: Any, row: Any, field: str = "") -> Light:
    """``market.pla_test`` (MAR32.42): red when Spearman (statistic) is below ``spearman_red`` or KS (metric) above
    ``ks_red``; green when Spearman is above ``spearman_green`` and KS below ``ks_green``; amber otherwise."""
    rho, ks = _num(row, "statistic"), _num(row, "metric")
    if rho is None or ks is None:
        return N
    if rho < _p(th, "spearman_red") or ks > _p(th, "ks_red"):
        return R
    return G if rho > _p(th, "spearman_green") and ks < _p(th, "ks_green") else A


def oprisk_capital_stability(th: Any, row: Any, field: str = "") -> Light:
    """``oprisk.capital_stability``: the worse of the dispersion (metric) and the bias (``|statistic|``) gates."""
    dispersion, bias = _num(row, "metric"), _num(row, "statistic")
    if dispersion is None or bias is None:
        return N
    return _worse(
        _upper_le(dispersion, _p(th, "dispersion_green"), _p(th, "dispersion_amber")),
        _upper_le(abs(bias), _p(th, "bias_green"), _p(th, "bias_amber")),
    )


def stress_satellite(th: Any, row: Any, field: str = "") -> Light:
    """``stress.satellite_backcast``: green when ``|1 - peak_ratio| <= peak_amber`` and the timing error (metric) is
    within ``timing_quarters``; amber when ``|1 - peak_ratio| <= peak_red``; red otherwise."""
    ratio, timing = _num(row, "extras.peak_ratio"), _num(row, "metric")
    if ratio is None or timing is None:
        return N
    dev = abs(1.0 - ratio)
    if dev <= _p(th, "peak_amber") + EPS and abs(timing) <= _p(th, "timing_quarters"):
        return G
    return A if dev <= _p(th, "peak_red") + EPS else R


def stress_severity(th: Any, row: Any, field: str = "") -> Light:
    """``stress.scenario_severity``: green at or above ``minimum``, amber below (no red)."""
    v = _num(row, "statistic")
    return N if v is None else (G if v >= _p(th, "minimum") - EPS else A)


def stress_reverse_solution(th: Any, row: Any, field: str = "") -> Light:
    """``stress.reverse_solution_check``: red when the statistic is above ``tolerance`` or a start is closer."""
    v = _num(row, "statistic")
    if v is None:
        return N
    return R if (v > _p(th, "tolerance") or _get(row, "extras.closer_start")) else G


def liquidity_htm_loss(th: Any, row: Any, field: str = "") -> Light:
    """``liq.htm_loss_effect``: green when the unrealised loss does not raise the CBC (statistic at least 0) and does
    not make a surviving scenario survive the horizon (``extras.survives_horizon``). The survival periods themselves
    are not in the row, so a lengthening between two finite periods is not visible to this rule."""
    removed = _num(row, "statistic")
    if removed is None:
        return N
    surv = _get(row, "extras.survives_horizon") or {}
    lengthened = bool(surv.get("with")) and not bool(surv.get("without"))
    return G if (removed >= -EPS and not lengthened) else R


def liquidity_haircut(th: Any, row: Any, field: str = "") -> Light:
    """``liq.cbc_haircut_rules``: green when the largest deviation is within ``tolerance`` and no label is wrong."""
    stat, wrong = _num(row, "statistic"), _num(row, "extras.wrong_labels")
    if stat is None or wrong is None:
        return N
    return G if (stat <= _p(th, "tolerance") + EPS and wrong == 0) else R


def amber_if_nonempty(th: Any, row: Any, field: str) -> Light:
    """Amber when the field (a list of flagged groups) is not empty, green when it is an empty list; never red."""
    v = _get(row, field)
    return N if v is None else (A if len(v) else G)


def amber_if_false(th: Any, row: Any, field: str) -> Light:
    """Amber when the field (a boolean, for example ``same_sign``) is false, green when it is true; never red."""
    v = _get(row, field)
    return N if v is None else (G if v else A)


def fair_interval_excludes_zero(th: Any, row: Any, field: str = "") -> Light:
    """``fair.conditional_disparity`` and ``fair.rate_spread_disparity``: amber when the Wald interval of the effect
    (``extras.effect.lo`` and ``hi``, at the level the row states) excludes zero; never red."""
    lo, hi = _num(row, "extras.effect.lo"), _num(row, "extras.effect.hi")
    if lo is None or hi is None:
        return N
    return A if (lo > 0.0 or hi < 0.0) else G


def ml_interval_above_zero(th: Any, row: Any, field: str = "") -> Light:
    """``ml.challenger_brier``: amber when the lower end of the paired-Brier interval (``extras.lo``) is above zero (the
    challenger is worse); green otherwise, a better challenger is not a finding."""
    lo = _num(row, "extras.lo")
    return N if lo is None else (A if lo > 0.0 else G)


def always_not_evaluated(th: Any, row: Any, field: str = "") -> Light:
    """``ml.cbpe_estimate``: an estimate, never a test, so never a colour."""
    return N


def worst_of_screens(th: Any, row: Any, field: str = "") -> Light:
    """``ml.leakage_screen`` needs the sub-entries of its policy: see ``Policy.classify_row``."""
    raise ValueError("the rule worst_of_screens reads the policy sub-entries: call Policy.classify_row")


GENERIC: dict[str, Callable[[Any, Any, str], Light]] = {
    "upper_le": upper_le,
    "lower_ge": lower_ge,
    "abs_upper_le": abs_upper_le,
    "red_if_positive": red_if_positive,
    "red_if_nonempty": red_if_nonempty,
    "always_green": always_green,
    "amber_if_nonempty": amber_if_nonempty,
    "amber_if_false": amber_if_false,
}

NAMED: dict[str, Callable[[Any, Any, str], Light]] = {
    f.__name__: f
    for f in (
        aggregation_sum_vc, aggregation_corr_stress, aggregation_diversification, ccr_path_convergence,
        ccr_pit_simulated, ifrs9_lifetime_pd, ifrs9_nonlinearity, ifrs9_sicr_effectiveness, ifrs9_sicr_stability,
        ifrs9_scenario_sensitivity, ifrs9_ecl_backtest, ifrs9_overlay, irrbb_eve_with_without, irrbb_srp31_92,
        p_value_negative_red, portfolio_size_power, market_pla_zone, oprisk_capital_stability, stress_satellite,
        stress_severity, stress_reverse_solution, liquidity_htm_loss, liquidity_haircut, fair_interval_excludes_zero,
        ml_interval_above_zero, always_not_evaluated, worst_of_screens,
    )
}

# Tests whose policy states no threshold (contracts engine-fairness-api 1.0.0 and engine-ml-api 2.0.0): the rule is the
# whole classification, so the packaged default holds no entry for them. test id -> (rule, the rule's constants)
RULE_ONLY: dict[str, tuple[str, dict[str, float]]] = {
    "fair.adverse_impact": ("amber_if_nonempty:extras.flagged_groups", {"four_fifths": 0.8}),
    "fair.conditional_disparity": ("fair_interval_excludes_zero", {}),
    "fair.rate_spread_disparity": ("fair_interval_excludes_zero", {}),
    "fair.replication": ("amber_if_false:extras.same_sign", {}),
    "ml.challenger_brier": ("ml_interval_above_zero", {}),
    "ml.cbpe_estimate": ("always_not_evaluated", {}),
    "ml.leakage_screen": ("worst_of_screens", {}),
}

RULES = tuple(sorted({*GENERIC, *NAMED}))


def classify_rule(rule: str, th: Any, row: Any) -> Light:
    """The light of ``row`` under the rule ``name`` or ``name:field`` and the effective entry ``th``."""
    name, _, field = rule.partition(":")
    fn = GENERIC.get(name) or NAMED.get(name)
    if fn is None:
        raise KeyError(f"unknown policy rule {rule!r}; the rules are {list(RULES)}")
    return fn(th, row, field)
