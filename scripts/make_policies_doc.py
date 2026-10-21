#!/usr/bin/env python3
"""Regenerate the rules table and the battery tables of ``docs/policies.md`` from the code and the packaged files.

The prose of the page is by hand. The block between ``<!-- rules:begin -->`` and ``<!-- rules:end -->`` states every rule
of ``riskvalidation.policies.rules`` in plain words with the cut points it reads; the block between
``<!-- batteries:begin -->`` and ``<!-- batteries:end -->`` lists every entry of every packaged battery file.
``tests/test_policy_format.py`` fails when a rule is missing from the table or the page is out of date.

Usage: python scripts/make_policies_doc.py [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "policies.md"
POLICIES = ROOT / "src" / "riskvalidation" / "policies"

# rule -> (reads, cut points, the light). Comparisons with `=` carry the 1e-12 rounding tolerance `rules.EPS` unless noted.
RULE_TEXT: dict[str, tuple[str, str, str]] = {
    "upper_le:F": (
        "the field F",
        "`amber`, `red` of the entry",
        "green when F is at most `amber`, amber when at most `red` (both with the 1e-12 tolerance), red above. "
        "A value equal to `amber` is green.",
    ),
    "lower_ge:F": (
        "the field F",
        "`amber`, `red`",
        "green when F is at least `amber`, amber when at least `red` (1e-12 tolerance), red below.",
    ),
    "abs_upper_le:F": ("the field F", "`amber`, `red`", "`upper_le` applied to the absolute value of F."),
    "red_if_positive:F": ("the field F (a count or a violation)", "none", "red when F is above 0, green otherwise."),
    "red_if_nonempty:F": (
        "the field F (a list of failures)", "none", "red when the list is not empty, green when it is empty."
    ),
    "always_green": ("nothing", "none", "green: a reference row shown for context, never a gate."),
    "amber_if_nonempty:F": (
        "the field F (a list of flagged groups)", "none", "amber when the list is not empty, green when empty; never red."
    ),
    "amber_if_false:F": (
        "the field F (a boolean)", "none", "amber when F is false, green when true; never red."
    ),
    "aggregation_sum_vc": (
        "`metric` (the ratio of the sum), `statistic` (the variance-covariance ratio)",
        "`tolerance`, `amber`",
        "red when the ratio is above 1 + `tolerance`; amber when |statistic - 1| is above `amber`; green otherwise.",
    ),
    "aggregation_corr_stress": (
        "`statistic` (the change of capital)", "`tolerance`, `amber`",
        "red when the statistic is below -`tolerance`; amber when above `amber`; green otherwise.",
    ),
    "aggregation_diversification": (
        "`statistic` (the largest factor)", "`tolerance`", "red when the statistic is above 1 + `tolerance`, green otherwise."
    ),
    "ccr_path_convergence": (
        "`statistic` (EE change), `metric` (CVA change)", "`k_se`",
        "the larger of the two is green at most `k_se`, amber at most 2 `k_se` (1e-12 tolerance), red above.",
    ),
    "ccr_pit_simulated": (
        "`statistic` (the Anderson-Darling W2), `extras.simulated_critical_values` (`p05`, `p01`)", "none (the critical values are data of the case)",
        "green when W2 is at most `p05`, amber when at most `p01`, red above (1e-12 tolerance).",
    ),
    "ifrs9_lifetime_pd": (
        "`extras.observed` with the band (`band_low`, `band_high`, `band_low_red`, `band_high_red`), or `extras.relative_gap`",
        "`gap_amber`, `gap_red` (the band alphas are in the row)",
        "with a band: red when an observed rate is outside the red band, amber outside the amber band, green inside; "
        "without one: the worst relative gap is green below `gap_amber`, amber below `gap_red`, red otherwise "
        "(strict comparisons).",
    ),
    "ifrs9_nonlinearity": (
        "`metric`, `extras.gauss_hermite_error`", "`gauss_hermite_tolerance`",
        "green when the Gauss-Hermite error is at most the tolerance and the metric is at least 0, red otherwise.",
    ),
    "ifrs9_sicr_effectiveness": (
        "`metric` (default-rate ratio), `extras.share_stage3_direct_from_stage1`", "`green_ratio`, `amber_ratio`, `direct_share_amber`",
        "green when the ratio is at least `green_ratio`, amber when at least `amber_ratio`, red below; a green becomes "
        "amber when the direct Stage 3 share is above `direct_share_amber`.",
    ),
    "ifrs9_sicr_stability": (
        "`extras.loosening_periods`, `extras.periods`", "`red_share_of_periods`",
        "green with no loosening period; red when their count is at least `periods` x `red_share_of_periods`; amber "
        "otherwise; not evaluated with fewer than two periods.",
    ),
    "ifrs9_scenario_sensitivity": (
        "`metric` (the nonlinearity index)", "`nl_amber`, `nl_red`",
        "red above `nl_red`, amber above `nl_amber` or below 0, green otherwise.",
    ),
    "ifrs9_ecl_backtest": (
        "`metric` (R_H)", "`r_green_low`, `r_green_high`, `r_amber_low`, `r_amber_high`",
        "green when R_H is inside the green range, amber inside the amber range, red outside (ranges closed).",
    ),
    "ifrs9_overlay": (
        "`metric` (the overlay share), `extras.direction_consistent`", "`share_amber`, `share_red`",
        "red above `share_red`; amber above `share_amber` or when the direction is inconsistent; green otherwise.",
    ),
    "irrbb_eve_with_without": (
        "`statistic`, `extras.hidden_breach`", "`diff_amber`, `diff_red`",
        "red when the behavioural assumptions hide a breach; otherwise `upper_le` on |statistic| with `diff_amber` and `diff_red`.",
    ),
    "irrbb_srp31_92": (
        "`extras.deviations` (by name)", "`tolerance_scalar` (for `alpha_short`), `tolerance_bp` (for the shocks)",
        "green when every deviation is at most its tolerance (1e-12 tolerance), red otherwise.",
    ),
    "p_value_negative_red": (
        "`p_value`, `extras.negative_entries`", "`amber`, `red` of the entry",
        "red when the recursion has a negative entry; otherwise green when p is at least `amber`, amber when at least `red`, red below.",
    ),
    "portfolio_size_power": (
        "`extras.role`, `p_value`, `metric`", "`amber`, `red` (size cells), `power_green`, `power_amber` (power cells)",
        "a size cell: green when p is at least `amber`, amber when at least `red`, red below; a power cell (role `power`): "
        "`lower_ge` on the rejection rate with `power_green` and `power_amber`.",
    ),
    "market_basel_zone": (
        "`statistic` (exceptions k), `extras.window`, `extras.coverage`", "`amber_level` (0.95), `red_level` (0.9999)",
        "with X ~ Binomial(window, 1 - coverage): amber from the smallest count whose P(X <= k) reaches `amber_level`, red "
        "from the one that reaches `red_level`, green below. 250 days at 99 percent: green 0 to 4, amber 5 to 9, red 10 or more (MAR32.9 Table 1).",
    ),
    "market_pla_zone": (
        "`statistic` (Spearman), `metric` (KS)", "`spearman_green` 0.80, `spearman_red` 0.70, `ks_green` 0.09, `ks_red` 0.12",
        "red when Spearman is below `spearman_red` or KS above `ks_red`; green when Spearman is above `spearman_green` "
        "and KS below `ks_green`; amber otherwise (MAR32.42; strict comparisons).",
    ),
    "oprisk_capital_stability": (
        "`metric` (dispersion), `statistic` (bias)", "`dispersion_green`, `dispersion_amber`, `bias_green`, `bias_amber`",
        "the worse of `upper_le` on the dispersion and `upper_le` on |bias|.",
    ),
    "stress_satellite": (
        "`extras.peak_ratio`, `metric` (timing error)", "`peak_amber`, `peak_red`, `timing_quarters`",
        "green when |1 - peak ratio| is at most `peak_amber` and |timing| at most `timing_quarters`; amber when "
        "|1 - peak ratio| is at most `peak_red`; red otherwise.",
    ),
    "stress_severity": (
        "`statistic`", "`minimum`", "green when the statistic is at least `minimum`, amber below; never red."
    ),
    "stress_reverse_solution": (
        "`statistic`, `extras.closer_start`", "`tolerance`",
        "red when the statistic is above `tolerance` or a start is closer; green otherwise.",
    ),
    "liquidity_htm_loss": (
        "`statistic`, `extras.survives_horizon`", "none",
        "green when the unrealised loss does not raise the counterbalancing capacity (statistic at least 0) and does not "
        "make a surviving scenario survive the horizon; red otherwise.",
    ),
    "liquidity_haircut": (
        "`statistic`, `extras.wrong_labels`", "`tolerance`",
        "green when the largest deviation is at most `tolerance` and no label is wrong; red otherwise.",
    ),
    "fair_interval_excludes_zero": (
        "`extras.effect.lo`, `extras.effect.hi`", "none", "amber when the interval excludes zero, green otherwise; never red."
    ),
    "ml_interval_above_zero": (
        "`extras.lo`", "none", "amber when the lower end of the paired-Brier interval is above 0, green otherwise."
    ),
    "always_not_evaluated": ("nothing", "none", "not evaluated: an estimate, never a colour."),
    "worst_of_screens": (
        "`extras.screens`", "the entries `ml.leakage_screen.*`", "the worst light of the screens, each read by the entry of its own key; needs the policy (`Policy.classify_row`)."
    ),
}


def _rules_block() -> str:
    from riskvalidation.policies.rules import GENERIC, NAMED

    names = {f"{n}:F" if n not in ("always_green",) else n for n in GENERIC} | set(NAMED)
    missing = names - set(RULE_TEXT)
    extra = set(RULE_TEXT) - names
    if missing or extra:
        raise SystemExit(f"RULE_TEXT is out of step with rules.py: missing {sorted(missing)}, extra {sorted(extra)}")
    rows = ["| Rule | Reads | Cut points | The light |", "|---|---|---|---|"]
    for name in sorted(RULE_TEXT):
        reads, cuts, light = RULE_TEXT[name]
        rows.append(f"| `{name}` | {reads} | {cuts} | {light} |")
    return "\n".join(rows)


def _batteries_block() -> str:
    out: list[str] = []
    for f in sorted(POLICIES.glob("contraste-*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        label = f"{doc['policy_id']} {doc['version']}"
        out.append(f"### `{label}`\n")
        out.append(f"File `policies/{f.name}`.\n")
        out.append("| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |")
        out.append("|---|---|---|---|---|---|---|---|")
        for tid, th in doc["thresholds"].items():
            params = ", ".join(f"{k}={v:g}" for k, v in sorted(th.get("parameters", {}).items()))
            keyed = ""
            if th.get("by"):
                cases = "; ".join(
                    f"`{k}` -> " + ", ".join(f"{n}={x:g}" for n, x in v.items()) for k, v in th["cases"].items()
                )
                keyed = f"`{th['by']}`: {cases}"
            src = "regulatory" if th["source"].startswith("regulatory:") else "convention"
            rule = f"`{th['rule']}`" if th.get("rule") else "two cut points"
            out.append(
                f"| `{tid}` | {th['kind']} | {rule} | {th['amber']:g} | {th['red']:g} | {params} | {keyed} | {src} |"
            )
        out.append("")
    return "\n".join(out)


def _replace(text: str, tag: str, block: str) -> str:
    begin, end = f"<!-- {tag}:begin -->", f"<!-- {tag}:end -->"
    i, j = text.index(begin), text.index(end)
    return text[: i + len(begin)] + "\n\n" + block + "\n\n" + text[j:]


def build(text: str) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    return _replace(_replace(text, "rules", _rules_block()), "batteries", _batteries_block())


def main() -> int:
    text = PAGE.read_text(encoding="utf-8")
    new = build(text)
    if "--check" in sys.argv:
        if new != text:
            print("docs/policies.md is out of date: run python scripts/make_policies_doc.py")
            return 1
        print("docs/policies.md is current")
        return 0
    PAGE.write_text(new, encoding="utf-8", newline="\n")
    print("wrote", PAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
