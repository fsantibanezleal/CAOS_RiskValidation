#!/usr/bin/env python3
"""Regenerate the computed documentation figures (deterministic; NumPy and SciPy only).

- docs/assets/psi-benchmark.svg: the Yurdakul-Naranjo critical PSI against sample size, with the 0.10 and 0.25
  conventional bands.
- docs/assets/roc-cap.svg: ROC and CAP curves of a binormal score with the areas behind AR = 2 AUC - 1.
- docs/assets/irb-risk-weights.svg, lhp-distribution.svg, most-prudent-bounds.svg, pd-curve-sp.svg: the U2
  engines drawn by the package itself against their published values (CRE99 Table 1, Pluto and Tasche, Tasche
  2013); theme-aware.
- docs/assets/binomial-true-size.svg, traffic-lights-ties.svg, hhi-statistic.svg (exact), size-of-every-test.svg and
  published-reproductions.svg (the harness on the suite's own scenarios and seeds, a few minutes): U3; theme-aware.
- docs/assets/transition-estimators.svg, markov-test-forms.svg (the harness at the size study's seeds),
  markov-test-power.svg (the harness against planted momentum and a recession regime), interval-coverage.svg (exact),
  ttc-drift.svg, rating-momentum.svg: U4, on the size study's Markov chain; theme-aware.

Usage: python scripts/make_doc_figures.py [figure.svg ...] (only the computed U3 and U4 figures named; all by default)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
sys.path.insert(0, str(ROOT / "src"))  # the U2 figures draw the package's own engines

STYLE = """
  <style>
    .bg { fill: #f8fafc; stroke: #cbd5e1; }
    .ax { stroke: #334155; stroke-width: 1.2; }
    .grid { stroke: #e2e8f0; stroke-width: 1; }
    .t { font: 600 14px system-ui, sans-serif; fill: #0f172a; }
    .s { font: 12px system-ui, sans-serif; fill: #334155; }
    .k { font: 11px system-ui, sans-serif; fill: #475569; }
    .c1 { fill: none; stroke: #4338ca; stroke-width: 2.4; }
    .c2 { fill: none; stroke: #0f766e; stroke-width: 2.4; }
    .c3 { fill: none; stroke: #c2410c; stroke-width: 1.8; stroke-dasharray: 6 4; }
    .c4 { fill: none; stroke: #b91c1c; stroke-width: 1.8; stroke-dasharray: 6 4; }
    .area1 { fill: #4338ca; fill-opacity: 0.12; }
    .area2 { fill: #0f766e; fill-opacity: 0.12; }
  </style>"""


def _poly(xs, ys, fx, fy) -> str:
    return " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys, strict=True))


def psi_benchmark_svg() -> str:
    w, h, ml, mr, mt, mb = 760, 380, 70, 30, 46, 54
    ns = np.unique(np.round(np.logspace(np.log10(50), np.log10(5000), 120))).astype(int)
    crit05 = (2.0 / ns) * stats.chi2.ppf(0.95, 9)
    crit01 = (2.0 / ns) * stats.chi2.ppf(0.99, 9)
    ymax = 1.0
    fx = lambda n: ml + (np.log10(n) - np.log10(50)) / (np.log10(5000) - np.log10(50)) * (w - ml - mr)  # noqa: E731
    fy = lambda y: h - mb - min(y, ymax) / ymax * (h - mt - mb)  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           'aria-labelledby="t d">',
           '  <title id="t">Critical PSI against sample size</title>',
           '  <desc id="d">The PSI value that a chi-square benchmark rejects at 5% and 1% for ten bins and equal '
           'development and current samples falls with sample size, while the conventional 0.10 and 0.25 bands '
           'stay flat.</desc>', STYLE,
           f'  <rect class="bg" x="1" y="1" width="{w - 2}" height="{h - 2}" rx="10"/>',
           f'  <text class="t" x="{ml}" y="28">Critical PSI, ten bins, equal samples n = m (Yurdakul and Naranjo 2020)'
           '</text>']
    for yv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{yv:.1f}</text>')
    for nv in (50, 100, 200, 400, 1000, 2000, 5000):
        out.append(f'  <line class="grid" x1="{fx(nv):.1f}" x2="{fx(nv):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(nv):.1f}" y="{h - mb + 18}" text-anchor="middle">{nv}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 12}" text-anchor="middle">sample size n = m '
               '(log scale)</text>')
    out.append(f'  <polyline class="c1" points="{_poly(ns, crit05, fx, fy)}"/>')
    out.append(f'  <polyline class="c2" points="{_poly(ns, crit01, fx, fy)}"/>')
    for yv, cls, label in ((0.10, "c3", "0.10 rule of thumb"), (0.25, "c4", "0.25 rule of thumb")):
        out.append(f'  <line class="{cls}" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{w - mr - 4}" y="{fy(yv) - 6:.1f}" text-anchor="end">{label}</text>')
    for nv in (100, 200, 400, 1000):
        cv = 2.0 / nv * stats.chi2.ppf(0.95, 9)
        out.append(f'  <circle cx="{fx(nv):.1f}" cy="{fy(cv):.1f}" r="4" fill="#4338ca"/>')
        out.append(f'  <text class="k" x="{fx(nv) + 6:.1f}" y="{fy(cv) - 6:.1f}">{cv:.3f}</text>')
    lx, ly = w - mr - 250, mt + 18
    out.append(f'  <line class="c1" x1="{lx}" x2="{lx + 24}" y1="{ly}" y2="{ly}"/><text class="s" x="{lx + 30}" '
               f'y="{ly + 4}">chi-square benchmark, alpha 5%</text>')
    out.append(f'  <line class="c2" x1="{lx}" x2="{lx + 24}" y1="{ly + 20}" y2="{ly + 20}"/><text class="s" '
               f'x="{lx + 30}" y="{ly + 24}">chi-square benchmark, alpha 1%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def roc_cap_svg() -> str:
    w, h = 820, 400
    mu, pi = 1.19, 0.2  # binormal separation giving AUC near 0.80; default share for the CAP
    auc = float(stats.norm.cdf(mu / np.sqrt(2.0)))
    thr = np.linspace(-6, 7, 600)
    fpr = stats.norm.sf(thr)  # survivors ~ N(0,1), defaulters ~ N(mu,1); higher score = riskier
    tpr = stats.norm.sf(thr - mu)
    xs_roc, ys_roc = fpr[::-1], tpr[::-1]
    pop = pi * tpr + (1 - pi) * fpr
    xs_cap, ys_cap = pop[::-1], tpr[::-1]

    def panel(x0, title):
        size = 300
        fx = lambda x: x0 + 40 + x * size  # noqa: E731
        fy = lambda y: 60 + (1 - y) * size  # noqa: E731
        return fx, fy, size, title

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           'aria-labelledby="t d">',
           '  <title id="t">ROC and CAP curves and the accuracy ratio</title>',
           f'  <desc id="d">For a binormal score with AUC {auc:.2f}, the ROC curve plots the share of defaulters '
           'captured against the share of non-defaulters; the CAP plots it against the share of the whole '
           'population. AR is the CAP area above the diagonal divided by the perfect model area, and equals '
           '2 AUC minus 1.</desc>', STYLE,
           f'  <rect class="bg" x="1" y="1" width="{w - 2}" height="{h - 2}" rx="10"/>']
    for x0, title, xs, ys, cls, area, xl in (
        (20, f"ROC curve, AUC = {auc:.3f}", xs_roc, ys_roc, "c1", "area1", "non-defaulters ranked riskier (FPR)"),
        (420, f"CAP curve, AR = 2 AUC - 1 = {2 * auc - 1:.3f}", xs_cap, ys_cap, "c2", "area2",
         "population ranked riskiest first"),
    ):
        fx, fy, size, _ = panel(x0, title)
        out.append(f'  <text class="t" x="{x0 + 40}" y="40">{title}</text>')
        out.append(f'  <rect x="{fx(0):.1f}" y="{fy(1):.1f}" width="{size}" height="{size}" fill="#ffffff" '
                   'stroke="#334155" stroke-width="1.2"/>')
        poly = _poly(np.concatenate([xs, [1.0, 0.0]]), np.concatenate([ys, [0.0, 0.0]]), fx, fy)
        out.append(f'  <polygon class="{area}" points="{poly}"/>')
        out.append(f'  <line class="c3" x1="{fx(0):.1f}" y1="{fy(0):.1f}" x2="{fx(1):.1f}" y2="{fy(1):.1f}"/>')
        out.append(f'  <polyline class="{cls}" points="{_poly(xs, ys, fx, fy)}"/>')
        out.append(f'  <text class="k" x="{fx(0.5):.1f}" y="{fy(0) + 20:.1f}" text-anchor="middle">{xl}</text>')
        out.append(f'  <text class="k" x="{fx(0) - 10:.1f}" y="{fy(0.5):.1f}" text-anchor="middle" '
                   f'transform="rotate(-90 {fx(0) - 10:.1f} {fy(0.5):.1f})">defaulters captured</text>')
        if cls == "c2":
            perfect = _poly([0.0, pi, 1.0], [0.0, 1.0, 1.0], fx, fy)
            out.append(f'  <polyline class="c4" points="{perfect}"/>')
            out.append(f'  <text class="k" x="{fx(pi) + 10:.1f}" y="{fy(1) + 30:.1f}">perfect model (default share '
                       f'{pi:.0%})</text>')
        else:
            out.append(f'  <text class="k" x="{fx(0.55):.1f}" y="{fy(0.35):.1f}">random ranking</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# U2 figures: drawn from the package's own engines, so a figure cannot disagree with the code it documents. Theme-aware:
# light by default, the shared dark palette under prefers-color-scheme.

THEMED = """
  <style>
    .bg { fill: #f8fafc; stroke: #cbd5e1; }
    .pn { fill: #ffffff; stroke: #94a3b8; }
    .ax { stroke: #334155; stroke-width: 1.2; }
    .grid { stroke: #e2e8f0; stroke-width: 1; }
    .t { font: 600 14px system-ui, sans-serif; fill: #0f172a; }
    .s { font: 12px system-ui, sans-serif; fill: #334155; }
    .k { font: 11px system-ui, sans-serif; fill: #475569; }
    .l0, .l1, .l2, .l3, .l4 { fill: none; stroke-width: 2.2; }
    .l0 { stroke: #0f172a; }
    .l1 { stroke: #4338ca; } .l2 { stroke: #0f766e; } .l3 { stroke: #c2410c; } .l4 { stroke: #7c3aed; }
    .dash { stroke-dasharray: 7 4; }
    .p0 { fill: #0f172a; } .p1 { fill: #4338ca; } .p2 { fill: #0f766e; } .p3 { fill: #c2410c; } .p4 { fill: #7c3aed; }
    .hollow { fill: none; stroke: #0f172a; stroke-width: 1.4; }
    .ref { stroke: #64748b; stroke-width: 1.2; stroke-dasharray: 4 4; fill: none; }
    @media (prefers-color-scheme: dark) {
      .bg { fill: #0d1117; stroke: #30363d; }
      .pn { fill: #161b22; stroke: #30363d; }
      .ax { stroke: #8b949e; }
      .grid { stroke: #21262d; }
      .t, .s { fill: #c9d1d9; }
      .k { fill: #8b949e; }
      .l0 { stroke: #c9d1d9; }
      .l1 { stroke: #58a6ff; } .l2 { stroke: #3fb950; } .l3 { stroke: #d29922; } .l4 { stroke: #bc8cff; }
      .p0 { fill: #c9d1d9; } .p1 { fill: #58a6ff; } .p2 { fill: #3fb950; } .p3 { fill: #d29922; } .p4 { fill: #bc8cff; }
      .hollow { stroke: #c9d1d9; }
      .ref { stroke: #8b949e; }
    }
  </style>"""


def _open(w: int, h: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        'aria-labelledby="t d">',
        f'  <title id="t">{title}</title>',
        f'  <desc id="d">{desc}</desc>',
        THEMED,
        f'  <rect class="bg" x="1" y="1" width="{w - 2}" height="{h - 2}" rx="10"/>',
    ]


def _pct(v: float) -> str:
    s = f"{100 * v:.4f}".rstrip("0").rstrip(".")
    return f"{s}%"


def _legend(out: list[str], x: float, y: float, cls: str, text: str, *, dot: bool = False) -> None:
    if dot:
        out.append(f'  <circle class="{cls}" cx="{x + 14}" cy="{y}" r="3"/>')
    else:
        out.append(f'  <line class="{cls}" x1="{x}" x2="{x + 28}" y1="{y}" y2="{y}"/>')
    out.append(f'  <text class="s" x="{x + 34}" y="{y + 4}">{text}</text>')


def irb_risk_weights_svg() -> str:
    from riskvalidation.regulatory import capital_requirement

    fix = json.loads((ROOT / "tests" / "fixtures" / "cre99_table1.json").read_text(encoding="utf-8"))
    w, h, ml, mr, mt, mb = 800, 440, 64, 212, 48, 56
    x_lo, x_hi, y_max = math.log10(0.0003), math.log10(0.3), 300.0
    fx = lambda pd: ml + (math.log10(pd) - x_lo) / (x_hi - x_lo) * (w - ml - mr)  # noqa: E731
    fy = lambda rw: h - mb - min(rw, y_max) / y_max * (h - mt - mb)  # noqa: E731
    style = {"corp_s50": ("l1", "p1", ""), "corp_s5": ("l1", "p1", " dash"), "mortgage_45": ("l2", "p2", ""),
             "mortgage_25": ("l2", "p2", " dash"), "other_45": ("l3", "p3", ""), "other_85": ("l3", "p3", " dash"),
             "qrre_50": ("l4", "p4", ""), "qrre_85": ("l4", "p4", " dash")}
    label = {"corp_s50": "corporate, LGD 40%, S = 50", "corp_s5": "corporate, LGD 40%, S = 5",
             "mortgage_45": "mortgage, LGD 45%", "mortgage_25": "mortgage, LGD 25%",
             "other_45": "other retail, LGD 45%", "other_85": "other retail, LGD 85%", "qrre_50": "QRRE, LGD 50%",
             "qrre_85": "QRRE, LGD 85%"}
    out = _open(w, h, "IRB risk weights against PD, and the CRE99 Table 1 cells",
                "Risk weights of the Basel III final IRB functions at maturity 2.5 years for the eight columns of BCBS "
                "CRE99 Table 1, computed by riskvalidation; the dots are the 144 printed cells. Below the 0.05% PD "
                "floor every curve is flat.")
    out.append(f'  <text class="t" x="{ml}" y="28">IRB risk weight (% of EAD), Basel III final, M = 2.5 years</text>')
    for yv in range(0, 301, 50):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{yv}</text>')
    for pv in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        out.append(f'  <line class="grid" x1="{fx(pv):.1f}" x2="{fx(pv):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(pv):.1f}" y="{h - mb + 18}" text-anchor="middle">{_pct(pv)}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">PD (log scale)</text>')
    out.append(f'  <line class="ref" x1="{fx(0.0005):.1f}" x2="{fx(0.0005):.1f}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="k" x="{fx(0.0005) + 4:.1f}" y="{mt + 12}">PD floor 0.05%</text>')
    pds = np.logspace(x_lo, x_hi, 240)
    for i, col in enumerate(fix["columns"]):
        line, dot, dash = style[col["id"]]

        def rw(pd: float, col: dict = col) -> float:
            res = capital_requirement(col["asset_class"], float(pd), col["lgd"], maturity=col.get("maturity", 2.5),
                                      sales_eur_m=col.get("sales_eur_m"))
            return 100.0 * res.risk_weight

        pts = " ".join(f"{fx(p):.1f},{fy(rw(p)):.1f}" for p in pds)
        out.append(f'  <polyline class="{line}{dash}" points="{pts}"/>')
        for row in fix["rows"]:
            out.append(f'  <circle class="{dot}" cx="{fx(row["pd"]):.1f}" cy="{fy(row["rw_pct"][i]):.1f}" r="2.6"/>')
        _legend(out, w - mr + 16, mt + 22 + 22 * i, f"{line}{dash}", label[col["id"]])
    _legend(out, w - mr + 16, mt + 22 + 22 * 8 + 6, "p0", "CRE99 Table 1 cell", dot=True)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def lhp_distribution_svg() -> str:
    from riskvalidation.engines import vasicek

    w, h, ml, mr, mt, mb = 800, 440, 70, 250, 48, 56
    pd, x_max = 0.01, 0.25
    d_lo, d_hi = -5, 0  # decades of the tail probability
    fx = lambda x: ml + x / x_max * (w - ml - mr)  # noqa: E731

    def fy(p: float) -> float:
        lp = math.log10(min(max(p, 10.0**d_lo), 1.0))
        return h - mb - (lp - d_lo) / (d_hi - d_lo) * (h - mt - mb)

    out = _open(w, h, "The large homogeneous portfolio: tail of the default fraction and its 99.9% quantile",
                "Probability that the default fraction of an infinitely granular portfolio with PD 1% exceeds a level, "
                "for asset correlations 4%, 12% and 24%, on a log scale. Each curve crosses the one-in-a-thousand line "
                "at its 99.9% quantile, the conditional PD the IRB formula charges.")
    out.append(f'  <text class="t" x="{ml}" y="28">Default fraction L of the large homogeneous portfolio, PD = 1%: '
               'P(L &gt; x)</text>')
    for lv in range(d_lo, d_hi + 1):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(10.0**lv):.1f}" y2="{fy(10.0**lv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(10.0**lv) + 4:.1f}" text-anchor="end">{10.0**lv:g}</text>')
    for xv in np.linspace(0.0, x_max, 6):
        out.append(f'  <line class="grid" x1="{fx(xv):.1f}" x2="{fx(xv):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(xv):.1f}" y="{h - mb + 18}" text-anchor="middle">{100 * xv:.0f}%</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">default fraction x '
               '(share of the portfolio)</text>')
    out.append(f'  <line class="ref" x1="{ml}" x2="{w - mr}" y1="{fy(1e-3):.1f}" y2="{fy(1e-3):.1f}"/>')
    out.append(f'  <text class="k" x="{w - mr - 4}" y="{fy(1e-3) - 6:.1f}" text-anchor="end">0.001: one year in a '
               'thousand</text>')
    out.append(f'  <line class="ref" x1="{fx(pd):.1f}" x2="{fx(pd):.1f}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="k" x="{fx(pd) + 4:.1f}" y="{mt + 12}">mean = PD</text>')
    xs = np.linspace(1e-4, x_max, 600)
    names = {0.04: "4% (QRRE)", 0.12: "12% (corporate, high PD)", 0.24: "24% (corporate, low PD)"}
    for i, rho in enumerate((0.04, 0.12, 0.24), start=1):
        tail = 1.0 - vasicek.lhp_cdf(xs, pd, rho)
        keep = tail >= 10.0**d_lo
        pts = " ".join(f"{fx(x):.1f},{fy(p):.1f}" for x, p in zip(xs[keep], tail[keep], strict=True))
        out.append(f'  <polyline class="l{i}" points="{pts}"/>')
        q = float(vasicek.lhp_quantile(0.999, pd, rho))
        out.append(f'  <line class="l{i} dash" x1="{fx(q):.1f}" x2="{fx(q):.1f}" y1="{fy(1e-3):.1f}" '
                   f'y2="{h - mb}"/>')
        out.append(f'  <circle class="p{i}" cx="{fx(q):.1f}" cy="{fy(1e-3):.1f}" r="4"/>')
        _legend(out, w - mr + 16, mt + 22 + 40 * (i - 1), f"l{i}", f"correlation {names[rho]}")
        out.append(f'  <text class="k" x="{w - mr + 50}" y="{mt + 40 + 40 * (i - 1)}">99.9% quantile '
                   f'{100 * q:.2f}%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def most_prudent_svg() -> str:
    from riskvalidation.engines.low_default import most_prudent_pd

    published = {  # Pluto and Tasche (2005), Tables 1 to 3 (independent) and 7 (rho 12%), zero defaults, percent
        0.0: ((0.09, 0.17, 0.29, 0.37, 0.57, 0.86), (0.10, 0.20, 0.33, 0.43, 0.66, 0.98),
              (0.23, 0.46, 0.76, 0.99, 1.52, 2.28)),
        0.12: ((0.15, 0.40, 0.86, 1.31, 2.65, 5.29), (0.17, 0.45, 0.96, 1.45, 2.92, 5.77),
               (0.37, 0.92, 1.89, 2.78, 5.30, 9.84)),
    }
    gammas = (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
    w, h, ml, mr, mt, mb = 800, 440, 64, 220, 48, 56
    t_of = lambda g: -math.log10(1.0 - g)  # noqa: E731
    t_lo, t_hi, y_max = t_of(0.5), t_of(0.999), 10.0
    fx = lambda g: ml + (t_of(g) - t_lo) / (t_hi - t_lo) * (w - ml - mr)  # noqa: E731
    fy = lambda y: h - mb - min(y, y_max) / y_max * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "Most prudent PD bounds for a zero-default portfolio, and the published values",
                "Upper confidence bounds for the PD of grades A, B and C (100, 400 and 300 obligors, no defaults) "
                "against the confidence level, under independence and under asset correlation 12%, computed by "
                "riskvalidation; the dots are the values printed by Pluto and Tasche.")
    out.append(f'  <text class="t" x="{ml}" y="28">Most prudent PD bound (%), grades A, B, C: 100, 400, 300 '
               'obligors, no defaults</text>')
    for yv in range(0, 11, 2):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{yv}</text>')
    for g in gammas:
        out.append(f'  <line class="grid" x1="{fx(g):.1f}" x2="{fx(g):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(g):.1f}" y="{h - mb + 18}" text-anchor="middle">{100 * g:g}%</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">confidence level '
               '(scale of -log10(1 - level))</text>')
    grid = 1.0 - np.logspace(math.log10(0.5), math.log10(0.001), 36)
    entries = []
    for rho, dash in ((0.0, ""), (0.12, " dash")):
        bounds = np.array([most_prudent_pd((100, 400, 300), (0, 0, 0), float(g), rho)["pd"] for g in grid]) * 100
        for j, (grade, cls) in enumerate((("A", 1), ("B", 2), ("C", 3))):
            pts = " ".join(f"{fx(g):.1f},{fy(b):.1f}" for g, b in zip(grid, bounds[:, j], strict=True))
            out.append(f'  <polyline class="l{cls}{dash}" points="{pts}"/>')
            for g, v in zip(gammas, published[rho][j], strict=True):
                out.append(f'  <circle class="p{cls}" cx="{fx(g):.1f}" cy="{fy(v):.1f}" r="3"/>')
            entries.append((f"l{cls}{dash}", f"grade {grade}, " + ("independent" if rho == 0 else "correlation 12%")))
    for i, (cls, text) in enumerate(entries):
        _legend(out, w - mr + 16, mt + 22 + 22 * i, cls, text)
    _legend(out, w - mr + 16, mt + 22 + 22 * 6 + 6, "p0", "printed by Pluto and Tasche", dot=True)
    out.append("</svg>")
    return "\n".join(out) + "\n"


SP_GRADES = ("AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
             "CCC-C")
SP_COUNTS = {  # Tasche (2013), Table 2: obligors rated and defaults, best grade first
    2009: ((81, 37, 188, 245, 340, 510, 546, 498, 541, 459, 266, 295, 441, 438, 482, 303, 190),
           (0, 0, 0, 0, 1, 2, 0, 2, 1, 5, 0, 3, 4, 24, 48, 52, 92)),
    2010: ((72, 25, 143, 209, 353, 474, 528, 457, 583, 430, 254, 276, 379, 393, 436, 290, 220),
           (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 3, 6, 49)),
}


def pd_curve_sp_svg() -> str:
    from riskvalidation.engines import pd_curve as pc

    m0 = pc.model_from_counts(*SP_COUNTS[2009])
    q0 = pc.qmm(m0["profile"], m0["pd"], m0["accuracy_ratio"], survival_profile=m0["survival_profile"])
    m1 = pc.model_from_counts(*SP_COUNTS[2010])
    pi1, p1 = m1["profile"], m1["pd"]
    forecasts = (
        ("l1", "invariant default profile (4.2)", pc.invariant_default_profile(m0["default_profile"], pi1, p1)),
        ("l2", "invariant accuracy ratio (4.3)", pc.invariant_accuracy_ratio(pi1, p1, m0["accuracy_ratio"])),
        ("l3", "scaled PDs (4.4)", pc.scaled_pds(q0["pd_curve"], pi1, p1)),
        ("l4", "scaled likelihood ratio (4.5)", pc.scaled_likelihood_ratio(q0["pd_curve"], q0["pd"], pi1, p1)),
    )
    w, h, ml, mr = 800, 760, 70, 230
    panels = ((60, 340), (430, 710))  # (top, bottom) of each plot area
    y_lo, y_hi = -4, 2  # decades of percent
    fx = lambda i: ml + (i + 0.5) / 17 * (w - ml - mr)  # noqa: E731

    def fy(v: float, top: int, bottom: int) -> float:
        lv = math.log10(max(v, 10.0**y_lo))
        return bottom - (lv - y_lo) / (y_hi - y_lo) * (bottom - top)

    out = _open(w, h, "PD curve calibration on the S&amp;P corporate data of Tasche (2013)",
                "Top: the 2009 grade default rates and the PD curve smoothed by quasi moment matching, which "
                "reproduces Table 5 of Tasche (2013). Bottom: the 2010 default rates and the four forecasts of section "
                "4.2 from the 2009 model at the observed 2010 default rate, which reproduce Table 7. Grades without a "
                "default are drawn as rings on the floor of the log scale.")
    titles = ("2009: grade default rates and the QMM curve (targets PD 3.99%, AR 82.7%)",
              "2010: default rates and the four case 1 forecasts from 2009 (p1 = 1.14%)")
    for (top, bottom), title in zip(panels, titles, strict=True):
        out.append(f'  <text class="t" x="{ml}" y="{top - 18}">{title}</text>')
        for lv in range(y_lo, y_hi + 1):
            yy = fy(10.0**lv, top, bottom)
            out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            out.append(f'  <text class="k" x="{ml - 8}" y="{yy + 4:.1f}" text-anchor="end">{10.0**lv:g}%</text>')
        out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{bottom}" y2="{bottom}"/>')
        out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{top}" y2="{bottom}"/>')
        for i, g in enumerate(SP_GRADES):
            out.append(f'  <text class="k" x="{fx(i):.1f}" y="{bottom + 16}" text-anchor="middle">{g}</text>')

    def observed(rates: list[float], top: int, bottom: int) -> None:
        for i, r in enumerate(rates):
            if r > 0:
                out.append(f'  <circle class="p0" cx="{fx(i):.1f}" cy="{fy(100 * r, top, bottom):.1f}" r="3.6"/>')
            else:
                out.append(f'  <circle class="hollow" cx="{fx(i):.1f}" cy="{bottom - 5:.1f}" r="3.6"/>')

    top, bottom = panels[0]
    pts = " ".join(f"{fx(i):.1f},{fy(100 * v, top, bottom):.1f}" for i, v in enumerate(q0["pd_curve"]))
    out.append(f'  <polyline class="l1" points="{pts}"/>')
    observed(m0["default_rate"], top, bottom)
    top, bottom = panels[1]
    for cls, _, res in forecasts:
        pts = " ".join(f"{fx(i):.1f},{fy(100 * v, top, bottom):.1f}" for i, v in enumerate(res["pd_curve"]))
        out.append(f'  <polyline class="{cls}" points="{pts}"/>')
    observed(m1["default_rate"], top, bottom)
    lx = w - mr + 14
    _legend(out, lx, panels[0][0] + 10, "l1", "QMM smoothed curve")
    for k, (cls, label, _) in enumerate(forecasts):
        _legend(out, lx, panels[1][0] + 10 + 22 * k, cls, label)
    for top_, offset in ((panels[0][0], 32), (panels[1][0], 10 + 22 * 4)):
        _legend(out, lx, top_ + offset, "p0", "observed default rate", dot=True)
        out.append(f'  <circle class="hollow" cx="{lx + 14}" cy="{top_ + offset + 22}" r="3.6"/>')
        out.append(f'  <text class="s" x="{lx + 34}" y="{top_ + offset + 26}">no default observed</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# U3: the size and power of the tests, measured by the harness on the suite's own scenarios


def _u3_imports():
    sys.path.insert(0, str(ROOT / "tests"))  # the figures draw what the suite measures, from the suite's scenarios
    import test_reproductions as reproductions
    import test_size as size

    return size, reproductions


def size_of_every_test_svg() -> str:
    from riskvalidation.generators import DefaultCounts, LogisticPortfolio, feed_grades
    from riskvalidation.harness import simulate, size_bound

    size, _ = _u3_imports()
    rows = []
    for tid, (gen, feed, kw) in sorted(size.NULL_SCENARIOS.items()):
        sim = simulate(tid, gen, n_rep=size.N_REP, seed=size.SEEDS[tid], feed=feed, test_kwargs=kw)
        rows.append((tid, sim.rate(0.05)))
    extra = [
        ("pd.hosmer_lemeshow, fitted=True (G - 2)", simulate("pd.hosmer_lemeshow", LogisticPortfolio(n=5000),
                                                              n_rep=size.N_REP, seed=6001,
                                                              test_kwargs={"fitted": True}).rate(0.05)),
        ("pd.traffic_lights, ties=upper", simulate("pd.traffic_lights", size.NULL_SCENARIOS["pd.traffic_lights"][0],
                                                   n_rep=size.N_REP, seed=6201, feed=size.feed_series(),
                                                   test_kwargs={"ties": "upper"}).rate(0.05)),
        ("pd.chi2_grades, a grade expecting 0.6 defaults",
         simulate("pd.chi2_grades", DefaultCounts(size.THIN_N, size.GRADES_PD), n_rep=size.N_REP, seed=6101,
                  feed=feed_grades()).rate(0.05)),
    ]
    rows += extra
    bound = size_bound(0.05, size.N_REP)
    w, ml, mr, mt, row_h = 900, 330, 90, 64, 22
    h = mt + row_h * len(rows) + 64
    x_max = 0.13
    fx = lambda v: ml + v / x_max * (w - ml - mr)  # noqa: E731
    out = _open(w, h, "Measured size of every test with a p-value",
                "Rejection rate at the 5% level of every registered test on a generator where its null holds at the "
                "boundary, 4,000 repetitions each, with 95% Wilson intervals; the nominal level and the bound "
                "consistent with a size of at most 5% are drawn as vertical lines.")
    out.append('  <text class="t" x="24" y="28">Rejection rate at the 5% level under the null, 4,000 repetitions '
               '(Wilson 95% intervals)</text>')
    for xv in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        out.append(f'  <line class="grid" x1="{fx(xv):.1f}" x2="{fx(xv):.1f}" y1="{mt - 8}" y2="{h - 52}"/>')
        out.append(f'  <text class="k" x="{fx(xv):.1f}" y="{h - 34}" text-anchor="middle">{100 * xv:.0f}%</text>')
    out.append(f'  <line class="ref" x1="{fx(0.05):.1f}" x2="{fx(0.05):.1f}" y1="{mt - 8}" y2="{h - 52}"/>')
    out.append(f'  <text class="k" x="{fx(0.05) - 4:.1f}" y="{mt - 14}" text-anchor="end">nominal 5%</text>')
    out.append(f'  <line class="l3 dash" x1="{fx(bound):.1f}" x2="{fx(bound):.1f}" y1="{mt - 8}" y2="{h - 52}"/>')
    out.append(f'  <text class="k" x="{fx(bound) + 4:.1f}" y="{mt - 14}">bound {100 * bound:.2f}%</text>')
    for i, (label, r) in enumerate(rows):
        y = mt + row_h * i + 6
        lo, hi = r.interval
        cls = "p3" if r.rate > bound else "p1"
        out.append(f'  <text class="s" x="{ml - 10}" y="{y + 4}" text-anchor="end">{label}</text>')
        out.append(f'  <line class="l0" x1="{fx(lo):.1f}" x2="{fx(min(hi, x_max)):.1f}" y1="{y}" y2="{y}" '
                   'stroke-width="1.4"/>')
        out.append(f'  <circle class="{cls}" cx="{fx(min(r.rate, x_max)):.1f}" cy="{y}" r="4"/>')
        out.append(f'  <text class="k" x="{w - 18}" y="{y + 4}" text-anchor="end">{100 * r.rate:.2f}%</text>')
    _legend(out, ml, h - 14, "p1", "within the bound", dot=True)
    _legend(out, ml + 200, h - 14, "p3", "above the bound", dot=True)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def binomial_true_size_svg() -> str:
    from riskvalidation.harness import rejection_probability

    w, h, ml, mr, mt, mb = 800, 420, 64, 250, 48, 56
    rhos = np.linspace(0.0, 0.20, 41)
    y_max = 0.30
    fx = lambda r: ml + r / 0.20 * (w - ml - mr)  # noqa: E731
    fy = lambda v: h - mb - min(v, y_max) / y_max * (h - mt - mb)  # noqa: E731
    curves = [
        ("pd.binomial", 0.05, {}, "l1", "", "binomial and Jeffreys, nominal 5%"),
        ("pd.binomial", 0.01, {}, "l1", " dash", "binomial, nominal 1%"),
        ("pd.jeffreys", 0.01, {}, "l4", " dash", "Jeffreys, nominal 1%"),
        ("pd.binomial_vasicek", 0.05, {"rho": 0.05}, "l2", "", "Vasicek, assumes 5%, nominal 5%"),
        ("pd.binomial_vasicek", 0.05, {"rho": 0.12}, "l3", "", "Vasicek, assumes 12%, nominal 5%"),
    ]
    out = _open(w, h, "True size of the count tests under asset correlation",
                "Exact rejection probability, when the PD applied is right, of the binomial, Jeffreys and "
                "Vasicek-corrected binomial tests for 1,000 obligors at PD 1%, against the true asset correlation.")
    out.append(f'  <text class="t" x="{ml}" y="28">True size when the PD is right: N = 1,000, PD 1% (exact)</text>')
    for yv in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{100 * yv:.0f}%</text>')
    for rv in (0.0, 0.05, 0.10, 0.15, 0.20):
        out.append(f'  <line class="grid" x1="{fx(rv):.1f}" x2="{fx(rv):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(rv):.1f}" y="{h - mb + 18}" text-anchor="middle">{100 * rv:.0f}%</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">true asset '
               'correlation</text>')
    for yv in (0.05, 0.01):
        out.append(f'  <line class="ref" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
    for i, (tid, a, kw, cls, dash, label) in enumerate(curves):
        ys = [rejection_probability(tid, 1000, 0.01, a, rho_true=float(r), **kw)["probability"] for r in rhos]
        pts = " ".join(f"{fx(r):.1f},{fy(v):.1f}" for r, v in zip(rhos, ys, strict=True))
        out.append(f'  <polyline class="{cls}{dash}" points="{pts}"/>')
        _legend(out, w - mr + 16, mt + 22 + 24 * i, f"{cls}{dash}", label)
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 22 + 24 * 5 + 8}">dashed grey: 5% and 1%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def traffic_lights_ties_svg() -> str:
    from riskvalidation.harness import traffic_lights_rejection_probability as tlr

    fx_ = json.loads((ROOT / "tests" / "fixtures" / "wp14_tables_5_to_8.json").read_text(encoding="utf-8"))
    levels = fx_["levels"]
    w, h, ml, mr, mt, mb = 800, 420, 64, 260, 48, 56
    fx = lambda i: ml + i / (len(levels) - 1) * (w - ml - mr)  # noqa: E731
    y_max = 0.15
    fy = lambda v: h - mb - min(v, y_max) / y_max * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "The traffic-lights test: printed tie rule and WP14's simulation",
                "Exact size of the traffic-lights test with independent years for WP14's scenarios I_SC (PD 0.3%) "
                "and I_LC (PD 3%) under the printed tie rule and the upper-colour rule, against WP14 Table 7.")
    out.append(f'  <text class="t" x="{ml}" y="28">Traffic lights, five independent years of 1,000 obligors: exact '
               'size</text>')
    for yv in (0.0, 0.05, 0.10, 0.15):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{100 * yv:.0f}%</text>')
    for i, a in enumerate(levels):
        out.append(f'  <line class="grid" x1="{fx(i):.1f}" x2="{fx(i):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(i):.1f}" y="{h - mb + 18}" text-anchor="middle">{_pct(a)}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">nominal level</text>')
    nominal = " ".join(f"{fx(i):.1f},{fy(a):.1f}" for i, a in enumerate(levels))
    out.append(f'  <polyline class="ref" points="{nominal}"/>')
    k = 0
    for name, cls in (("I_SC", "l1"), ("I_LC", "l2")):
        pf = np.array(fx_["scenarios_type_i"][name]["forecast_pct"]) / 100.0
        for ties, dash in (("lower", " dash"), ("upper", "")):
            ys = [tlr([1000] * 5, pf, a, ties=ties)["probability"] for a in levels]
            pts = " ".join(f"{fx(i):.1f},{fy(v):.1f}" for i, v in enumerate(ys))
            out.append(f'  <polyline class="{cls}{dash}" points="{pts}"/>')
            _legend(out, w - mr + 16, mt + 22 + 24 * k, f"{cls}{dash}", f"{name}, ties={ties}")
            k += 1
        dot = "p1" if cls == "l1" else "p2"
        for i, v in enumerate(fx_["table_7_type_i"][name]["traffic"]):
            out.append(f'  <circle class="{dot}" cx="{fx(i):.1f}" cy="{fy(v):.1f}" r="4"/>')
    _legend(out, w - mr + 16, mt + 22 + 24 * 4, "p0", "WP14 Table 7, printed", dot=True)
    _legend(out, w - mr + 16, mt + 22 + 24 * 5, "ref", "nominal level")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def published_reproductions_svg() -> str:
    from riskvalidation.generators import DefaultCounts, TwoSample, feed_series
    from riskvalidation.harness import metric_above, simulate

    _, rep = _u3_imports()
    fx_ = json.loads((ROOT / "tests" / "fixtures" / "wp14_tables_5_to_8.json").read_text(encoding="utf-8"))
    pts: dict[str, list[tuple[float, float, str, bool]]] = {"wp14_i": [], "wp14_ii": [], "yn": []}
    for table, scen, pub, seed0, key in (("7", fx_["scenarios_type_i"], fx_["table_7_type_i"], 1000, "wp14_i"),
                                         ("8", fx_["scenarios_type_ii"], fx_["table_8_type_ii"], 2000, "wp14_ii")):
        seed = seed0
        for name, sc in scen.items():
            gen = DefaultCounts.over_years([1000] * 5, np.array(sc["true_pct"]) / 100.0, rho=sc["rho"],
                                           pd_forecast=np.array(sc["forecast_pct"]) / 100.0, theta=sc["theta"])
            for short, (tid, kw) in rep.TESTS.items():
                seed += 1
                sim = simulate(tid, gen, n_rep=25_000, seed=seed, feed=feed_series(), alphas=fx_["levels"],
                               test_kwargs=kw)
                for level, p in zip(fx_["levels"], pub[name][short], strict=True):
                    r = sim.rate(level)
                    v = r.rate if table == "7" else 1.0 - r.rate
                    bad = (table, name, short, level) in rep.WP14_NOT_REPRODUCED
                    pts[key].append((p, v, short, bad))
    yn = json.loads((ROOT / "tests" / "fixtures" / "yurdakul_naranjo_table4.json").read_text(encoding="utf-8"))
    seed = 2000
    for row in yn["rows"]:
        for si, shift in enumerate((0.0, 0.25, 0.5)):
            seed += 1
            sim = simulate("stability.psi", TwoSample(n_base=row["n"], n_target=row["m"], shift=shift), n_rep=10_000,
                           seed=seed, alphas=(0.05,), rules=(metric_above(0.10), metric_above(0.25)))
            for label, p in zip(("metric>0.1", "metric>0.25", "p<0.05"), row["rates"][3 * si: 3 * si + 3],
                                strict=True):
                bad = (row["m"], row["n"], shift, label) in rep.YN_NOT_REPRODUCED
                pts["yn"].append((p, sim.rate(label).rate, label, bad))
    w, h = 1040, 410
    panels = [("wp14_i", "WP14 Table 7, type I errors", 0.15), ("wp14_ii", "WP14 Table 8, type II errors", 1.0),
              ("yn", "Yurdakul-Naranjo Table 4", 1.0)]
    out = _open(w, h, "Published simulation studies reproduced by the harness",
                "Each published rejection rate against the harness's rate on the same design: BCBS WP14 Tables 7 and 8 "
                "(normal and traffic-lights tests, 25,000 runs) and Yurdakul and Naranjo Table 4 (PSI, 10,000 runs). "
                "Points on the diagonal agree; hollow circles are the two cells that do not reproduce.")
    out.append('  <text class="t" x="24" y="28">Published rate (horizontal) against the harness (vertical), cell by '
               'cell</text>')
    pw, ph, top, gap, left = 270, 270, 60, 66, 64
    cls_of = {"normal": "p1", "traffic": "p2", "metric>0.1": "p3", "metric>0.25": "p4", "p<0.05": "p1"}
    for k, (key, title, vmax) in enumerate(panels):
        x0 = left + k * (pw + gap)
        fx = lambda v, x0=x0, vmax=vmax: x0 + min(v, vmax) / vmax * pw  # noqa: E731
        fy = lambda v, vmax=vmax: top + ph - min(v, vmax) / vmax * ph  # noqa: E731
        out.append(f'  <rect class="pn" x="{x0}" y="{top}" width="{pw}" height="{ph}"/>')
        out.append(f'  <text class="s" x="{x0}" y="{top - 8}">{title}</text>')
        out.append(f'  <line class="ref" x1="{fx(0):.1f}" y1="{fy(0):.1f}" x2="{fx(vmax):.1f}" y2="{fy(vmax):.1f}"/>')
        for tv in ((0.0, 0.05, 0.10, 0.15) if vmax < 0.5 else (0.0, 0.25, 0.5, 0.75, 1.0)):
            out.append(f'  <text class="k" x="{fx(tv):.1f}" y="{top + ph + 16}" text-anchor="middle">'
                       f'{100 * tv:.0f}%</text>')
            out.append(f'  <text class="k" x="{x0 - 6}" y="{fy(tv) + 4:.1f}" text-anchor="end">{100 * tv:.0f}%</text>')
        for p, v, lab, bad in pts[key]:
            if bad:
                out.append(f'  <circle class="hollow" cx="{fx(p):.1f}" cy="{fy(v):.1f}" r="6"/>')
            out.append(f'  <circle class="{cls_of[lab]}" cx="{fx(p):.1f}" cy="{fy(v):.1f}" r="2.6"/>')
    _legend(out, left, h - 26, "p1", "normal test; PSI chi-square benchmark", dot=True)
    _legend(out, left + 300, h - 26, "p2", "traffic lights (ties=upper)", dot=True)
    _legend(out, left + 520, h - 26, "p3", "PSI > 0.10", dot=True)
    _legend(out, left + 650, h - 26, "p4", "PSI > 0.25", dot=True)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def hhi_statistic_svg() -> str:
    w, h, ml, mr, mt, mb = 800, 420, 64, 250, 48, 56
    y_lo, y_hi = 0.0, 2.5
    fx = lambda v: ml + v / 5.0 * (w - ml - mr)  # noqa: E731
    fy = lambda v: h - mb - (max(min(v, y_hi), y_lo) - y_lo) / (y_hi - y_lo) * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "The ECB concentration statistic against the current coefficient of variation",
                "The statistic of the ECB Herfindahl index test as a function of the current CV, for 10 and 26 grades "
                "and several initial CVs; it rises, peaks and falls, and for an initial CV of 0.6 with ten grades it "
                "never reaches the 5% critical value 1.645.")
    out.append(f'  <text class="t" x="{ml}" y="28">ECB HHI test statistic S against the current CV</text>')
    for yv in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(yv) + 4:.1f}" text-anchor="end">{yv:g}</text>')
    for xv in range(0, 6):
        out.append(f'  <line class="grid" x1="{fx(xv):.1f}" x2="{fx(xv):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(xv):.1f}" y="{h - mb + 18}" text-anchor="middle">{xv}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">current coefficient '
               'of variation</text>')
    out.append(f'  <line class="l3 dash" x1="{ml}" x2="{w - mr}" y1="{fy(1.645):.1f}" y2="{fy(1.645):.1f}"/>')
    out.append(f'  <text class="k" x="{w - mr - 4}" y="{fy(1.645) - 6:.1f}" text-anchor="end">reject at 5% above '
               '1.645</text>')
    cases = [(10, 0.3, "l1", ""), (10, 0.6, "l1", " dash"), (10, 1.0, "l2", ""), (26, 0.6, "l4", "")]
    for i, (k, ci, cls, dash) in enumerate(cases):
        c = np.linspace(ci, np.sqrt(k - 1), 400)  # from no change to every obligor in one grade (CV sqrt(k - 1))
        s = np.sqrt(k - 1) * (c - ci) / (c * np.sqrt(0.5 + c * c))
        pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(c, s, strict=True))
        out.append(f'  <polyline class="{cls}{dash}" points="{pts}"/>')
        end = np.sqrt(k - 1)
        s_end = np.sqrt(k - 1) * (end - ci) / (end * np.sqrt(0.5 + end * end))
        out.append(f'  <circle class="p0" cx="{fx(end):.1f}" cy="{fy(s_end):.1f}" r="3.4"/>')
        _legend(out, w - mr + 16, mt + 22 + 24 * i, f"{cls}{dash}", f"K = {k}, initial CV {ci}")
    _legend(out, w - mr + 16, mt + 22 + 24 * 4, "p0", "all obligors in one grade", dot=True)
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 22 + 24 * 5 + 8}">each curve starts at its initial '
               'CV (S = 0)</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# U4: rating transitions, drawn by the package on its own generator (the size study's Markov chain, no agency data)


def _q6() -> np.ndarray:
    size, _ = _u3_imports()
    return np.asarray(size.Q6, dtype=float)


def transition_estimators_svg() -> str:
    from scipy.linalg import expm

    from riskvalidation.generators.paths import RatingPaths
    from riskvalidation.transitions.estimators import cohort, duration_generator

    q = _q6()
    truth = expm(q)[:5, -1]
    gen = RatingPaths(q, (1000,) * 5, horizon=1.0, snapshots=(0.0, 1.0))
    rng = np.random.default_rng(2026)
    n_years = 1000
    coh = np.empty((n_years, 5))
    dur = np.empty((n_years, 5))
    for y in range(n_years):
        d = gen.draw(rng)
        coh[y] = cohort(d["steps"][0]["counts"], d["steps"][0]["withdrawn"])["matrix"][:5, -1]
        dur[y] = expm(duration_generator(d["segments"], 6)["generator"])[:5, -1]
    w, h = 1000, 470
    out = _open(w, h, "Cohort against duration PDs by grade on a known generator",
                "One-year PD by grade estimated from 1,000 simulated years of 1,000 obligors per grade: the median "
                "and the 5% to 95% range of the cohort and duration estimates against the true PD (log scale), and "
                "the share of years in which each estimate is exactly zero.")
    out.append('  <text class="t" x="24" y="28">One-year PD by grade, 1,000 obligors per grade, 1,000 simulated years '
               '(generator known)</text>')
    # left: PD by grade on a log scale
    x0, top, pw, ph = 84, 64, 520, 300
    lo_e, hi_e = -6.0, 0.0  # 1e-6 to 1
    zero_y = top + ph + 22  # an estimate of exactly zero has no place on a log scale: its own row below
    fx = lambda g: x0 + (g + 0.5) / 5 * pw  # noqa: E731
    fy = lambda v: zero_y if v <= 0 else top + ph - (np.log10(max(v, 10**lo_e)) - lo_e) / (hi_e - lo_e) * ph  # noqa: E731
    out.append(f'  <rect class="pn" x="{x0}" y="{top}" width="{pw}" height="{ph}"/>')
    out.append(f'  <line class="ref" x1="{x0}" x2="{x0 + pw}" y1="{zero_y}" y2="{zero_y}"/>')
    out.append(f'  <text class="k" x="{x0 - 6}" y="{zero_y + 4}" text-anchor="end">0</text>')
    for e in range(int(lo_e), int(hi_e) + 1):
        y = fy(10.0**e)
        out.append(f'  <line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        label = {-6: "0.01bp", -5: "0.1bp", -4: "1bp", -3: "10bp", -2: "1%", -1: "10%", 0: "100%"}[e]
        out.append(f'  <text class="k" x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for g in range(5):
        out.append(f'  <text class="k" x="{fx(g):.1f}" y="{zero_y + 22}" text-anchor="middle">grade {g + 1}</text>')
    pts = " ".join(f"{fx(g):.1f},{fy(truth[g]):.1f}" for g in range(5))
    out.append(f'  <polyline class="l0" points="{pts}"/>')
    for off, est, cls in ((-12, coh, "1"), (12, dur, "2")):
        for g in range(5):
            lo, med, hi = np.quantile(est[:, g], [0.05, 0.5, 0.95])
            x = fx(g) + off
            out.append(f'  <line class="l{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{fy(hi):.1f}" y2="{fy(lo):.1f}" '
                       'stroke-width="3"/>')
            out.append(f'  <circle class="p{cls}" cx="{x:.1f}" cy="{fy(med):.1f}" r="4.5"/>')
    _legend(out, x0, h - 30, "l0", "true PD")
    _legend(out, x0 + 130, h - 30, "p1", "cohort: median, 5% to 95%", dot=True)
    _legend(out, x0 + 360, h - 30, "p2", "duration: median, 5% to 95%", dot=True)
    # right: share of years with a zero estimate
    x1, pw2 = 680, 280
    fy2 = lambda v: top + ph - v * ph  # noqa: E731
    out.append(f'  <rect class="pn" x="{x1}" y="{top}" width="{pw2}" height="{ph}"/>')
    out.append(f'  <text class="s" x="{x1}" y="{top - 10}">share of years with a zero PD estimate</text>')
    out.append(f'  <text class="k" x="{x1 + pw2 - 8}" y="{top + 18}" text-anchor="end">the duration estimate is never '
               'zero</text>')
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'  <line class="grid" x1="{x1}" x2="{x1 + pw2}" y1="{fy2(v):.1f}" y2="{fy2(v):.1f}"/>')
        out.append(f'  <text class="k" x="{x1 - 6}" y="{fy2(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    bw = pw2 / 5
    for g in range(5):
        for k, (est, cls) in enumerate(((coh, "p1"), (dur, "p2"))):
            share = float(np.mean(est[:, g] == 0.0))
            x = x1 + g * bw + 10 + k * (bw - 20) / 2
            out.append(f'  <rect class="{cls}" x="{x:.1f}" y="{fy2(share):.1f}" width="{(bw - 20) / 2 - 2:.1f}" '
                       f'height="{top + ph - fy2(share):.1f}"/>')
            if share > 0:
                out.append(f'  <text class="k" x="{x + (bw - 20) / 4:.1f}" y="{fy2(share) - 4:.1f}" '
                           f'text-anchor="middle">{100 * share:.0f}%</text>')
        out.append(f'  <text class="k" x="{x1 + (g + 0.5) * bw:.1f}" y="{top + ph + 18}" text-anchor="middle">'
                   f'grade {g + 1}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


MARKOV_FORMS = [
    ("rating.time_homogeneity", {}, "time homogeneity, chi-square (3.8), default"),
    ("rating.time_homogeneity", {"form": "lr"}, "time homogeneity, likelihood ratio (3.5)"),
    ("rating.markov_order", {}, "order, grouped by direction, chi-square, default"),
    ("rating.markov_order", {"form": "lr"}, "order, grouped by direction, likelihood ratio"),
    ("rating.markov_order", {"grouping": "none"}, "order, full table, chi-square (3.14)"),
    ("rating.markov_order", {"grouping": "none", "form": "lr"}, "order, full table, likelihood ratio (3.12)"),
    ("rating.matrix_reference", {}, "reference matrix, likelihood ratio, default"),
    ("rating.matrix_reference", {"form": "chi2"}, "reference matrix, chi-square (3.1) as printed"),
    ("rating.momentum", {}, "momentum, downward, default"),
    ("rating.momentum", {"direction": "up"}, "momentum, upward"),
]


def markov_test_forms_svg() -> str:
    from riskvalidation.harness import simulate, size_bound

    size, _ = _u3_imports()
    rows = []
    for tid, kw, label in MARKOV_FORMS:
        gen, feed, base = size.NULL_SCENARIOS[tid]
        sim = simulate(tid, gen, n_rep=size.N_REP, seed=size.SEEDS[tid], feed=feed, test_kwargs={**base, **kw})
        rows.append((label, sim.rate(0.05), sim.rate(0.01)))
    b5, b1 = size_bound(0.05, size.N_REP), size_bound(0.01, size.N_REP)
    w, ml, mr, mt, row_h = 1000, 360, 150, 70, 26
    h = mt + row_h * len(rows) + 70
    x_max = 0.15
    fx = lambda v: ml + min(v, x_max) / x_max * (w - ml - mr)  # noqa: E731
    out = _open(w, h, "Size of every form of the four transition tests",
                "Rejection rate at the 5% level of each form of the time-homogeneity, order, reference-matrix and "
                "momentum tests on a Markov chain (the null), 4,000 repetitions with 95% Wilson intervals; the "
                "default forms hold the bound, three printed forms do not.")
    out.append('  <text class="t" x="24" y="28">Rejection rate under the null at the 5% level (dot) and the 1% level '
               '(ring), 4,000 repetitions</text>')
    for xv in (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15):
        out.append(f'  <line class="grid" x1="{fx(xv):.1f}" x2="{fx(xv):.1f}" y1="{mt - 10}" y2="{h - 58}"/>')
        out.append(f'  <text class="k" x="{fx(xv):.1f}" y="{h - 40}" text-anchor="middle">{100 * xv:g}%</text>')
    for v, lab, cls in ((0.05, "5%", "ref"), (0.01, "1%", "ref")):
        out.append(f'  <line class="{cls}" x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{mt - 10}" y2="{h - 58}"/>')
        out.append(f'  <text class="k" x="{fx(v):.1f}" y="{mt - 16}" text-anchor="middle">{lab}</text>')
    for v in (b5, b1):
        out.append(f'  <line class="l3 dash" x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{mt - 10}" y2="{h - 58}" '
                   'stroke-width="1.4"/>')
    out.append(f'  <text class="k" x="{w - 18}" y="{mt - 16}" text-anchor="end">at 5% / at 1%</text>')
    for i, (label, r5, r1) in enumerate(rows):
        y = mt + row_h * i + 8
        bad = r5.rate > b5 or r1.rate > b1
        lo, hi = r5.interval
        out.append(f'  <text class="s" x="{ml - 10}" y="{y + 4}" text-anchor="end">{label}</text>')
        out.append(f'  <line class="l0" x1="{fx(lo):.1f}" x2="{fx(hi):.1f}" y1="{y}" y2="{y}" stroke-width="1.4"/>')
        out.append(f'  <circle class="{"p3" if bad else "p1"}" cx="{fx(r5.rate):.1f}" cy="{y}" r="4.5"/>')
        out.append(f'  <circle class="hollow" cx="{fx(r1.rate):.1f}" cy="{y}" r="4.5"/>')
        out.append(f'  <text class="k" x="{w - 18}" y="{y + 4}" text-anchor="end">{100 * r5.rate:.2f}% / '
                   f'{100 * r1.rate:.2f}%</text>')
    _legend(out, ml, h - 16, "p1", "within the bound", dot=True)
    _legend(out, ml + 170, h - 16, "p3", "above the bound", dot=True)
    out.append(f'  <line class="l3 dash" x1="{ml + 340}" x2="{ml + 368}" y1="{h - 16}" y2="{h - 16}"/>')
    out.append(f'  <text class="s" x="{ml + 374}" y="{h - 12}">bounds {100 * b5:.2f}% and {100 * b1:.2f}%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def interval_coverage_svg() -> str:
    from riskvalidation.transitions.intervals import pd_agresti_coull, pd_jeffreys, pd_wald

    n = 200
    ps = np.linspace(0.001, 0.06, 237)
    d = np.arange(n + 1)
    nn = np.full(n + 1, n)
    curves = []
    for f, cls, label in ((pd_wald, "l3", "Wald (2.2)"), (pd_agresti_coull, "l1", "Agresti-Coull (3.3)"),
                          (pd_jeffreys, "l2", "Jeffreys")):
        r = f(d, nn)
        cov = [float(np.sum(stats.binom.pmf(d, n, p) * ((r["lower"] <= p) & (p <= r["upper"])))) for p in ps]
        curves.append((cls, label, cov))
    w, h, ml, mr, mt, mb = 900, 420, 70, 220, 52, 56
    y_lo, y_hi = 0.0, 1.0
    fx = lambda p: ml + (p - ps[0]) / (ps[-1] - ps[0]) * (w - ml - mr)  # noqa: E731
    fy = lambda v: h - mb - (max(v, y_lo) - y_lo) / (y_hi - y_lo) * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "Exact coverage of three 95% intervals for a PD",
                "Coverage probability, computed exactly by enumerating the binomial, of the Wald, Agresti-Coull and "
                "Jeffreys 95% intervals for a grade of 200 obligors, against the true PD from 0.1% to 6%.")
    out.append(f'  <text class="t" x="{ml}" y="30">Exact coverage of a nominal 95% interval, N = 200 obligors</text>')
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    for p in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        out.append(f'  <line class="grid" x1="{fx(p):.1f}" x2="{fx(p):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(p):.1f}" y="{h - mb + 18}" text-anchor="middle">{100 * p:.0f}%</text>')
    out.append(f'  <line class="ref" x1="{ml}" x2="{w - mr}" y1="{fy(0.95):.1f}" y2="{fy(0.95):.1f}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">true PD (expected '
               'defaults = 200 x PD)</text>')
    for i, (cls, label, cov) in enumerate(curves):
        pts = " ".join(f"{fx(p):.1f},{fy(c):.1f}" for p, c in zip(ps, cov, strict=True))
        out.append(f'  <polyline class="{cls}" points="{pts}" stroke-width="1.6"/>')
        _legend(out, w - mr + 16, mt + 20 + 26 * i, cls, label)
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 20 + 26 * 3 + 6}">dashed grey: 95%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def ttc_drift_svg() -> str:
    from scipy.linalg import expm

    from riskvalidation.transitions.ttc import project

    p = expm(_q6())
    o = [0.0, 0.3, 0.4, 0.3, 0.0]
    years = 40
    starts = [([1.0, 0, 0, 0, 0], "l1", "all in grade 1"), ([0, 0, 0, 0, 1.0], "l3", "all in grade 5"),
              ([0, 0.3, 0.4, 0.3, 0], "l2", "the origination mix"), ([0.2] * 5, "l4 dash", "uniform")]
    paths = [(project(p, s, o, years), cls, label) for s, cls, label in starts]
    ttc_pd = paths[0][0]["ttc"]["default_rate"]
    w, h, ml, mr, mt, mb = 900, 420, 70, 230, 52, 56
    y_max = 0.20
    fx = lambda t: ml + t / (years - 1) * (w - ml - mr)  # noqa: E731
    fy = lambda v: h - mb - min(v, y_max) / y_max * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "Projected portfolio default rate under an unstressed matrix",
                "Default rate of four starting portfolios projected under the same one-year matrix with defaults "
                "re-originated (Engelmann's propagation): every path converges to the TTC portfolio's default rate, "
                "and none of the movement comes from a scenario.")
    out.append(f'  <text class="t" x="{ml}" y="30">Portfolio default rate by year, one unstressed matrix, defaults '
               're-originated</text>')
    for v in (0.0, 0.04, 0.08, 0.12, 0.16, 0.20):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    for year in (1, 5, 10, 15, 20, 25, 30, 35, 40):
        out.append(f'  <text class="k" x="{fx(year - 1):.1f}" y="{h - mb + 18}" text-anchor="middle">{year}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">projection '
               'year</text>')
    out.append(f'  <line class="ref" x1="{ml}" x2="{w - mr}" y1="{fy(ttc_pd):.1f}" y2="{fy(ttc_pd):.1f}"/>')
    for i, (pr, cls, label) in enumerate(paths):
        pts = " ".join(f"{fx(t):.1f},{fy(v):.1f}" for t, v in enumerate(pr["default_rate"]))
        out.append(f'  <polyline class="{cls}" points="{pts}"/>')
        _legend(out, w - mr + 16, mt + 20 + 26 * i, cls, label)
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 20 + 26 * 4 + 6}">dashed grey: the TTC portfolio, '
               f'{100 * ttc_pd:.2f}%</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def rating_momentum_svg() -> str:
    q = _q6()
    exit_rate = -np.diag(q)
    alpha, beta = 1.0, 1.0
    # an illustrative history: grade 2 from 0, downgraded to 3 at 1.0 and to 4 at 1.8, upgraded to 3 at 3.4
    moves = [(0.0, 1), (1.0, 2), (1.8, 3), (3.4, 2)]
    downs = [1.0, 1.8]
    ts = np.linspace(0.0, 5.0, 1001)
    state = np.array([[s for t0, s in moves if t0 <= t][-1] for t in ts])
    base = exit_rate[state]
    exc = np.zeros_like(ts)
    for tau in downs:
        exc += np.where(ts >= tau, beta * alpha * np.exp(-beta * (ts - tau)), 0.0)
    w, h, ml, mr, mt, mb = 900, 400, 70, 250, 52, 56
    y_max = 2.6
    fx = lambda t: ml + t / 5.0 * (w - ml - mr)  # noqa: E731
    fy = lambda v: h - mb - min(v, y_max) / y_max * (h - mt - mb)  # noqa: E731
    out = _open(w, h, "A self-exciting downgrade intensity",
                "The intensity of an obligor's next transition in dos Reis, Pfeuffer and Smith's momentum model: the "
                "Markov baseline of its current grade plus a term that jumps by beta times alpha at each downgrade and "
                "decays at rate beta (alpha = beta = 1 here).")
    out.append(f'  <text class="t" x="{ml}" y="30">Intensity of the next transition, baseline plus momentum '
               '(alpha = beta = 1)</text>')
    for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        out.append(f'  <line class="grid" x1="{ml}" x2="{w - mr}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/>')
        out.append(f'  <text class="k" x="{ml - 8}" y="{fy(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    for t in range(6):
        out.append(f'  <text class="k" x="{fx(t):.1f}" y="{h - mb + 18}" text-anchor="middle">{t}</text>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    out.append(f'  <line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}"/>')
    out.append(f'  <text class="s" x="{(ml + w - mr) / 2:.0f}" y="{h - 14}" text-anchor="middle">years</text>')
    pts_b = " ".join(f"{fx(t):.1f},{fy(v):.1f}" for t, v in zip(ts, base, strict=True))
    pts_t = " ".join(f"{fx(t):.1f},{fy(v):.1f}" for t, v in zip(ts, base + exc, strict=True))
    out.append(f'  <polyline class="l2 dash" points="{pts_b}"/>')
    out.append(f'  <polyline class="l1" points="{pts_t}"/>')
    for t0, s in moves[1:]:
        kind = "down" if t0 in downs else "up"
        out.append(f'  <line class="ref" x1="{fx(t0):.1f}" x2="{fx(t0):.1f}" y1="{mt}" y2="{h - mb}"/>')
        out.append(f'  <text class="k" x="{fx(t0) + 4:.1f}" y="{mt + 12}">{kind} to {s + 1}</text>')
    _legend(out, w - mr + 16, mt + 20, "l1", "total intensity")
    _legend(out, w - mr + 16, mt + 46, "l2 dash", "Markov baseline q_i")
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 80}">each downgrade adds</text>')
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 96}">beta x alpha, decaying</text>')
    out.append(f'  <text class="k" x="{w - mr + 16}" y="{mt + 112}">as exp(-beta (t - tau))</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _downgrades_times(q: np.ndarray, k: float) -> np.ndarray:
    """The generator with every downgrade rate multiplied by k (a recession regime)."""
    r = q.copy()
    r[np.triu_indices(r.shape[0], k=1)] *= k
    r[-1] = 0.0
    np.fill_diagonal(r, 0.0)
    np.fill_diagonal(r, -r.sum(axis=1))
    return r


MARKOV_POWER_TESTS = [
    ("rating.time_homogeneity", lambda d: {"counts_by_period": [s["counts"] for s in d["steps"]]}, "time homogeneity",
     "p1"),
    ("rating.markov_order", lambda d: {"triplets": d["triplets"]}, "order (grouped)", "p2"),
    ("rating.momentum", lambda d: {"paths": d["segments"]}, "momentum hazard", "p3"),
]


def markov_test_power_rows() -> list[tuple[str, list[float]]]:
    """Rejection rates at 5% of the three path tests against two planted defects, 400 repetitions each."""
    from riskvalidation.generators.paths import RatingPaths
    from riskvalidation.harness import simulate

    q = _q6()
    snaps = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    designs = [("no defect (the null)", RatingPaths(q, (1000,) * 5, horizon=5.0, snapshots=snaps))]
    for a in (0.25, 0.5, 1.0):
        designs.append((f"momentum, alpha = {a:g}", RatingPaths(q, (1000,) * 5, horizon=5.0, snapshots=snaps,
                                                                 momentum=((a, 1.0), (a, 1.0)), investment_grades=3)))
    for k in (1.25, 1.5, 2.0):
        designs.append((f"downgrades x {k:g} from year 2", RatingPaths(q, (1000,) * 5, horizon=5.0, snapshots=snaps,
                                                                        regimes=((2.0, _downgrades_times(q, k)),))))
    rows = []
    seed = 7400
    for label, gen in designs:
        rates = []
        for tid, feed, _, _ in MARKOV_POWER_TESTS:
            seed += 1
            rates.append(simulate(tid, gen, n_rep=400, seed=seed, feed=feed).rate(0.05).rate)
        rows.append((label, rates))
    return rows


def markov_test_power_svg() -> str:
    rows = markov_test_power_rows()
    w, ml, mr, mt, row_h = 1000, 250, 80, 74, 52
    h = mt + row_h * len(rows) + 60
    fx = lambda v: ml + v * (w - ml - mr)  # noqa: E731
    out = _open(w, h, "Which transition test sees which defect",
                "Rejection rate at the 5% level of the time-homogeneity, order and momentum tests on the rating-path "
                "generator with no defect, with downward momentum of three strengths, and with a recession regime "
                "that multiplies every downgrade rate from the third year, 400 repetitions each.")
    out.append('  <text class="t" x="24" y="28">Rejection rate at the 5% level, 1,000 obligors per grade, five annual '
               'snapshots, 400 repetitions</text>')
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'  <line class="grid" x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{mt - 10}" y2="{h - 50}"/>')
        out.append(f'  <text class="k" x="{fx(v):.1f}" y="{h - 34}" text-anchor="middle">{100 * v:.0f}%</text>')
    out.append(f'  <line class="ref" x1="{fx(0.05):.1f}" x2="{fx(0.05):.1f}" y1="{mt - 10}" y2="{h - 50}"/>')
    out.append(f'  <text class="k" x="{fx(0.05):.1f}" y="{mt - 16}" text-anchor="middle">5%</text>')
    bar = 12
    for i, (label, rates) in enumerate(rows):
        y0 = mt + row_h * i
        out.append(f'  <text class="s" x="{ml - 10}" y="{y0 + 18}" text-anchor="end">{label}</text>')
        for k, ((_, _, _, cls), r) in enumerate(zip(MARKOV_POWER_TESTS, rates, strict=True)):
            y = y0 + 2 + k * (bar + 3)
            out.append(f'  <rect class="{cls}" x="{ml}" y="{y}" width="{max(fx(r) - ml, 1.0):.1f}" height="{bar}"/>')
            out.append(f'  <text class="k" x="{fx(r) + 6:.1f}" y="{y + bar - 1}">{100 * r:.1f}%</text>')
    for k, (_, _, name, cls) in enumerate(MARKOV_POWER_TESTS):
        _legend(out, ml + 220 * k, h - 12, cls, name, dot=True)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    u2 = {"irb-risk-weights.svg": irb_risk_weights_svg, "lhp-distribution.svg": lhp_distribution_svg,
          "most-prudent-bounds.svg": most_prudent_svg, "pd-curve-sp.svg": pd_curve_sp_svg}
    if not only:
        (ASSETS / "psi-benchmark.svg").write_text(psi_benchmark_svg(), encoding="utf-8", newline="\n")
        (ASSETS / "roc-cap.svg").write_text(roc_cap_svg(), encoding="utf-8", newline="\n")
        for name, make in u2.items():
            (ASSETS / name).write_text(make(), encoding="utf-8", newline="\n")
    u3 = {"binomial-true-size.svg": binomial_true_size_svg, "traffic-lights-ties.svg": traffic_lights_ties_svg,
          "hhi-statistic.svg": hhi_statistic_svg, "size-of-every-test.svg": size_of_every_test_svg,
          "published-reproductions.svg": published_reproductions_svg}
    u4 = {"transition-estimators.svg": transition_estimators_svg, "markov-test-forms.svg": markov_test_forms_svg,
          "markov-test-power.svg": markov_test_power_svg, "interval-coverage.svg": interval_coverage_svg,
          "ttc-drift.svg": ttc_drift_svg, "rating-momentum.svg": rating_momentum_svg}
    unknown = only - set(u3) - set(u4)
    if unknown:
        raise SystemExit(f"unknown figures: {sorted(unknown)}")
    drawn = []
    for name, make in {**u3, **u4}.items():
        if only and name not in only:
            continue
        print("drawing", name, flush=True)
        (ASSETS / name).write_text(make(), encoding="utf-8", newline="\n")
        drawn.append(name)
    print("wrote docs/assets/" + ", ".join(([] if only else ["psi-benchmark.svg", "roc-cap.svg", *u2]) + drawn))


if __name__ == "__main__":
    main()
