#!/usr/bin/env python3
"""Regenerate the computed documentation figures (deterministic; NumPy and SciPy only).

- docs/assets/psi-benchmark.svg: the Yurdakul-Naranjo critical PSI against sample size, with the 0.10 and 0.25
  conventional bands.
- docs/assets/roc-cap.svg: ROC and CAP curves of a binormal score with the areas behind AR = 2 AUC - 1.
- docs/assets/irb-risk-weights.svg, lhp-distribution.svg, most-prudent-bounds.svg, pd-curve-sp.svg: the U2
  engines drawn by the package itself against their published values (CRE99 Table 1, Pluto and Tasche, Tasche
  2013); theme-aware.

Usage: python scripts/make_doc_figures.py
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


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "psi-benchmark.svg").write_text(psi_benchmark_svg(), encoding="utf-8", newline="\n")
    (ASSETS / "roc-cap.svg").write_text(roc_cap_svg(), encoding="utf-8", newline="\n")
    u2 = {"irb-risk-weights.svg": irb_risk_weights_svg, "lhp-distribution.svg": lhp_distribution_svg,
          "most-prudent-bounds.svg": most_prudent_svg, "pd-curve-sp.svg": pd_curve_sp_svg}
    for name, make in u2.items():
        (ASSETS / name).write_text(make(), encoding="utf-8", newline="\n")
    print("wrote docs/assets/" + ", ".join(["psi-benchmark.svg", "roc-cap.svg", *u2]))


if __name__ == "__main__":
    main()
