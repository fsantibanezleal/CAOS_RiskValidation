#!/usr/bin/env python3
"""Regenerate the computed documentation figures (deterministic; NumPy and SciPy only).

- docs/assets/psi-benchmark.svg: the Yurdakul-Naranjo critical PSI against sample size, with the 0.10 and 0.25
  conventional bands.
- docs/assets/roc-cap.svg: ROC and CAP curves of a binormal score with the areas behind AR = 2 AUC - 1.

Usage: python scripts/make_doc_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

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


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "psi-benchmark.svg").write_text(psi_benchmark_svg(), encoding="utf-8", newline="\n")
    (ASSETS / "roc-cap.svg").write_text(roc_cap_svg(), encoding="utf-8", newline="\n")
    print("wrote docs/assets/psi-benchmark.svg and docs/assets/roc-cap.svg")


if __name__ == "__main__":
    main()
