"""Credit risk mathematics with riskvalidation (unit U2): IRB capital under two regimes, the large homogeneous portfolio
behind it, most prudent PDs for a low-default portfolio, and a PD curve calibrated to a new year and tested for fit."""

import json
import os
import sys
from pathlib import Path

from riskvalidation.engines import low_default, pd_curve, vasicek
from riskvalidation.regulatory import capital_requirement
from riskvalidation.validation import pd_default_profile

# IRB capital of a corporate loan, Basel III final against Basel II (the 1.06 factor; the floors differ below 0.05%)
for regime in ("basel3", "basel2"):
    r = capital_requirement("corporate", 0.01, 0.45, maturity=2.5, ead=1_000_000, regime=regime)
    print(f"{regime}: K = {r.k:.4f}, risk weight {r.risk_weight:.2%}, RWA {r.rwa:,.0f} ({r.references['risk_weight']})")

# a retail function is LGD times the 99.9% quantile of the large homogeneous portfolio, less the expected loss
mortgage = capital_requirement("residential_mortgage", 0.02, 0.25)
print(f"mortgage K / LGD + PD = {mortgage.k / 0.25 + 0.02:.6f}; "
      f"LHP 99.9% quantile at R = 15%: {vasicek.lhp_quantile(0.999, 0.02, 0.15):.6f}")

# most prudent PDs of three grades without defaults (Pluto and Tasche), independent and with asset correlation 12%
for rho in (0.0, 0.12):
    res = low_default.most_prudent_pd((100, 400, 300), (0, 0, 0), gamma=0.9, rho=rho)
    print(f"most prudent PDs at 90%, correlation {rho:.0%}:", ", ".join(f"{p:.2%}" for p in res["pd"]))

# a PD curve smoothed on one year and calibrated to the next (Tasche 2013, S&P corporates, best grade first)
# The counts are third-party data and are not in the repository: they are read from
# <RISKVALIDATION_DATA>/tasche2013/table2.json ({"years": {"2009": {"obligors": [...], "defaults": [...]}, ...}}).
_data = os.environ.get("RISKVALIDATION_DATA")
_table2 = Path(_data) / "tasche2013" / "table2.json" if _data else None
if _table2 is None or not _table2.is_file():
    print("PD curve part skipped: the S&P Table 2 counts are read from RISKVALIDATION_DATA/tasche2013/table2.json")
    sys.exit(0)
_years = json.loads(_table2.read_text(encoding="utf-8"))["years"]
obligors_2009, defaults_2009 = _years["2009"]["obligors"], _years["2009"]["defaults"]
obligors_2010, defaults_2010 = _years["2010"]["obligors"], _years["2010"]["defaults"]

m0 = pd_curve.model_from_counts(obligors_2009, defaults_2009)
q0 = pd_curve.qmm(m0["profile"], m0["pd"], m0["accuracy_ratio"], survival_profile=m0["survival_profile"])
print(f"2009: PD {m0['pd']:.2%}, AR {m0['accuracy_ratio']:.1%}; smoothed PD of AAA {q0['pd_curve'][0]:.4%}, "
      f"of CCC-C {q0['pd_curve'][-1]:.2%}")
m1 = pd_curve.model_from_counts(obligors_2010, defaults_2010)
for name, fc in (
    ("scaled PDs", pd_curve.scaled_pds(q0["pd_curve"], m1["profile"], m1["pd"])),
    ("scaled likelihood ratio", pd_curve.scaled_likelihood_ratio(q0["pd_curve"], q0["pd"], m1["profile"], m1["pd"])),
):
    test = pd_default_profile(obligors_2010, defaults_2010, fc["pd_curve"], n_sim=20_000, seed=2013)
    scale = f"c_PD = {fc['c_pd']:.4f}" if "c_pd" in fc else f"c_LR = {fc['c_lr']:.4f}"
    print(f"2010, {name}: {scale}; default-profile test p = {test.p_value:.3f} ({test.light.value})")
