# U19 FRTB and CVA calculators: SA-CVA, the default risk charge for non-securitisations and the residual risk add-on (BL-076)

Sources: the Basel Framework on bis.org in the versions in force from 2023-01-01, transcribed verbatim in the research report of the
Contraste initiative (CAOS_MANAGE `wip/contraste/m14-frtb-cva-research-2026-11-15.md`: D1 MAR50.27 to MAR50.53, D2 MAR50.54 to
MAR50.77 with every risk weight, bucket and correlation table, D3 MAR22.1 to MAR22.26, D4 MAR23.1 to MAR23.8, D5 the interfaces and
the DERIVED fixtures), and the decisions of 2026-11-15 (CAOS_MANAGE `plans/contraste/post-1-0-cycle.md`, "Engine 1.3.0"). MAR50:
https://www.bis.org/basel_framework/chapter/MAR/50.htm?inforce=20230101&published=20200708 ; MAR22:
https://www.bis.org/basel_framework/chapter/MAR/22.htm?inforce=20230101&published=20200327 ; MAR23:
https://www.bis.org/basel_framework/chapter/MAR/23.htm?inforce=20230101&published=20200327 . Transcribed into
`docs/regulatory/` pages of their own.

EARS (Mavin et al., RE'09, doi:10.1109/RE.2009.9). Every requirement names the gate that fails when it is violated. The engine
computes and states no compliance claim. The Basel text holds no numeric example of the SA-CVA and none of the non-securitisation
bucket formula: the fixtures marked DERIVED in the research report are computed from the transcription, and the tests recompute them
by a second route (an independent loop over the formula, or a closed form), never by typing the answer. Module conventions follow
`regulatory/ba_cva.py` and `regulatory/frtb.py`: constants as `RegulatoryParameter(value, paragraph, table, in_force)`, frozen
dataclasses with `paragraphs`, read-only numpy arrays, the validators of `regulatory/standardised.py`.

## SA-CVA (`regulatory/sa_cva.py`, MAR50.27 to MAR50.77)

| ID | Requirement | Gate |
|---|---|---|
| RV-2001 | THE module SHALL hold R = 0.01 (MAR50.53), m_CVA = 1 (MAR50.41) and the RWA factor 12.5 (MAR50.1) as regulatory parameters naming their paragraphs, and every risk weight, bucket, tenor and correlation table of MAR50.54 to MAR50.77 as read-only parameters pinned cell by cell against the transcription (Tables 3 to 14, the symmetry of Tables 4, 6 and 9, the bucket counts 8, 17, 13 and 11 of Tables 5, 8, 11 and 13). | `tests/regulatory/test_sa_cva.py::test_parameters_pinned_to_mar50` |
| RV-2002 | THE `weighted_sensitivity` SHALL give WS^CVA = RW x s^CVA, WS^Hdg = RW x s^Hdg and WS = WS^CVA - WS^Hdg (MAR50.51, 50.52); THE `bucket_k` SHALL be the square root of max(0, sum WS_k^2 + sum over k != l of rho_kl WS_k WS_l + R sum (WS_k^Hdg)^2) with the hedging disallowance inside the root (MAR50.53 (1)); a bucket with a single risk factor SHALL take no correlation term (MAR50 states no within-bucket correlation for it: the reading taken, stated in the docstring); THE `bucket_s` SHALL be max(-K_b, min(sum WS_k, K_b)) always (MAR50.53 (3)). DERIVED fixtures SA-A to SA-D (100, 10, 146.9694 and 118.5285, 30,904.485 from EUR sensitivities). | `tests/regulatory/test_sa_cva.py::test_bucket_formula_derived_fixtures` |
| RV-2003 | THE `risk_class_k` SHALL be m_CVA times the square root of max(0, sum K_b^2 + sum over b != c of gamma_bc S_b S_c) (MAR50.53 (2)); DERIVED fixture SA-E (87.178); two buckets with gamma 1 and same-sign S SHALL give the sum of their K_b; with no hedge R SHALL play no role; K SHALL be non-negative. | `tests/regulatory/test_sa_cva.py::test_cross_bucket_and_invariants` |
| RV-2004 | THE `sa_cva_capital` SHALL return capital = K_delta + K_vega, each the simple sum of the risk classes' K (MAR50.42 to 50.45), with no vega for counterparty credit spread (MAR50.45, 50.63), no curvature and no default risk charge (MAR50.27), the interest-rate delta by currency class (the specified currencies with five tenors and inflation, other currencies one parallel factor at 1.58%), flat vega risk weights (100%, 78% for large-cap equity), and RWA = 12.5 x capital (MAR50.1); it SHALL refuse a counterparty credit spread vega, an unknown risk class and a non-finite sensitivity, naming them. | `tests/regulatory/test_sa_cva.py::test_capital_composition_and_refusals` |
| RV-2005 | THE module docstring and its docs page SHALL state that the CVA sensitivities are the caller's (MAR50.31 to MAR50.36, CAP50), that index look-through (MAR50.49, 50.50) is the caller's, and the single-factor reading of RV-2002. | `tests/regulatory/test_sa_cva.py::test_stated_limits` |

## Default risk charge, non-securitisations (`regulatory/drc.py`, MAR22.1 to MAR22.26)

| ID | Requirement | Gate |
|---|---|---|
| RV-2006 | THE module SHALL hold the LGDs 100%, 75% and 25% (MAR22.12), the one-year horizon and the three-month maturity floor (MAR22.15, 22.18), the three buckets (MAR22.22) and the Table 2 risk weights (AAA 0.5%, AA 2%, A 3%, BBB 6%, BB 15%, B 30%, CCC 50%, unrated 15%, defaulted 100%; MAR22.24) as regulatory parameters naming their paragraphs. | `tests/regulatory/test_drc.py::test_parameters_pinned_to_mar22` |
| RV-2007 | THE `gross_jtd` SHALL be max(LGD x notional + P&L, 0) for a long and min(LGD x notional + P&L, 0) for a short (MAR22.11), weighted by min(1, max(0.25, maturity in years)) (MAR22.15, 22.18), zero when the legal terms allow unwinding (MAR22.13); the Basel FAQ under MAR22.15 and 22.16 (-10 and +10 on three-month maturities) SHALL give a net JTD of 0, and DERIVED fixture DRC-B 75, and 37.5 at six months. | `tests/regulatory/test_drc.py::test_gross_jtd_formula_and_the_basel_faq` |
| RV-2008 | THE `net_jtd` SHALL offset per obligor a short only where it has the same or lower seniority relative to the long (MAR22.19: "a short exposure in an equity may offset a long exposure in a bond, but a short exposure in a bond cannot offset a long exposure in the equity"), with positions under one year weighted by their maturity over the one-year horizon (MAR22.19), returning net long and net short per obligor and rating category. | `tests/regulatory/test_drc.py::test_offsetting_by_seniority` |
| RV-2009 | THE `hedge_benefit_ratio` SHALL be the net long sum over the net long plus the absolute net short sum (MAR22.23), defined as 0 when both are 0 (the text leaves it undefined; the bucket's charge is then 0; stated in the docstring); THE `bucket_drc` SHALL be max(sum RW x net long - HBR x sum RW x |net short|, 0) (MAR22.25), DERIVED fixtures DRC-C 1.0 and DRC-D 0; THE `drc_non_securitisation` SHALL be the plain sum of the buckets (MAR22.26, DRC-E 3.0) with no diversification against securitisations (MAR22.4). | `tests/regulatory/test_drc.py::test_hbr_bucket_and_total` |

## Residual risk add-on (`regulatory/rrao.py`, MAR23.1 to MAR23.8)

| ID | Requirement | Gate |
|---|---|---|
| RV-2010 | THE `rrao` SHALL be 1.0% of the simple sum of the gross notionals with an exotic underlying plus 0.1% of those with other residual risks (MAR23.8), with no offsetting of long and short (RRAO-A 3.0 from 200 and 1,000; RRAO-C); THE `classify_instrument` SHALL encode only the exclusions of MAR23.7 (a back-to-back pair excluded; a listed or centrally cleared instrument excluded from other residual risks and included when its underlying is exotic, RRAO-B), the instrument-level judgement of MAR23.3 to 23.6 staying the caller's, stated in the docstring. | `tests/regulatory/test_rrao.py::test_rrao_formula_and_exclusions` |
