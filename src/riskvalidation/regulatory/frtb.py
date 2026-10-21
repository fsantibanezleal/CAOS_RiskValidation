"""The capital aggregation of the sensitivities-based method (SBM) and the liquidity-horizon cascade (MAR21, MAR33;
dossier 01 D.2 and D.3, research addendum R7).

Bucket and cross-bucket aggregation of delta (MAR21.4):

    WS_k = RW_k s_k,   K_b = sqrt(max(0, sum_k WS_k^2 + sum_{k != l} rho_kl WS_k WS_l)),   S_b = sum_{k in b} WS_k
    delta = sqrt(max(0, sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c))

with S_b replaced by max(min(sum WS, K_b), -K_b) when the radicand is negative (MAR21.4(5)(b)). Under the three
correlation scenarios of MAR21.6 (medium: the prescribed value; high: min(1.25 rho, 1); low: max(2 rho - 1, 0.75 rho))
the SBM requirement is the maximum over the scenarios of delta + vega + curvature (MAR21.7). GIRR delta sensitivities
are s_k = (V(r + 0.0001) - V(r)) / 0.0001 (MAR21.19), the change in value for a one basis point shift DIVIDED BY
0.0001, not the PV01 per basis point. The GIRR parameters are those of the research addendum R7 (MAR21.42 Table 1,
MAR21.45, MAR21.46 Table 2, MAR21.50, MAR21.99); for a specified currency the weights may be divided by sqrt(2) at the
bank's discretion (MAR21.44). The cascade is MAR33.4. Not implemented (stated limits): vega (the footnote 24
risk-weight formula is not transcribed), the MAR21.101 curvature cross-bucket step beyond MAR21.5 (UNVERIFIED), the
other six risk classes, DRC, RRAO and SA-CVA. The existing `riskvalidation.market.frtb` holds the backtesting and PLA
statistics; the two modules do not import each other. The engine computes and states no compliance claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from riskvalidation.regulatory.standardised import RegulatoryParameter, _finite, _nonnegative

MAR21_IN_FORCE = "2023-01-01"
MAR33_IN_FORCE = "2023-01-01"
SCENARIOS = ("low", "medium", "high")
SQRT2 = math.sqrt(2.0)

#: MAR21.8: the GIRR delta tenors in years
GIRR_TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
#: MAR21.42 Table 1: risk weights in percent by tenor
_GIRR_RISK_WEIGHTS_PERCENT = (1.7, 1.7, 1.6, 1.3, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1)
#: MAR21.46 Table 2 (same bucket, different tenor, same curve), in percent as printed (addendum R7)
_TABLE_2_PERCENT = (
    (100.0, 97.0, 91.4, 81.1, 71.9, 56.6, 40.0, 40.0, 40.0, 40.0),
    (97.0, 100.0, 97.0, 91.4, 86.1, 76.3, 56.6, 41.9, 40.0, 40.0),
    (91.4, 97.0, 100.0, 97.0, 94.2, 88.7, 76.3, 65.7, 56.6, 41.9),
    (81.1, 91.4, 97.0, 100.0, 98.5, 95.6, 88.7, 82.3, 76.3, 65.7),
    (71.9, 86.1, 94.2, 98.5, 100.0, 98.0, 93.2, 88.7, 84.4, 76.3),
    (56.6, 76.3, 88.7, 95.6, 98.0, 100.0, 97.0, 94.2, 91.4, 86.1),
    (40.0, 56.6, 76.3, 88.7, 93.2, 97.0, 100.0, 98.5, 97.0, 94.2),
    (40.0, 41.9, 65.7, 82.3, 88.7, 94.2, 98.5, 100.0, 99.0, 97.0),
    (40.0, 40.0, 56.6, 76.3, 84.4, 91.4, 97.0, 99.0, 100.0, 98.5),
    (40.0, 40.0, 41.9, 65.7, 76.3, 86.1, 94.2, 97.0, 98.5, 100.0),
)
GIRR_SAME_TENOR_OTHER_CURVE = 0.999  # MAR21.45
GIRR_GAMMA = 0.5  # MAR21.50
GIRR_CURVATURE_SHOCK = 0.017  # MAR21.99: the highest delta risk weight, applied to all tenors

#: MAR33.4 Table 1: the liquidity horizons in days
LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)
LIQUIDITY_HORIZONS_TABLE_1: dict[tuple[str, str], RegulatoryParameter] = {
    (f"j{j}", "liquidity_horizon_days"): RegulatoryParameter(float(lh), "MAR33.4", "MAR33.4 Table 1", MAR33_IN_FORCE)
    for j, lh in enumerate(LIQUIDITY_HORIZONS, start=1)
}
#: MAR33.12 Table 2: the liquidity horizon in days by risk factor class (the row "commodity volatility, 60 or 120
# days" is
#: ambiguous in dossier 01 D.3 and is not transcribed)
_TABLE_2_DAYS = {
    "interest-rate-specified-currencies": 10,
    "interest-rate-other": 20,
    "interest-rate-volatility": 60,
    "credit-spread-sovereign-investment-grade": 20,
    "credit-spread-sovereign-high-yield": 40,
    "credit-spread-corporate-investment-grade": 40,
    "credit-spread-corporate-high-yield": 60,
    "credit-spread-volatility": 120,
    "equity-large-cap": 10,
    "equity-small-cap": 20,
    "equity-large-cap-volatility": 20,
    "equity-small-cap-volatility": 60,
    "fx-specified-pairs": 10,
    "fx-other-pairs": 20,
    "fx-volatility": 40,
    "commodity-energy-and-carbon": 20,
    "commodity-precious-and-non-ferrous-metals": 20,
    "commodity-other": 60,
}
LIQUIDITY_HORIZONS_TABLE_2: dict[tuple[str, str], RegulatoryParameter] = {
    (key, "liquidity_horizon_days"): RegulatoryParameter(float(days), "MAR33.12", "MAR33.12 Table 2", MAR33_IN_FORCE)
    for key, days in _TABLE_2_DAYS.items()
}


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class GirrParameters:
    """The GIRR delta parameters of the research addendum R7 with their paragraphs and the in-force date."""

    tenors: np.ndarray  # years (MAR21.8)
    risk_weights: np.ndarray  # decimals as applied: Table 1, divided by sqrt(2) when `sqrt2_applied`
    risk_weights_table: np.ndarray  # decimals as printed in MAR21.42 Table 1
    correlation: np.ndarray  # (10, 10), MAR21.46 Table 2 as decimals
    same_tenor_other_curve: float  # MAR21.45
    gamma: float  # MAR21.50
    curvature_shock: float  # MAR21.99, divided by sqrt(2) when `sqrt2_applied` (derived, addendum R7)
    curvature_shock_undivided: float  # MAR21.99: 0.017
    sqrt2_applied: bool  # MAR21.44: at the bank's discretion for a specified currency
    in_force: str
    paragraphs: dict[str, str]


def girr_parameters(specified_currency: bool = True) -> GirrParameters:
    """The GIRR delta tenors, risk weights, correlations, gamma and curvature shock (research addendum R7).

    The curvature shock a consumer applies is `curvature_shock_undivided` (0.017): MAR21.99 bases it on "the highest
    prescribed delta risk weight", and MAR21.44's division by sqrt(2) names the delta risk weights of MAR21.42 to
    MAR21.43 only. `curvature_shock` (0.017 / sqrt(2) = 0.012021) is a derived reading, kept exposed and labelled.

    With `specified_currency` the risk weights (and the curvature shock) are divided by sqrt(2) (MAR21.44) and
    `sqrt2_applied` says so; the specified currencies are EUR, USD, GBP, AUD, JPY, SEK, CAD and the bank's domestic
    reporting currency.
    """
    table = np.array(_GIRR_RISK_WEIGHTS_PERCENT) / 100.0
    divisor = SQRT2 if specified_currency else 1.0
    paragraphs = {
        "tenors": "MAR21.8",
        "risk_weights": "MAR21.42 Table 1" + (" divided by sqrt(2), MAR21.44" if specified_currency else ""),
        "correlation": "MAR21.46 Table 2",
        "same_tenor_other_curve": "MAR21.45",
        "gamma": "MAR21.50",
        "curvature_shock": "MAR21.99" + (" divided by sqrt(2), MAR21.44 (derived)" if specified_currency else ""),
    }
    return GirrParameters(
        tenors=_readonly(np.array(GIRR_TENORS)),
        risk_weights=_readonly(table / divisor),
        risk_weights_table=_readonly(table),
        correlation=_readonly(np.array(_TABLE_2_PERCENT) / 100.0),
        same_tenor_other_curve=GIRR_SAME_TENOR_OTHER_CURVE,
        gamma=GIRR_GAMMA,
        curvature_shock=GIRR_CURVATURE_SHOCK / divisor,
        curvature_shock_undivided=GIRR_CURVATURE_SHOCK,
        sqrt2_applied=bool(specified_currency),
        in_force=MAR21_IN_FORCE,
        paragraphs=paragraphs,
    )


def _array(name: str, x: object, ndim: int | None = None, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        a = np.array(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _correlation_under(name: str, rho: object, scenario: str) -> np.ndarray:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario {scenario!r} is not one of low, medium, high (MAR21.6)")
    r = _array(name, rho)
    if np.any(np.abs(r) > 1.0 + 1e-12):
        raise ValueError(f"{name} must lie in [-1, 1]")
    if scenario == "high":
        return np.minimum(1.25 * r, 1.0)
    if scenario == "low":
        return np.maximum(2.0 * r - 1.0, 0.75 * r)
    return r.copy()


def scenario_correlation(rho: np.ndarray, scenario: str) -> np.ndarray:
    """MAR21.6: medium is rho; high is min(1.25 rho, 1); low is max(2 rho - 1, 0.75 rho). Applies to rho and to gamma
    alike."""
    return np.asarray(_correlation_under("rho", rho, scenario), dtype=float)


def _offdiagonal(m: np.ndarray) -> np.ndarray:
    out = np.array(m, dtype=float)
    np.fill_diagonal(out, 0.0)
    return out


def bucket_charge(ws: np.ndarray, rho: np.ndarray, scenario: str = "medium") -> float:
    """MAR21.4: K_b = sqrt(max(0, sum WS_k^2 + sum_{k != l} rho_kl WS_k WS_l)) with rho of the scenario."""
    w = _array("ws", ws, ndim=1)
    n = w.shape[0]
    r = _correlation_under("rho", _array("rho", rho, shape=(n, n)), scenario)
    radicand = float(w @ w + w @ _offdiagonal(r) @ w)
    return math.sqrt(max(0.0, radicand))


@dataclass(frozen=True)
class CrossBucket:
    """The cross-bucket charge and whether the MAR21.4(5)(b) fallback applied (the radicand with the plain S_b was
    negative)."""

    charge: float
    fallback_used: bool


def cross_bucket_charge_detail(
    k_b: np.ndarray, s_b: np.ndarray, gamma: np.ndarray, scenario: str = "medium"
) -> CrossBucket:
    """MAR21.4: the charge of `cross_bucket_charge` and `fallback_used`, true exactly when the radicand with the plain
    S_b is negative and S_b is replaced by max(min(S_b, K_b), -K_b) (MAR21.4(5)(b))."""
    k = _array("k_b", k_b, ndim=1)
    if np.any(k < 0.0):
        raise ValueError("k_b must be at least 0")
    s = _array("s_b", s_b, shape=k.shape)
    g = _offdiagonal(_correlation_under("gamma", _array("gamma", gamma, shape=(k.size, k.size)), scenario))
    radicand = float(k @ k + s @ g @ s)
    fallback = radicand < 0.0
    if fallback:
        s = np.maximum(np.minimum(s, k), -k)
        radicand = float(k @ k + s @ g @ s)
    return CrossBucket(math.sqrt(max(0.0, radicand)), fallback)


def cross_bucket_charge(k_b: np.ndarray, s_b: np.ndarray, gamma: np.ndarray, scenario: str = "medium") -> float:
    """MAR21.4: sqrt(sum K_b^2 + sum_{b != c} gamma_bc S_b S_c), with the MAR21.4(5)(b) replacement of S_b if it is
    negative."""
    return cross_bucket_charge_detail(k_b, s_b, gamma, scenario).charge


@dataclass(frozen=True, eq=False)
class SbmResult:
    """The weighted sensitivities, the bucket charge under low, medium and high, and the charge, their maximum
    (MAR21.7)."""

    weighted: np.ndarray
    k_by_scenario: tuple[float, float, float]
    charge: float


def girr_delta_charge(sensitivities: np.ndarray, tenors: np.ndarray, params: GirrParameters) -> SbmResult:
    """MAR21.4 to MAR21.7 for one GIRR bucket (a single currency): the delta charge of ten tenor sensitivities.

    `sensitivities` is s_k = (V(r + 0.0001) - V(r)) / 0.0001 at each tenor (MAR21.19), money per unit of rate; a PV01
    of 4,500 USD per basis point is s_k = 4.5e7.
    """
    n = len(params.tenors)
    s = _array("sensitivities", sensitivities, shape=(n,))
    t = _array("tenors", tenors, ndim=1)
    if t.shape != params.tenors.shape or not np.allclose(t, params.tenors, rtol=0.0, atol=1e-12):
        raise ValueError(f"tenors must be the GIRR tenors {tuple(params.tenors.tolist())} (MAR21.8)")
    weighted = _readonly(params.risk_weights * s)
    k = tuple(bucket_charge(weighted, params.correlation, name) for name in SCENARIOS)
    return SbmResult(weighted, (k[0], k[1], k[2]), max(k))


def _psi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """psi(x, y) = 0 if both are negative and 1 otherwise (MAR21.5), as a matrix over the pairs."""
    return (~((x < 0.0)[:, None] & (y < 0.0)[None, :])).astype(float)


def curvature_charge(
    cvr_plus: np.ndarray, cvr_minus: np.ndarray, rho: np.ndarray, gamma: np.ndarray, scenario: str = "medium"
) -> float:
    """MAR21.5: the curvature charge of B buckets of K factors, with the psi rule.

    `rho` is the (K, K) DELTA correlation and `gamma` the (B, B) DELTA cross-bucket correlation; both are squared here
    (MAR21.100 and MAR21.101: the curvature correlations are the squares of the delta ones) and then take the
    correlation scenario (MAR21.6). K_b = max(K_b+, K_b-); S_b is the sum of the CVR of the scenario that selected K_b:
    the upward one when K_b+ > K_b-, the downward one when K_b+ < K_b-, and on a tie (K_b+ = K_b-) the upward one if
    sum CVR+ > sum CVR-, otherwise the downward one (MAR21.5(3)(a)(iii)).
    """
    plus = _array("cvr_plus", cvr_plus, ndim=2)
    minus = _array("cvr_minus", cvr_minus, shape=plus.shape)
    n_buckets, n_factors = plus.shape
    delta_rho = _array("rho", rho, shape=(n_factors, n_factors))
    r = _offdiagonal(_correlation_under("rho", delta_rho**2, scenario))
    delta_gamma = _array("gamma", gamma, shape=(n_buckets, n_buckets))
    g = _offdiagonal(_correlation_under("gamma", delta_gamma**2, scenario))
    k_b = np.zeros(n_buckets)
    s_b = np.zeros(n_buckets)
    for b in range(n_buckets):
        roots = []
        for cvr in (plus[b], minus[b]):
            radicand = float(np.sum(np.maximum(cvr, 0.0) ** 2) + cvr @ (r * _psi(cvr, cvr)) @ cvr)
            roots.append(math.sqrt(max(0.0, radicand)))
        sum_plus, sum_minus = float(np.sum(plus[b])), float(np.sum(minus[b]))
        up = roots[0] > roots[1] or (roots[0] == roots[1] and sum_plus > sum_minus)  # MAR21.5(3)(a)(iii)
        k_b[b] = max(roots)
        s_b[b] = sum_plus if up else sum_minus
    radicand = float(k_b @ k_b + s_b @ (g * _psi(s_b, s_b)) @ s_b)
    return math.sqrt(max(0.0, radicand))


@dataclass(frozen=True)
class SbmCapital:
    """The SBM requirement: the sum of the three measures per scenario (low, medium, high), their maximum and its
    scenario."""

    by_scenario: tuple[float, float, float]
    capital: float
    scenario: str


def _by_scenario(label: str, values: dict[str, float]) -> dict[str, float]:
    missing = [s for s in SCENARIOS if s not in values]
    if missing:
        raise ValueError(f"{label} must hold the scenarios low, medium and high; missing {missing[0]!r}")
    return {s: _nonnegative(label, values[s]) for s in SCENARIOS}


def sbm_capital(delta: dict[str, float], vega: dict[str, float], curvature: dict[str, float]) -> SbmCapital:
    """MAR21.7: the maximum over the three scenarios of delta + vega + curvature, each summed over the risk classes."""
    d = _by_scenario("delta", delta)
    v = _by_scenario("vega", vega)
    c = _by_scenario("curvature", curvature)
    totals = tuple(d[s] + v[s] + c[s] for s in SCENARIOS)
    best = max(range(3), key=lambda i: totals[i])
    return SbmCapital((totals[0], totals[1], totals[2]), totals[best], SCENARIOS[best])


def liquidity_cascade_es(es_t_full: float, es_t_by_horizon: dict[int, float], base_horizon: int = 10) -> float:
    """MAR33.4: sqrt(ES_T(P)^2 + sum_{j >= 2} (ES_T(P, j) sqrt((LH_j - LH_{j-1}) / T))^2) with the Table 1 horizons.

    `es_t_by_horizon` maps a liquidity horizon in days (10, 20, 40, 60, 120) to ES_T(P, j), the ES shocking only the
    factors with horizon at least LH_j; a missing horizon contributes nothing, and an entry for 10 days is ES_T(P)
    itself and must equal `es_t_full`.
    """
    full = _nonnegative("es_t_full", es_t_full)
    t = _finite("base_horizon", base_horizon)
    if t <= 0.0:
        raise ValueError(f"base_horizon must be positive, got {base_horizon!r}")
    for lh in es_t_by_horizon:
        if lh not in LIQUIDITY_HORIZONS:
            raise ValueError(
                f"es_t_by_horizon has the horizon {lh!r}, not one of {LIQUIDITY_HORIZONS} days (MAR33.4 Table 1)"
            )
    values = {lh: _nonnegative("es_t_by_horizon", es_t_by_horizon[lh]) for lh in es_t_by_horizon}
    if 10 in values and not math.isclose(values[10], full, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("es_t_by_horizon[10] is ES_T(P) and must equal es_t_full")
    total = full * full
    for j in range(1, len(LIQUIDITY_HORIZONS)):
        es_j = values.get(LIQUIDITY_HORIZONS[j], 0.0)
        total += (es_j * math.sqrt((LIQUIDITY_HORIZONS[j] - LIQUIDITY_HORIZONS[j - 1]) / t)) ** 2
    return math.sqrt(total)

__all__ = [
    "CrossBucket",
    "GirrParameters",
    "SbmCapital",
    "SbmResult",
    "bucket_charge",
    "cross_bucket_charge",
    "cross_bucket_charge_detail",
    "curvature_charge",
    "girr_delta_charge",
    "girr_parameters",
    "liquidity_cascade_es",
    "sbm_capital",
    "scenario_correlation",
]
