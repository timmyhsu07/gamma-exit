"""Black-Scholes-Merton European option pricing (continuous dividend yield q).

Conventions used across the project:
- T is time to expiry in YEARS (ACT/365 or trading-day fraction decided at the
  data layer; the pricing layer is agnostic).
- sigma is annualized volatility.
- r and q are continuously compounded annual rates.
- kind is "call" or "put".

Edge cases: T <= 0 returns intrinsic value; sigma <= 0 returns the
deterministic forward-discounted intrinsic (the sigma -> 0 limit).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS_T = 1e-12
_EPS_SIGMA = 1e-12


def _validate_kind(kind: str) -> float:
    if kind == "call":
        return 1.0
    if kind == "put":
        return -1.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def d1_d2(s, k, t, r, sigma, q=0.0):
    """Return (d1, d2). Inputs broadcast like numpy arrays.

    Undefined where t <= 0 or sigma <= 0; callers must mask those (bs_price does).
    """
    s, k, t, r, sigma, q = np.broadcast_arrays(
        *map(np.asarray, (s, k, t, r, sigma, q)), subok=False
    )
    t_safe = np.where(t > _EPS_T, t, np.nan)
    sig_safe = np.where(sigma > _EPS_SIGMA, sigma, np.nan)
    sqrt_t = np.sqrt(t_safe)
    d1 = (np.log(s / k) + (r - q + 0.5 * sig_safe**2) * t_safe) / (sig_safe * sqrt_t)
    d2 = d1 - sig_safe * sqrt_t
    return d1, d2


def bs_price(s, k, t, r, sigma, q=0.0, kind: str = "call"):
    """Black-Scholes-Merton price of a European option.

    Returns intrinsic value at t <= 0 and the sigma -> 0 deterministic limit,
    so it is safe to call on whole grids that include expiry.
    """
    phi = _validate_kind(kind)
    s, k, t, r, sigma, q = np.broadcast_arrays(
        *map(np.asarray, (s, k, t, r, sigma, q)), subok=False
    )
    s = s.astype(float)

    intrinsic = np.maximum(phi * (s - k), 0.0)

    # sigma -> 0 limit: discounted deterministic payoff on the forward
    fwd_payoff = np.maximum(phi * (s * np.exp(-q * t) - k * np.exp(-r * t)), 0.0)

    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    price = phi * (
        s * np.exp(-q * t) * norm.cdf(phi * d1) - k * np.exp(-r * t) * norm.cdf(phi * d2)
    )

    out = np.where(t <= _EPS_T, intrinsic, np.where(sigma <= _EPS_SIGMA, fwd_payoff, price))
    return out if out.ndim else float(out)
