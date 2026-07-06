"""Implied volatility via bracketed root-finding (Brent).

Returns NaN when the target price violates no-arbitrage bounds or when no
vol in [SIGMA_LO, SIGMA_HI] reproduces it -- callers must treat NaN as
"quote unusable", never as zero.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from gamma_exit.pricing.black_scholes import bs_price

SIGMA_LO = 1e-6
SIGMA_HI = 5.0


def _implied_vol_scalar(price, s, k, t, r, q, kind) -> float:
    if not np.isfinite(price) or t <= 0:
        return np.nan
    lo_price = bs_price(s, k, t, r, SIGMA_LO, q, kind)
    hi_price = bs_price(s, k, t, r, SIGMA_HI, q, kind)
    if not (lo_price <= price <= hi_price):
        return np.nan
    try:
        return brentq(
            lambda sig: bs_price(s, k, t, r, sig, q, kind) - price,
            SIGMA_LO,
            SIGMA_HI,
            xtol=1e-10,
            maxiter=200,
        )
    except ValueError:
        return np.nan


def implied_vol(price, s, k, t, r, q=0.0, kind: str = "call"):
    """Implied vol for scalar or array inputs (broadcast). NaN where unsolvable."""
    b = np.broadcast_arrays(*map(np.asarray, (price, s, k, t, r, q)), subok=False)
    if b[0].ndim == 0:
        p_, s_, k_, t_, r_, q_ = (float(x) for x in b)
        return _implied_vol_scalar(p_, s_, k_, t_, r_, q_, kind)
    out = np.empty(b[0].shape)
    out_flat = out.ravel()  # view: out is freshly allocated C-contiguous
    pf, sf, kf, tf, rf, qf = (x.ravel() for x in b)
    for i in range(pf.size):
        out_flat[i] = _implied_vol_scalar(
            float(pf[i]), float(sf[i]), float(kf[i]), float(tf[i]), float(rf[i]),
            float(qf[i]), kind,
        )
    return out
