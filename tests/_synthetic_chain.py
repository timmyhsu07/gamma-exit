"""Synthetic option-chain quotes in the canonical replay format.

Generates what a perfect data vendor would have recorded if the world were
GBM and the option were marked at BS(sigma_iv): consecutive business days,
mid = model price, bid/ask = mid -/+ half the spread. Lets the M3 pipeline be
validated end-to-end (loader shape -> IV solve -> deltas -> core) against the
M1 synthetic engine before any paid historical data exists.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from gamma_exit.conventions import TRADING_DAYS_PER_YEAR
from gamma_exit.pnl.engine import simulate_gbm_paths
from gamma_exit.pnl.replay import PositionSpec, QUOTE_COLUMNS
from gamma_exit.pricing.black_scholes import bs_price

START = date(2026, 1, 5)  # a Monday


def synthetic_quotes(
    n_days: int = 126,
    s0: float = 100.0,
    k: float = 100.0,
    kind: str = "call",
    sigma_iv=0.20,  # scalar, or array of length n_days+1 (per-day marking vol)
    sigma_real: float = 0.30,
    mu: float = 0.06,
    r: float = 0.02,
    q: float = 0.0,
    spread: float = 0.02,
    seed: int = 314,
):
    """Returns (quotes frame, PositionSpec entry->expiry, path row vector)."""
    dates = pd.bdate_range(START, periods=n_days + 1).date
    t_years = n_days / TRADING_DAYS_PER_YEAR
    path = simulate_gbm_paths(s0, mu, sigma_real, t_years, n_days, 1, seed)[0]

    tte = (n_days - np.arange(n_days + 1)) / TRADING_DAYS_PER_YEAR
    iv = np.broadcast_to(np.asarray(sigma_iv, dtype=float), (n_days + 1,))
    mid = np.array(
        [bs_price(path[j], k, tte[j], r, iv[j], q, kind) for j in range(n_days + 1)]
    )

    quotes = pd.DataFrame(
        {
            "date": dates,
            "spot": path,
            "bid": mid - 0.5 * spread,
            "ask": mid + 0.5 * spread,
            "mid": mid,
            "volume": 1000,
            "open_interest": 5000,
        }
    )[QUOTE_COLUMNS]
    spec = PositionSpec(
        underlying="SYN",
        expiry=dates[-1],
        strike=k,
        kind=kind,  # type: ignore[arg-type]
        entry_date=dates[0],
    )
    return quotes, spec, path[None, :]
