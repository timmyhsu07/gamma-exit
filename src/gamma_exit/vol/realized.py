"""EX-POST realized volatility. NOT available to causal exit policies.

These estimators look at a window of returns that has already happened; when
computed over [t0, t1] they are only known at t1. The backtest runner is
responsible for never handing them to a causal policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def close_to_close(closes: pd.Series, annualize: bool = True) -> float:
    """Realized vol of a window of closing prices (std of log returns)."""
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) < 2:
        return float("nan")
    vol = float(rets.std(ddof=1))
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def parkinson(high: pd.Series, low: pd.Series, annualize: bool = True) -> float:
    """Parkinson range estimator; more efficient than close-to-close, biased
    down by discrete sampling of the true range."""
    hl = np.log(high / low) ** 2
    var = float(hl.mean()) / (4.0 * np.log(2.0))
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol
