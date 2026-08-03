"""CAUSAL volatility forecasts -- the only vol a tradable exit policy may use.

Every function here maps a history of returns UP TO time t into a forecast
usable AT time t. EWMA is the only estimator implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gamma_exit.conventions import TRADING_DAYS_PER_YEAR as TRADING_DAYS


def ewma_vol(closes: pd.Series, lam: float = 0.94, annualize: bool = True) -> pd.Series:
    """RiskMetrics-style EWMA vol series.

    The value at index t uses returns up to and INCLUDING t, so it is a valid
    input for a decision taken at t (decisions act on the close).
    """
    rets = np.log(closes / closes.shift(1))
    var = rets.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol
