"""Regime tagging: label each position by what its window actually did.

All tags are EX-POST descriptions used for slicing results (which regimes a
policy wins in) -- they are never inputs to causal policies. The vol tag is
the study's core axis: did the position's window realize more or less vol
than was paid for at entry?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRIFT_FLAT_BAND = 0.10  # |annualized window drift| below this = "flat"


def tag_regimes(results: pd.DataFrame) -> pd.DataFrame:
    """Adds `vol_regime` and `drift_regime` columns (copy, input untouched).

    vol_regime:   rv_above_iv / rv_below_iv  (realized window vol vs entry IV)
    drift_regime: bull / bear / flat  (annualized window drift vs a +/-10% band)
    """
    out = results.copy()
    out["vol_regime"] = np.where(
        out["realized_vol_window"] > out["entry_iv"], "rv_above_iv", "rv_below_iv"
    )
    ann_drift = np.log1p(out["window_return"]) / (out["dte_days"] / 252.0)
    out["drift_regime"] = np.select(
        [ann_drift > DRIFT_FLAT_BAND, ann_drift < -DRIFT_FLAT_BAND],
        ["bull", "bear"],
        default="flat",
    )
    return out


def summarize_by_regime(results: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Mean P&L per (regime, policy) at each cost level -- the paper's
    Table-4-style scenario breakdown, computed on tagged real results."""
    tagged = results if regime_col in results.columns else tag_regimes(results)
    return tagged.pivot_table(
        index=["cost_level", regime_col], columns="policy", values="pnl", aggfunc="mean"
    )
