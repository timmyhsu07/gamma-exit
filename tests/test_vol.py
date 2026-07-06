"""Decision 8A: the vol estimators are the input signals for every causal
policy and for P&L attribution -- verify they compute the right number, using
the same trick as the Milestone 1 gate: synthetic GBM with KNOWN volatility.
"""

import numpy as np
import pandas as pd
import pytest

from gamma_exit.conventions import TRADING_DAYS_PER_YEAR
from gamma_exit.pnl.engine import simulate_gbm_paths
from gamma_exit.vol.forecast import ewma_vol
from gamma_exit.vol.realized import close_to_close, parkinson

SIGMA = 0.20


def _daily_closes(n_days: int, sigma: float = SIGMA, seed: int = 5) -> pd.Series:
    t_years = n_days / TRADING_DAYS_PER_YEAR
    path = simulate_gbm_paths(100.0, 0.05, sigma, t_years, n_days, 1, seed)[0]
    return pd.Series(path)


class TestCloseToClose:
    def test_recovers_known_gbm_sigma(self):
        n = 10_000
        est = close_to_close(_daily_closes(n))
        # sampling sd of a vol estimate ~ sigma / sqrt(2n)
        tol = 5 * SIGMA / np.sqrt(2 * n)
        assert est == pytest.approx(SIGMA, abs=tol)

    def test_annualization_is_sqrt_252(self):
        closes = _daily_closes(500)
        daily = close_to_close(closes, annualize=False)
        assert close_to_close(closes) == pytest.approx(daily * np.sqrt(252))

    def test_too_short_window_is_nan(self):
        assert np.isnan(close_to_close(pd.Series([100.0])))
        assert np.isnan(close_to_close(pd.Series([100.0, 101.0])))  # 1 return


class TestParkinson:
    def test_recovers_known_sigma_within_discretization_bias(self):
        # build daily high/low from a fine intraday grid; with m=40 intraday
        # steps the observed range under-samples the true range slightly, so
        # accept [0.85, 1.03] * sigma
        n_days, m = 1500, 40
        t_years = n_days / TRADING_DAYS_PER_YEAR
        path = simulate_gbm_paths(100.0, 0.03, SIGMA, t_years, n_days * m, 1, seed=9)[0]
        days = path[1:].reshape(n_days, m)
        highs = pd.Series(days.max(axis=1))
        lows = pd.Series(days.min(axis=1))
        est = parkinson(highs, lows)
        assert 0.85 * SIGMA < est < 1.03 * SIGMA


class TestEwma:
    def test_long_run_level_matches_known_sigma(self):
        vols = ewma_vol(_daily_closes(4000))
        # E[EWMA variance] = sigma^2; the mean over a long window pins it down
        assert float(vols.iloc[500:].mean()) == pytest.approx(SIGMA, abs=0.02)

    def test_causality_prefix_invariance(self):
        """THE property the causal/ex-post split rests on: the forecast at t
        must not change when future rows are appended."""
        closes = _daily_closes(600)
        full = ewma_vol(closes)
        for t in (50, 300, 599):
            prefix = ewma_vol(closes.iloc[: t + 1])
            assert float(prefix.iloc[-1]) == pytest.approx(float(full.iloc[t]), rel=1e-12)

    def test_annualization_is_sqrt_252(self):
        closes = _daily_closes(300)
        daily = ewma_vol(closes, annualize=False)
        ann = ewma_vol(closes)
        ratio = (ann / daily).dropna()
        assert np.allclose(ratio, np.sqrt(252))
