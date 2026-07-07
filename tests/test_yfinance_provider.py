"""Offline fixture tests for the yfinance provider.

A canned Yahoo payload (frozen from a real pull, trimmed) exercises the
normalization boundary without network: NaN handling, zero-bid mids, spot
attachment, and the fail-loud empty-history guards. When yfinance changes
its payload shape, these tests name the break instead of a backtest quietly
losing quotes.
"""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gamma_exit.data.providers.yfinance_provider as yfp
from gamma_exit.data.schema import OPTION_COLUMNS

EXPIRY = "2026-08-21"


def _calls_frame() -> pd.DataFrame:
    # column names as Yahoo sends them (via yfinance), incl. the warts:
    # NaN volume, zero bid on an illiquid strike, stale lastPrice
    return pd.DataFrame(
        {
            "contractSymbol": ["SPY260821C00600000", "SPY260821C00700000"],
            "strike": [600.0, 700.0],
            "lastPrice": [34.1, 0.9],
            "bid": [33.8, 0.0],
            "ask": [34.4, 1.4],
            "volume": [1200.0, np.nan],
            "openInterest": [5400.0, 12.0],
            "impliedVolatility": [0.191, 0.243],
        }
    )


def _puts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": ["SPY260821P00600000"],
            "strike": [600.0],
            "lastPrice": [12.2],
            "bid": [12.0],
            "ask": [12.5],
            "volume": [800.0],
            "openInterest": [3100.0],
            "impliedVolatility": [0.205],
        }
    )


def _history_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02")])
    return pd.DataFrame(
        {
            "Open": [618.0, 620.5],
            "High": [621.0, 623.9],
            "Low": [616.2, 619.8],
            "Close": [620.0, 622.3],
            "Adj Close": [620.0, 622.3],
            "Volume": [50_000_000, np.nan],
            "Dividends": [0.0, 1.83],
        },
        index=idx,
    )


class FakeTicker:
    def __init__(self, sym: str, history: pd.DataFrame | None = None, options=(EXPIRY,)):
        self.sym = sym
        self._history = _history_frame() if history is None else history
        self.options = tuple(options)

    def history(self, **kwargs) -> pd.DataFrame:
        return self._history

    def option_chain(self, expiry: str) -> SimpleNamespace:
        assert expiry == EXPIRY
        return SimpleNamespace(calls=_calls_frame(), puts=_puts_frame())


@pytest.fixture
def provider(monkeypatch):
    def factory(*, history=None, options=(EXPIRY,)):
        monkeypatch.setattr(
            yfp.yf, "Ticker", lambda sym: FakeTicker(sym, history=history, options=options)
        )
        return yfp.YFinanceProvider()

    return factory


class TestFetchChain:
    def test_normalizes_to_canonical_frame(self, provider):
        df = provider().fetch_chain("SPY")
        assert list(df.columns) == OPTION_COLUMNS
        assert len(df) == 3  # 2 calls + 1 put
        assert set(df["kind"]) == {"call", "put"}
        assert (df["provider"] == "yfinance").all()
        # spot from the last close is attached to every row
        assert (df["underlying_price"] == 622.3).all()
        assert (df["expiry"] == pd.Timestamp("2026-08-21")).all()

    def test_quote_warts_normalized(self, provider):
        df = provider().fetch_chain("SPY").set_index(["kind", "strike"])
        liquid = df.loc[("call", 600.0)]
        assert liquid["mid"] == pytest.approx(0.5 * (33.8 + 34.4))
        assert liquid["volume"] == 1200
        illiquid = df.loc[("call", 700.0)]
        assert pd.isna(illiquid["mid"])  # zero bid -> no tradable mid
        assert pd.isna(illiquid["volume"])  # NaN volume -> None, not 0
        assert illiquid["provider_iv"] == pytest.approx(0.243)  # advisory only

    def test_empty_history_raises_with_ticker_context(self, provider):
        p = provider(history=_history_frame().iloc[:0])
        with pytest.raises(ValueError, match="SPYY"):
            p.fetch_chain("SPYY")

    def test_no_expiries_raises(self, provider):
        p = provider(options=())
        with pytest.raises(ValueError, match="expiries"):
            p.fetch_chain("SPY")


class TestFetchUnderlying:
    def test_normalizes_bars(self, provider):
        bars = provider().fetch_underlying("SPY", date(2026, 7, 1), date(2026, 7, 3))
        assert len(bars) == 2
        assert bars["close"].tolist() == [620.0, 622.3]
        assert bars["dividend"].tolist() == [0.0, 1.83]
        assert pd.isna(bars["volume"].iloc[1])  # NaN volume -> None

    def test_empty_history_raises_with_range_context(self, provider):
        p = provider(history=_history_frame().iloc[:0])
        with pytest.raises(ValueError, match=r"BAD.*2026-07-01"):
            p.fetch_underlying("BAD", date(2026, 7, 1), date(2026, 7, 3))


class TestOptionalCasts:
    def test_opt_float(self):
        assert yfp._opt_float("3.5") == 3.5
        assert yfp._opt_float(np.nan) is None
        assert yfp._opt_float(None) is None
        assert yfp._opt_float("abc") is None

    def test_opt_int(self):
        assert yfp._opt_int(1200.0) == 1200
        assert yfp._opt_int(np.nan) is None
