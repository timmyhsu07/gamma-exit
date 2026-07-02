"""yfinance provider: free path for prototyping the schema/cache pipeline.

Limitations to keep in mind (why this is a prototype path, not the study's
final data source):
- Chains are CURRENT snapshots only; yfinance has no historical chains, so
  real replay backtests need ThetaData / OptionMetrics.
- Yahoo's impliedVolatility column is frequently stale or nonsensical; we
  store it as provider_iv but recompute IV from bid/ask mids ourselves.
- Quotes are 15-min delayed and can be crossed/zero on illiquid strikes;
  OptionRecord.mid already returns None for those.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

from gamma_exit.data.providers.base import Provider
from gamma_exit.data.schema import (
    OptionRecord,
    UnderlyingBar,
    bars_to_frame,
    options_to_frame,
)


def _opt_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def _opt_int(v) -> int | None:
    f = _opt_float(v)
    return int(f) if f is not None else None


class YFinanceProvider(Provider):
    name = "yfinance"

    def fetch_chain(self, underlying: str, max_expiries: int | None = None) -> pd.DataFrame:
        ticker = yf.Ticker(underlying)
        asof = datetime.now(timezone.utc)
        spot = _opt_float(ticker.history(period="1d")["Close"].iloc[-1])

        expiries = ticker.options
        if max_expiries is not None:
            expiries = expiries[:max_expiries]

        records: list[OptionRecord] = []
        for expiry_str in expiries:
            chain = ticker.option_chain(expiry_str)
            expiry = date.fromisoformat(expiry_str)
            for kind, frame in (("call", chain.calls), ("put", chain.puts)):
                for row in frame.itertuples(index=False):
                    records.append(
                        OptionRecord(
                            asof=asof,
                            provider=self.name,
                            underlying=underlying,
                            expiry=expiry,
                            strike=float(row.strike),
                            kind=kind,
                            bid=_opt_float(getattr(row, "bid", None)),
                            ask=_opt_float(getattr(row, "ask", None)),
                            last=_opt_float(getattr(row, "lastPrice", None)),
                            volume=_opt_int(getattr(row, "volume", None)),
                            open_interest=_opt_int(getattr(row, "openInterest", None)),
                            provider_iv=_opt_float(getattr(row, "impliedVolatility", None)),
                            underlying_price=spot,
                        )
                    )
        return options_to_frame(records)

    def fetch_underlying(self, ticker_sym: str, start: date, end: date) -> pd.DataFrame:
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(start=str(start), end=str(end), auto_adjust=False)
        records = [
            UnderlyingBar(
                provider=self.name,
                ticker=ticker_sym,
                date=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                adj_close=_opt_float(row.get("Adj Close")),
                volume=_opt_int(row.get("Volume")),
                dividend=float(row.get("Dividends", 0.0) or 0.0),
            )
            for idx, row in hist.iterrows()
        ]
        return bars_to_frame(records)
